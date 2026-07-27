from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import generate_a_held_out_splits
from scipy.stats import binomtest, linregress, wilcoxon
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / "predictions" / "ea_ip_lomo"
OUT = ROOT / "analysis" / "model_diagnostics"
DATA = ROOT / "data" / "ea_ip.csv"
MODELS = ["hpg_hier", "wdmpnn", "hpg_hier_octamer", "hpg_hier_junction", "hpg_hier_attention"]
TOKENS = {"EA": "EA_vs_SHE_eV", "IP": "IP_vs_SHE_eV"}
TARGETS = {"EA": "EA vs SHE (eV)", "IP": "IP vs SHE (eV)"}


def path(model: str, target: str, fold: int, seed: int) -> Path:
    return PRED / f"ea_ip__{TOKENS[target]}__{model}__monomer_heldout__fold{fold}__s{seed}.npz"


def frame_from_npz(df: pd.DataFrame, npz: np.lib.npyio.NpzFile) -> pd.DataFrame:
    indices = npz["test_indices"].astype(int).ravel()
    frame = df.iloc[indices][["smiles_A", "smiles_B", "fracA", "poly_type"]].copy().reset_index(drop=True)
    frame["y_true"] = npz["y_true"].astype(float).ravel()
    frame["y_pred"] = npz["y_pred"].astype(float).ravel()
    frame["group"] = frame.smiles_A.astype(str) + "||" + frame.smiles_B.astype(str) + "||" + frame.fracA.astype(str)
    return frame


def grouped(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame.groupby("group").poly_type.nunique()
    matched = frame[frame.group.isin(valid[valid >= 2].index)]
    return matched.groupby("group", as_index=False)[["y_true", "y_pred"]].mean()


def ordering(frame: pd.DataFrame) -> float:
    scores = []
    for _, group in frame.groupby("group"):
        if group.poly_type.nunique() < 2:
            continue
        yt, yp = group.y_true.to_numpy(), group.y_pred.to_numpy()
        pairs = [(i, j) for i in range(len(group)) for j in range(i + 1, len(group)) if yt[i] != yt[j]]
        if pairs:
            scores.append(np.mean([(yt[i] - yt[j]) * (yp[i] - yp[j]) > 0 for i, j in pairs]))
    return float(np.mean(scores)) if scores else np.nan


def metric_row(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    gm = grouped(frame)
    matched = frame[frame.group.isin(gm.group)]
    dt = matched.y_true - matched.groupby("group").y_true.transform("mean")
    dp = matched.y_pred - matched.groupby("group").y_pred.transform("mean")
    return {
        "group_mean_r2": r2_score(gm.y_true, gm.y_pred),
        "delta_r2": r2_score(dt, dp),
        "ordering": ordering(matched),
        "overall_r2": r2_score(frame.y_true, frame.y_pred),
        "mae": mean_absolute_error(frame.y_true, frame.y_pred),
        "fold_bias": float((frame.y_pred - frame.y_true).mean()),
        "compression_ratio": float(gm.y_pred.std(ddof=0) / gm.y_true.std(ddof=0)),
        "true_fold_mean": float(frame.y_true.mean()),
        "pred_fold_mean": float(frame.y_pred.mean()),
    }, gm


def holm(pvalues: list[float]) -> list[float]:
    valid = [(i, p) for i, p in enumerate(pvalues) if np.isfinite(p)]
    out = [np.nan] * len(pvalues)
    running = 0.0
    for rank, (idx, value) in enumerate(sorted(valid, key=lambda item: item[1])):
        running = max(running, min(1.0, (len(valid) - rank) * value))
        out[idx] = running
    return out


def null_floor(df: pd.DataFrame, target: str, train_indices: np.ndarray, test_indices: np.ndarray) -> float:
    train = df.iloc[train_indices].copy()
    test = df.iloc[test_indices].copy()
    value = TARGETS[target]
    primary = train.groupby(["smiles_B", "fracA", "poly_type"])[value].mean()
    secondary = train.groupby(["smiles_B", "poly_type"])[value].mean()
    global_mean = train[value].mean()
    pred = []
    for row in test.itertuples(index=False):
        key = (row.smiles_B, row.fracA, row.poly_type)
        fallback = (row.smiles_B, row.poly_type)
        pred.append(primary.loc[key] if key in primary.index else secondary.loc[fallback] if fallback in secondary.index else global_mean)
    temp = test[["smiles_A", "smiles_B", "fracA", "poly_type"]].copy()
    temp["y_true"] = test[value].to_numpy(float)
    temp["y_pred"] = pred
    temp["group"] = temp.smiles_A.astype(str) + "||" + temp.smiles_B.astype(str) + "||" + temp.fracA.astype(str)
    return metric_row(temp)[0]["group_mean_r2"]


def markdown(frame: pd.DataFrame, columns: list[str]) -> str:
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        cells = []
        for col in columns:
            value = row[col]
            cells.append(f"{value:.5f}" if isinstance(value, (float, np.floating)) else str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--reference", default="hpg_hier")
    args = parser.parse_args()
    seeds = [int(item) for item in args.seeds.split(",")]
    df = pd.read_csv(DATA)
    lomao_train, _, lomao_test, _ = generate_a_held_out_splits(
        df.smiles_A.astype(str).values, len(df), seed=42, n_splits=9, protocol="leave_one_A_out"
    )
    inventory, rows, pooled = [], [], []
    for model in MODELS:
        for target in TOKENS:
            group_means = []
            for fold in range(9):
                test_indices = None
                for seed in seeds:
                    source = path(model, target, fold, seed)
                    exists = source.exists()
                    inventory.append({"model": model, "target": target, "fold": fold, "seed": seed, "path": str(source), "available": exists})
                    if not exists:
                        continue
                    z = np.load(source, allow_pickle=True)
                    frame = frame_from_npz(df, z)
                    metrics, gm = metric_row(frame)
                    test_indices = z["test_indices"].astype(int).ravel()
                    if not np.array_equal(test_indices, np.asarray(lomao_test[fold], dtype=int)):
                        raise AssertionError(f"NPZ test indices disagree with fixed LOMO fold {fold}")
                    rows.append({"model": model, "target": target, "fold": fold, "seed": seed, "a_blind_null_group_mean_r2": null_floor(df, target, np.asarray(lomao_train[fold], dtype=int), test_indices), **metrics})
                    group_means.append(gm.assign(fold=fold, seed=seed))
            if group_means:
                all_gm = pd.concat(group_means, ignore_index=True)
                # Per-fold seed averages first, then pooled calculation over averaged group predictions.
                avg_gm = all_gm.groupby(["fold", "group"], as_index=False)[["y_true", "y_pred"]].mean()
                fold_means = pd.DataFrame(rows).query("model == @model and target == @target").groupby("fold", as_index=False)[["true_fold_mean", "pred_fold_mean", "fold_bias"]].mean()
                slope, intercept, _, _, _ = linregress(fold_means.true_fold_mean, fold_means.pred_fold_mean) if len(fold_means) >= 2 else (np.nan, np.nan, np.nan, np.nan, np.nan)
                pooled.append({"model": model, "target": target, "pooled_group_mean_r2": r2_score(avg_gm.y_true, avg_gm.y_pred), "fold_placement_r2": r2_score(fold_means.true_fold_mean, fold_means.pred_fold_mean), "fold_placement_slope": slope, "fold_placement_intercept": intercept, "fold_bias_sd": float(fold_means.fold_bias.std(ddof=0)), "n_available_fold_seed_cells": len(all_gm.groupby(["fold", "seed"]))})
    inv = pd.DataFrame(inventory)
    detail = pd.DataFrame(rows)
    missing = inv[~inv.available]
    if detail.empty:
        raise SystemExit("No requested prediction NPZs were found.")
    seed_fold = detail.groupby(["model", "target", "fold"], as_index=False).agg({**{metric: "mean" for metric in ["group_mean_r2", "delta_r2", "ordering", "overall_r2", "mae", "fold_bias", "compression_ratio", "a_blind_null_group_mean_r2"]}, "seed": "nunique"}).rename(columns={"seed": "n_seeds"})
    spreads = detail.groupby(["model", "target", "fold"], as_index=False)[["group_mean_r2", "delta_r2", "ordering", "overall_r2", "mae", "fold_bias", "compression_ratio", "a_blind_null_group_mean_r2"]].std(ddof=0).add_suffix("_seed_sd").rename(columns={"model_seed_sd": "model", "target_seed_sd": "target", "fold_seed_sd": "fold"})
    fold_summary = seed_fold.merge(spreads, on=["model", "target", "fold"], how="left")
    summary = fold_summary.groupby(["model", "target"], as_index=False).agg({metric: ["median", "mean"] for metric in ["group_mean_r2", "delta_r2", "ordering", "overall_r2", "mae", "fold_bias", "compression_ratio", "a_blind_null_group_mean_r2"]})
    summary.columns = ["_".join(part for part in col if part) for col in summary.columns.to_flat_index()]
    comparisons = []
    for target in TOKENS:
        reference = fold_summary.query("model == @args.reference and target == @target").set_index("fold")
        for model in MODELS:
            if model == args.reference:
                continue
            candidate = fold_summary.query("model == @model and target == @target").set_index("fold")
            common = reference.index.intersection(candidate.index)
            for metric in ["group_mean_r2", "delta_r2", "ordering", "overall_r2", "mae"]:
                diff = candidate.loc[common, metric] - reference.loc[common, metric]
                if metric == "mae":
                    diff = -diff
                wins, losses = int((diff > 0).sum()), int((diff < 0).sum())
                nonzero = wins + losses
                sign_p = binomtest(wins, nonzero, 0.5).pvalue if nonzero else 1.0
                try:
                    wilcox_p = wilcoxon(diff, alternative="two-sided").pvalue
                except ValueError:
                    wilcox_p = np.nan
                comparisons.append({"model": model, "reference": args.reference, "target": target, "metric": metric, "signed_differences_by_fold": json.dumps({str(f): float(diff.loc[f]) for f in common}), "wins": wins, "losses": losses, "ties": 9 - nonzero, "sign_test_p": sign_p, "wilcoxon_p": wilcox_p})
    pooled = pd.DataFrame(pooled)
    comparisons = pd.DataFrame(comparisons)
    comparisons["holm_wilcoxon_p"] = holm(comparisons.wilcoxon_p.tolist())
    OUT.mkdir(parents=True, exist_ok=True)
    detail.to_csv(OUT / "_multiseed_results.csv", index=False)
    try:
        detail.to_parquet(OUT / "_multiseed_results.parquet", index=False)
    except ImportError:
        print("Parquet not written: install pyarrow or fastparquet; CSV output remains available.")
    inv.to_csv(OUT / "_multiseed_inventory.csv", index=False)
    summary.to_csv(OUT / "_multiseed_summary.csv", index=False)
    pooled.to_csv(OUT / "_multiseed_pooled_metrics.csv", index=False)
    comparisons.to_csv(OUT / "_multiseed_comparisons.csv", index=False)
    report = ["# LOMO Multi-Seed Results", "", "Metrics use the existing LOMO group key and matched-architecture definition. Metrics are averaged across seeds within each fold before fold medians/means and paired tests. Missing cells are explicit below. With nine folds, the minimum attainable exact two-sided sign-test p-value is 0.0039. Holm adjustment is across this complete comparison family.", "", "## Inventory", "", f"Available cells: `{int(inv.available.sum())}`; missing cells: `{len(missing)}`.", "", markdown(missing, ["model", "target", "fold", "seed", "path"]) if len(missing) else "No missing cells.", "", "## Per-fold seed-averaged metrics", "", markdown(fold_summary, ["model", "target", "fold", "n_seeds", "group_mean_r2", "delta_r2", "ordering", "overall_r2", "mae", "a_blind_null_group_mean_r2"]), "", "## Across-fold summary", "", markdown(summary, list(summary.columns)), "", "## Pooled placement metrics", "", markdown(pooled, list(pooled.columns)), "", "## Paired comparisons", "", markdown(comparisons, ["model", "reference", "target", "metric", "wins", "losses", "ties", "sign_test_p", "wilcoxon_p", "holm_wilcoxon_p"]), ""]
    (OUT / "_multiseed_results.md").write_text("\n".join(report))


if __name__ == "__main__":
    main()
