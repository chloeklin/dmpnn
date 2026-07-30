"""Octamer K=1 vs K=16 analysis for the clustered B-heldout split.

Loads predictions from:
    predictions/octamer_k1/ea_ip_lomo_b_clustered/   (K=1, filenames end in __k1)
    predictions/regen_v1/ea_ip_lomo_b_clustered/     (K=16 comparators)

For hpg_hier_octamer, averages the three seeds at the prediction level and reports
every metric on three row subsets (random only; block + alternating; all rows) using
poly_type from data/ea_ip.csv.  S and D fold groups are reported separately and are
never pooled.

Outputs (stem is _octamer_k1_r3_results):
    *_individual_runs.csv
    *_cells.csv
    *_comparisons.csv
    *_fold_composition.csv
    *_results.md

Usage:
    python scripts/python/analyze_octamer_k1.py [--partial]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation.metrics import compute_copolymer_metrics
from analyze_regen_v1 import (
    markdown,
    holm_adjust,
    flag_undertrained,
    UNDERTRAINED_BEST_EPOCH_THRESHOLD,
)
from analyze_regen_v1_r3 import null_floors, scaffold_structure

DATA_PATH = ROOT / "data" / "ea_ip.csv"
K1_PRED_DIR = ROOT / "predictions" / "octamer_k1" / "ea_ip_lomo_b_clustered"
K16_PRED_DIR = ROOT / "predictions" / "regen_v1" / "ea_ip_lomo_b_clustered"
OUTPUT_PATH = ROOT / "analysis" / "model_diagnostics" / "_octamer_k1_r3_results.md"

SPLIT = "monomer_b_heldout_clustered"
MODEL = "hpg_hier_octamer"
TARGETS = {"EA": "EA_vs_SHE_eV", "IP": "IP_vs_SHE_eV"}
SETTINGS = ("k1", "k16")
SEEDS = (42, 43, 44)
FOLDS = tuple(range(9))
METRICS = (
    "group_mean_r2",
    "delta_r2",
    "ordering",
    "overall_r2",
    "mae",
    "rmse",
    "group_mean_rmse",
    "mean_signed_bias",
    "compression_ratio",
)
COMPARISON_METRICS = ("group_mean_r2", "delta_r2", "ordering", "overall_r2", "mae", "rmse")
ROW_SETS = ("all", "random", "block_alternating")
ROW_SET_FILTERS: dict[str, tuple[str, ...] | None] = {
    "all": None,
    "random": ("random",),
    "block_alternating": ("block", "alternating"),
}
EPOCH_CAP = 100


def prediction_path(setting: str, target: str, fold: int, seed: int) -> Path:
    root = K1_PRED_DIR if setting == "k1" else K16_PRED_DIR
    suffix = "__k1" if setting == "k1" else ""
    return root / f"ea_ip__{TARGETS[target]}__{MODEL}__{SPLIT}__fold{fold}__s{seed}{suffix}.npz"


def _empty_metrics() -> dict:
    return {metric: np.nan for metric in METRICS}


def _nan_group_means() -> pd.DataFrame:
    return pd.DataFrame(columns=["group", "y_true", "y_pred"])


def compute_rowset_metrics(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    global_indices: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    """Wrap compute_copolymer_metrics so single-architecture subsets return NaN group metrics."""
    if len(global_indices) == 0:
        return _empty_metrics(), _nan_group_means()
    try:
        return compute_copolymer_metrics(df, y_true, y_pred, global_indices)
    except ValueError:
        yt = np.asarray(y_true, dtype=np.float64)
        yp = np.asarray(y_pred, dtype=np.float64)
        return {
            "group_mean_r2": np.nan,
            "delta_r2": np.nan,
            "ordering": np.nan,
            "overall_r2": float(r2_score(yt, yp)),
            "mae": float(mean_absolute_error(yt, yp)),
            "rmse": float(np.sqrt(np.mean((yp - yt) ** 2))),
            "group_mean_rmse": np.nan,
            "mean_signed_bias": float((yp - yt).mean()),
            "compression_ratio": np.nan,
        }, _nan_group_means()


def load_run(
    df: pd.DataFrame,
    path: Path,
    setting: str,
    target: str,
    fold: int,
    seed: int,
) -> tuple[dict, dict, str | None]:
    with np.load(path, allow_pickle=True) as archive:
        y_true = archive["y_true"].astype(float).ravel()
        y_pred = archive["y_pred"].astype(float).ravel()
        y_pred_final = archive["y_pred_final"].astype(float).ravel()
        indices = archive["test_indices"].astype(int).ravel()
        split_hash = (
            str(archive["split_indices_sha256"].item())
            if "split_indices_sha256" in archive.files
            else None
        )

    if y_pred_final.shape != y_pred.shape:
        raise AssertionError(f"Missing or mismatched y_pred_final in {path}")

    sidecar = json.loads(path.with_suffix(".config.json").read_text())
    if sidecar["epochs_actually_run"] <= 0 or sidecar["wall_time_seconds"] <= 60:
        raise AssertionError(f"Run does not look trained: {path}")

    n_random_samples = (sidecar.get("cli_args") or {}).get("n_random_samples")
    expected = 1 if setting == "k1" else 16
    if n_random_samples is not None and int(n_random_samples) != expected:
        raise AssertionError(
            f"{path} has n_random_samples={n_random_samples}, expected {expected} for {setting}"
        )

    row: dict = {
        "setting": setting,
        "model": MODEL,
        "target": target,
        "fold": fold,
        "seed": seed,
    }
    payloads: dict = {}
    best_mae_all = np.nan

    for row_set, allowed in ROW_SET_FILTERS.items():
        if allowed is None:
            mask = np.ones(indices.shape, dtype=bool)
        else:
            mask = df.iloc[indices]["poly_type"].astype(str).isin(allowed).to_numpy()
        yt, yp, ind = y_true[mask], y_pred[mask], indices[mask]
        metrics, _ = compute_rowset_metrics(df, yt, yp, ind)
        row[f"n_test_rows_{row_set}"] = int(ind.size)
        for metric in METRICS:
            row[f"{metric}_{row_set}"] = metrics[metric]
            if row_set == "all" and metric == "mae":
                best_mae_all = metrics["mae"]
        payloads[row_set] = {
            "y_true": yt,
            "y_pred": yp,
            "indices": ind,
            "split_hash": split_hash,
        }

    final_metrics, _ = compute_rowset_metrics(df, y_true, y_pred_final, indices)
    row["final_mae"] = final_metrics["mae"]
    row["final_minus_best_mae"] = final_metrics["mae"] - best_mae_all
    row["epochs"] = sidecar["epochs_actually_run"]
    row["best_epoch"] = sidecar["best_epoch"]
    row["best_val_loss"] = sidecar["best_val_loss"]
    row["wall_time_seconds"] = sidecar["wall_time_seconds"]
    row["reached_epoch_cap"] = sidecar["best_epoch"] >= EPOCH_CAP
    row["n_random_samples"] = n_random_samples

    return row, payloads, split_hash


def build_cells(detail: pd.DataFrame, arrays: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Three-seed averaged metrics per (setting, target, fold, row_set)."""
    cell_rows = []
    for setting in SETTINGS:
        for target in TARGETS:
            for fold in FOLDS:
                subset = detail[
                    (detail.setting == setting)
                    & (detail.target == target)
                    & (detail.fold == fold)
                ]
                if subset.empty:
                    continue
                for row_set in ROW_SETS:
                    payloads = [
                        arrays[(setting, target, fold, seed)][row_set]
                        for seed in subset.seed.tolist()
                    ]
                    first = payloads[0]
                    if any(
                        not np.array_equal(first["y_true"], p["y_true"])
                        or not np.array_equal(first["indices"], p["indices"])
                        for p in payloads[1:]
                    ):
                        raise AssertionError(
                            f"Rows differ across seeds for {setting} {target} fold {fold} {row_set}"
                        )
                    averaged = np.mean(
                        np.stack([p["y_pred"] for p in payloads]), axis=0
                    )
                    y_true, indices = first["y_true"], first["indices"]
                    metrics, _ = compute_rowset_metrics(df, y_true, averaged, indices)
                    result = {
                        "setting": setting,
                        "model": MODEL,
                        "target": target,
                        "fold": fold,
                        "row_set": row_set,
                        "n_test_rows": int(indices.size),
                    }
                    for metric in METRICS:
                        result[metric] = metrics[metric]
                        col = f"{metric}_{row_set}"
                        result[f"{metric}_seed_mean"] = float(subset[col].mean())
                        result[f"{metric}_seed_sd"] = float(subset[col].std(ddof=1))
                    cell_rows.append(result)

    cells = pd.DataFrame(cell_rows)
    cells = cells.merge(null_floors(), on=["target", "fold"], validate="many_to_one")

    null_cols = [c for c in ("null_group_mean_r2", "null_overall_r2", "null_mae", "null_rmse") if c in cells.columns]
    for col in null_cols:
        cells[col] = np.where(cells.row_set == "all", cells[col], np.nan)

    cells["beats_null_floor"] = np.where(
        cells.row_set == "all",
        cells.group_mean_r2 > cells.null_group_mean_r2,
        np.nan,
    )
    cells["skill_group_mean"] = np.where(
        cells.row_set == "all",
        (cells.group_mean_r2 - cells.null_group_mean_r2) / (1.0 - cells.null_group_mean_r2),
        np.nan,
    )
    cells["skill_overall"] = np.where(
        cells.row_set == "all",
        (cells.overall_r2 - cells.null_overall_r2) / (1.0 - cells.null_overall_r2),
        np.nan,
    )
    cells["skill_vs_null"] = cells["skill_group_mean"]
    cells["null_floor_headroom_used"] = cells["skill_group_mean"]
    return cells


def _safe_median(series: pd.Series) -> float:
    finite = series.dropna()
    return float(finite.median()) if not finite.empty else np.nan


def build_comparisons(
    cells: pd.DataFrame, folds: tuple[int, ...], row_set: str, label: str
) -> pd.DataFrame:
    """Paired per-fold comparison of K=1 vs K=16 within a fold group and row set."""
    rows = []
    reference = cells[
        (cells.setting == "k16") & (cells.row_set == row_set) & (cells.fold.isin(folds))
    ]
    candidate = cells[
        (cells.setting == "k1") & (cells.row_set == row_set) & (cells.fold.isin(folds))
    ]
    for target in TARGETS:
        ref = reference[reference.target == target].set_index("fold")
        cand = candidate[candidate.target == target].set_index("fold")
        common = ref.index.intersection(cand.index)
        if len(common) == 0:
            continue
        ref = ref.loc[common]
        cand = cand.loc[common]
        for metric in COMPARISON_METRICS:
            differences = cand[metric] - ref[metric]
            better = differences < 0 if metric == "mae" else differences > 0
            worse = differences > 0 if metric == "mae" else differences < 0
            wins = int(better.sum())
            losses = int(worse.sum())
            non_ties = wins + losses
            p_value = float(binomtest(wins, non_ties, 0.5).pvalue) if non_ties else 1.0
            row = {
                "fold_group": label,
                "row_set": row_set,
                "n_folds": len(common),
                "model": MODEL,
                "setting": "k1",
                "reference_setting": "k16",
                "target": target,
                "metric": metric,
                "median_paired_difference": _safe_median(differences),
                "wins": wins,
                "losses": losses,
                "exact_sign_p": p_value,
                "min_attainable_p": (
                    float(binomtest(len(common), len(common), 0.5).pvalue)
                    if len(common)
                    else np.nan
                ),
            }
            seed_sd_col = f"{metric}_seed_sd"
            if seed_sd_col in cand.columns:
                seed_sd = np.maximum(cand[seed_sd_col], ref[seed_sd_col])
                row["folds_smaller_than_measured_seed_sd"] = int(
                    (differences.abs() < seed_sd).sum()
                )
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    partial = "--partial" in sys.argv
    df = pd.read_csv(DATA_PATH)

    try:
        composition, _, cluster_stats = scaffold_structure()
        rdkit_note = ""
    except ImportError:
        composition, _, cluster_stats = pd.DataFrame(), None, {}
        rdkit_note = (
            "**RDKit unavailable — fold stratification was skipped. "
            "S/D grouping cannot be derived and results are not split by fold group.**"
        )

    if not composition.empty:
        group_s = tuple(
            composition.loc[composition.fold_group == "S_within_scaffold", "fold"].astype(int)
        )
        group_d = tuple(
            composition.loc[composition.fold_group == "D_cross_scaffold", "fold"].astype(int)
        )
    else:
        group_s, group_d = (), ()

    inventory = [
        {
            "setting": setting,
            "target": target,
            "fold": fold,
            "seed": seed,
            "available": prediction_path(setting, target, fold, seed).is_file(),
            "sidecar": prediction_path(setting, target, fold, seed)
            .with_suffix(".config.json")
            .is_file(),
        }
        for setting in SETTINGS
        for target in TARGETS
        for fold in FOLDS
        for seed in SEEDS
    ]
    inventory = pd.DataFrame(inventory)
    inventory["complete"] = inventory.available & inventory.sidecar
    n_complete = int(inventory.complete.sum())

    total_inventory = len(inventory)
    if n_complete < total_inventory and not partial:
        pending = inventory[~inventory.complete]
        report = [
            "# Octamer K=1 vs K=16 — clustered B-heldout",
            "",
            f"## Status: pending — {n_complete}/{total_inventory} runs complete",
            "",
            "Re-run with `--partial` to analyse the subset that has landed.",
            "",
            "## Missing cells",
            "",
            markdown(pending),
            "",
        ]
        OUTPUT_PATH.write_text("\n".join(report))
        print(f"Wrote pending report: {OUTPUT_PATH} ({n_complete}/{total_inventory} runs)")
        return

    if partial:
        inventory = inventory[inventory.complete].reset_index(drop=True)

    run_rows, arrays, split_hashes = [], {}, {}
    for row in inventory.itertuples(index=False):
        path = prediction_path(row.setting, row.target, row.fold, row.seed)
        run_row, payloads, split_hash = load_run(
            df, path, row.setting, row.target, row.fold, row.seed
        )
        run_rows.append(run_row)
        arrays[(row.setting, row.target, row.fold, row.seed)] = payloads
        split_hashes[(row.setting, row.target, row.fold, row.seed)] = split_hash

    detail = pd.DataFrame(run_rows)
    detail = flag_undertrained(detail)

    hash_rows = []
    for target in TARGETS:
        for fold in FOLDS:
            present = [
                split_hashes[k]
                for k in split_hashes
                if k[1] == target and k[2] == fold
            ]
            if len(present) > 1:
                hash_rows.append(
                    {
                        "target": target,
                        "fold": fold,
                        "n_runs": len(present),
                        "identical": len(set(present)) == 1,
                    }
                )
    hash_check = pd.DataFrame(hash_rows)
    if not hash_check.empty and not hash_check.identical.all():
        raise AssertionError(f"Split hashes differ across runs:\n{hash_check[~hash_check.identical]}")

    cells = build_cells(detail, arrays, df)
    if not composition.empty:
        cells = cells.merge(
            composition[["fold", "fold_group"]], on="fold", validate="many_to_one"
        )
    else:
        cells["fold_group"] = np.nan

    comparison_blocks = []
    if group_s:
        for row_set in ROW_SETS:
            comparison_blocks.append(
                (
                    f"S — {row_set}",
                    build_comparisons(cells, group_s, row_set, "S_within_scaffold"),
                )
            )
    if group_d:
        for row_set in ROW_SETS:
            comparison_blocks.append(
                (
                    f"D — {row_set}",
                    build_comparisons(cells, group_d, row_set, "D_cross_scaffold"),
                )
            )

    comparisons = (
        pd.concat([frame for _, frame in comparison_blocks], ignore_index=True)
        if comparison_blocks
        else pd.DataFrame()
    )
    if not comparisons.empty:
        comparisons["holm_p"] = holm_adjust(comparisons.exact_sign_p.tolist())

    cap_counts = (
        detail.groupby("setting")
        .reached_epoch_cap.sum()
        .astype(int)
        .reset_index()
    )
    cap_counts.columns = ["setting", "runs_at_epoch_cap"]
    cap_runs = detail[detail.reached_epoch_cap]

    stem = OUTPUT_PATH.with_suffix("")
    detail.to_csv(stem.with_name(stem.name + "_individual_runs.csv"), index=False)
    cells.to_csv(stem.with_name(stem.name + "_cells.csv"), index=False)
    if not composition.empty:
        composition.to_csv(stem.with_name(stem.name + "_fold_composition.csv"), index=False)
    if not comparisons.empty:
        comparisons.to_csv(stem.with_name(stem.name + "_comparisons.csv"), index=False)

    report = [
        "# Octamer K=1 vs K=16 — clustered B-heldout",
        "",
        "**Convention:** every cell is the mean prediction of three seeds (42, 43, 44), "
        "averaged at the prediction level.  Metrics are reported on three row subsets: "
        "all rows, random rows only, and block + alternating rows only.",
        "",
        f"**Coverage:** {n_complete}/{total_inventory} runs."
        + ("  *(partial analysis)*" if partial else ""),
        "",
        rdkit_note,
        "",
        "## Split structure",
        "",
        markdown(composition) if not composition.empty else rdkit_note or "_Scaffold structure unavailable._",
        "",
        "## Run-quality diagnostic",
        "",
        f"A run is flagged **potentially undertrained** if `best_epoch < {UNDERTRAINED_BEST_EPOCH_THRESHOLD}`. "
        f"A run is flagged **at the 100-epoch cap** if `best_epoch >= {EPOCH_CAP}`. "
        "Both flags are diagnostic only; no run is excluded from any table.",
        "",
        markdown(flag_undertrained(detail).groupby("setting").undertrained.sum().astype(int).reset_index().rename(columns={"undertrained": "undertrained_runs"})),
        "",
        "### Runs that reached the 100-epoch cap",
        "",
        markdown(cap_counts),
        "",
    ]
    if not cap_runs.empty:
        report += [markdown(cap_runs[["setting", "target", "fold", "seed", "best_epoch", "epochs"]]), ""]

    report += [
        "## Split-hash consistency across runs",
        "",
        markdown(hash_check) if not hash_check.empty else "_Only one run per cell; nothing to compare._",
        "",
        "## Three-seed averaged cells",
        "",
        "`delta_r2_seed_sd` is the across-seed SD of delta_r2 for that cell. "
        "`skill_*` and `null_*` are only meaningful for the `all` row set, "
        "because the fold-level null floor is computed on the full test fold.",
        "",
    ]
    for row_set in ROW_SETS:
        report += [f"### Row set: {row_set}", "", markdown(cells[cells.row_set == row_set]), ""]

    report += ["## Paired per-fold comparisons (K=1 minus K=16)", ""]
    if comparisons.empty:
        report += ["_No comparisons computed._", ""]
    else:
        report += [
            "Signed differences are K=1 minus K=16.  Tests are run within fold group and row set. "
            "No pooled comparisons are reported.",
            "",
        ]
        for label, frame in comparison_blocks:
            report += [f"### {label}", "", markdown(frame), ""]

    OUTPUT_PATH.write_text("\n".join(report))
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
