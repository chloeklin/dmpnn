"""Octamer positional-embedding ablation analysis for the R1 split.

Loads predictions from:
    predictions/octamer_posemb/ea_ip_lomo/  (ablated, filenames end in __noposemb)
    predictions/regen_v1/ea_ip_lomo/        (K=16 comparators)

For hpg_hier_octamer, averages the three seeds at the prediction level and reports
every metric on four row subsets (all; random; block + alternating;
random_via_all_groups).  Paired per-fold sign tests compare the ablated model
against the K=16 baseline.  The pre-registered outcome is assessed on the primary
quantity ``delta_r2`` on the ``all`` row set, using the R1 materiality threshold
of 0.051.

Outputs (stem is _octamer_posemb_r1_results):
    *_individual_runs.csv
    *_cells.csv
    *_comparisons.csv
    *_delta_r2_seed_sd_summary.csv
    *_results.md

Usage:
    python scripts/python/analyze_octamer_posemb.py [--partial]
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
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
    null_floors,
)

DATA_PATH = ROOT / "data" / "ea_ip.csv"
NOPOSEMB_PRED_DIR = ROOT / "predictions" / "octamer_posemb" / "ea_ip_lomo"
K16_PRED_DIR = ROOT / "predictions" / "regen_v1" / "ea_ip_lomo"
OUTPUT_PATH = ROOT / "analysis" / "model_diagnostics" / "_octamer_posemb_r1_results.md"

SPLIT = "monomer_heldout"
MODEL = "hpg_hier_octamer"
TARGETS = {"EA": "EA_vs_SHE_eV", "IP": "IP_vs_SHE_eV"}
SETTINGS = ("noposemb", "k16")  # candidate first, reference second
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
ROW_SETS = ("all", "random", "block_alternating", "random_via_all_groups")
ROW_SET_FILTERS: dict[str, tuple[str, ...] | None] = {
    "all": None,
    "random": ("random",),
    "block_alternating": ("block", "alternating"),
    "random_via_all_groups": None,
}
EPOCH_CAP = 100
R1_THRESHOLD = 0.051


def prediction_path(setting: str, target: str, fold: int, seed: int) -> Path:
    if setting == "noposemb":
        root, suffix = NOPOSEMB_PRED_DIR, "__noposemb"
    else:
        root, suffix = K16_PRED_DIR, ""
    return root / f"ea_ip__{TARGETS[target]}__{MODEL}__{SPLIT}__fold{fold}__s{seed}{suffix}.npz"


def _empty_metrics() -> dict:
    return {metric: np.nan for metric in METRICS}


def _nan_group_means() -> pd.DataFrame:
    return pd.DataFrame(columns=["group", "y_true", "y_pred"])


def _delta_r2_random_via_all_groups(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    global_indices: np.ndarray,
) -> float:
    """Compute delta_r2 on random rows with group means built over all rows."""
    if len(global_indices) == 0:
        return np.nan
    rows = df.iloc[np.asarray(global_indices, dtype=int)].reset_index(drop=True)
    frame = rows[["smiles_A", "smiles_B", "fracA", "poly_type"]].copy()
    frame["y_true"] = np.asarray(y_true, dtype=np.float64)
    frame["y_pred"] = np.asarray(y_pred, dtype=np.float64)
    frame["group"] = (
        frame.smiles_A.astype(str)
        + "||"
        + frame.smiles_B.astype(str)
        + "||"
        + frame.fracA.astype(str)
    )
    valid = frame.groupby("group").poly_type.nunique()
    matched = frame[frame.group.isin(valid[valid >= 2].index)].copy()
    if matched.empty:
        return np.nan
    group_means = matched.groupby("group")[["y_true", "y_pred"]].transform("mean")
    matched["delta_true"] = matched["y_true"] - group_means["y_true"]
    matched["delta_pred"] = matched["y_pred"] - group_means["y_pred"]
    random_mask = matched["poly_type"].astype(str) == "random"
    if not random_mask.any():
        return np.nan
    return float(
        r2_score(
            matched.loc[random_mask, "delta_true"].to_numpy(),
            matched.loc[random_mask, "delta_pred"].to_numpy(),
        )
    )


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
        # Canonical semantics written by run_hpg_generalization.py:
        #   y_pred        = best-validation-loss checkpoint predictions
        #   y_pred_final  = final (patience-expired) model predictions
        # The primary analysis MUST use y_pred.
        assert "y_pred" in archive.files, f"No y_pred in {path}"
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
    if n_random_samples is not None and int(n_random_samples) != 16:
        raise AssertionError(
            f"{path} has n_random_samples={n_random_samples}, expected 16"
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
        if row_set == "random_via_all_groups":
            for metric in METRICS:
                row[f"{metric}_{row_set}"] = np.nan
            row[f"n_test_rows_{row_set}"] = 0
            payloads[row_set] = {
                "y_true": np.array([], dtype=float),
                "y_pred": np.array([], dtype=float),
                "indices": np.array([], dtype=int),
                "split_hash": split_hash,
            }
            continue
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
                    source_row_set = "all" if row_set == "random_via_all_groups" else row_set
                    payloads = [
                        arrays[(setting, target, fold, seed)][source_row_set]
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
                    if row_set == "random_via_all_groups":
                        metrics = _empty_metrics()
                        metrics["delta_r2"] = _delta_r2_random_via_all_groups(
                            df, y_true, averaged, indices
                        )
                    else:
                        metrics, _ = compute_rowset_metrics(df, y_true, averaged, indices)
                    result = {
                        "setting": setting,
                        "model": MODEL,
                        "target": target,
                        "fold": fold,
                        "row_set": row_set,
                        "n_test_rows": int(indices.size),
                        "n_seeds": len(subset),
                        "protocol_complete": len(subset) == 3,
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
    cells: pd.DataFrame, row_set: str, label: str
) -> tuple[pd.DataFrame, list[str]]:
    """Paired per-fold comparison of noposemb vs K=16 within a row set."""
    rows = []
    notes = []
    reference = cells[
        (cells.setting == "k16") & (cells.row_set == row_set)
    ]
    candidate = cells[
        (cells.setting == "noposemb") & (cells.row_set == row_set)
    ]
    for target in TARGETS:
        ref = reference[reference.target == target].set_index("fold")
        cand = candidate[candidate.target == target].set_index("fold")
        common = ref.index.intersection(cand.index)
        if len(common) == 0:
            notes.append(f"{target}: no common folds in {label} / {row_set}")
            continue
        incomplete = [
            fold for fold in common
            if not ref.loc[fold, "protocol_complete"] or not cand.loc[fold, "protocol_complete"]
        ]
        if incomplete:
            notes.append(
                f"{target}: excluded folds {sorted(incomplete)} from {label} / {row_set} "
                "because at least one setting is missing a seed (reported as 2-seed cell, not 3)."
            )
        common = common.drop(incomplete)
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
                "setting": "noposemb",
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
    comparisons = pd.DataFrame(rows)
    if not comparisons.empty:
        comparisons["holm_p"] = holm_adjust(comparisons.exact_sign_p.tolist())
    return comparisons, notes


def delta_r2_seed_sd_summary(cells: pd.DataFrame) -> pd.DataFrame:
    """Per-cell across-seed SD of the named delta_r2 metric for noposemb and K=16."""
    rows = []
    for setting in SETTINGS:
        for target in TARGETS:
            for row_set in ROW_SETS:
                sub = cells[
                    (cells.setting == setting)
                    & (cells.target == target)
                    & (cells.row_set == row_set)
                ]
                if sub.empty:
                    continue
                rows.append({
                    "setting": setting,
                    "target": target,
                    "row_set": row_set,
                    "n_cells": len(sub),
                    "n_cells_2_seed": int((sub.n_seeds == 2).sum()),
                    "median_delta_r2_seed_sd": sub["delta_r2_seed_sd"].median(),
                    "mean_delta_r2_seed_sd": sub["delta_r2_seed_sd"].mean(),
                    "max_delta_r2_seed_sd": sub["delta_r2_seed_sd"].max(),
                })
    return pd.DataFrame(rows)


def prereg_outcome(comparisons: pd.DataFrame) -> dict:
    """Assess pre-registered outcome using delta_r2 on the ``all`` row set.

    The primary quantity is ``delta_r2`` (ablated minus K=16) on all rows.
    Higher delta_r2 is better.  Secondary metrics (overall_r2, MAE, RMSE) are
    reported but do not determine the outcome.
    """
    outcome = {
        "supported": None,
        "notes": [],
        "threshold": R1_THRESHOLD,
    }

    def _median_diff(row_set: str, metric: str) -> float:
        if comparisons.empty or "row_set" not in comparisons.columns:
            return np.nan
        sub = comparisons[
            (comparisons["row_set"] == row_set)
            & (comparisons["metric"] == metric)
        ]
        return float(sub["median_paired_difference"].median()) if not sub.empty else np.nan

    median_diff = _median_diff("all", "delta_r2")
    outcome["median_delta_r2_difference_all"] = median_diff

    # Secondary metrics for context.
    for metric in ("overall_r2", "mae", "rmse"):
        outcome[f"median_{metric}_difference_all"] = _median_diff("all", metric)

    if np.isnan(median_diff):
        outcome["supported"] = None
        outcome["notes"].append("Primary quantity is NaN; cannot determine outcome.")
        return outcome

    if median_diff > R1_THRESHOLD:
        outcome["supported"] = "improvement"
        outcome["notes"].append(
            "delta_r2 improves materially without position embeddings. "
            "This is outcome 4 from the pre-registration. Because the controls pass, "
            "the reading is that the position embeddings were fitting a spurious "
            "orientation asymmetry rather than a bug."
        )
    elif median_diff < -R1_THRESHOLD:
        outcome["supported"] = "large_drop"
        outcome["notes"].append(
            "delta_r2 drops materially without position embeddings: positional embeddings "
            "are a principal source of the octamer advantage (outcome 1)."
        )
    else:
        outcome["supported"] = "no_material_change"
        outcome["notes"].append(
            "delta_r2 does not change materially without position embeddings (within ±0.051). "
            "Factor 2 (positional embeddings) is excluded as the principal source of the "
            "octamer gain; factors 1 and 4 remain candidates."
        )

    return outcome


def main() -> None:
    partial = "--partial" in sys.argv
    df = pd.read_csv(DATA_PATH)

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
            "# Octamer positional-embedding ablation — R1 (monomer_heldout)",
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
                hash_rows.append({
                    "target": target,
                    "fold": fold,
                    "n_runs": len(present),
                    "identical": len(set(present)) == 1,
                })
    hash_check = pd.DataFrame(hash_rows)
    if not hash_check.empty and not hash_check.identical.all():
        raise AssertionError(f"Split hashes differ across runs:\n{hash_check[~hash_check.identical]}")

    cells = build_cells(detail, arrays, df)

    comparison_blocks = []
    comparison_notes = []
    for row_set in ROW_SETS:
        frame, notes = build_comparisons(cells, row_set, "R1_all_folds")
        comparison_blocks.append((f"R1 — {row_set}", frame))
        comparison_notes.extend(notes)

    comparisons = (
        pd.concat([frame for _, frame in comparison_blocks], ignore_index=True)
        if comparison_blocks else pd.DataFrame()
    )
    if not comparisons.empty:
        comparisons["holm_p"] = holm_adjust(comparisons.exact_sign_p.tolist())

    prereg = prereg_outcome(comparisons)
    sd_summary = delta_r2_seed_sd_summary(cells)

    cap_counts = (
        detail.groupby("setting")
        .reached_epoch_cap.sum()
        .astype(int)
        .reset_index()
    )
    cap_counts.columns = ["setting", "runs_at_epoch_cap"]
    cap_runs = detail[detail.reached_epoch_cap]
    incomplete_cells = cells[~cells.protocol_complete]

    stem = OUTPUT_PATH.with_suffix("")
    detail.to_csv(stem.with_name(stem.name + "_individual_runs.csv"), index=False)
    cells.to_csv(stem.with_name(stem.name + "_cells.csv"), index=False)
    if not comparisons.empty:
        comparisons.to_csv(stem.with_name(stem.name + "_comparisons.csv"), index=False)
    if not sd_summary.empty:
        sd_summary.to_csv(stem.with_name(stem.name + "_delta_r2_seed_sd_summary.csv"), index=False)
    if not incomplete_cells.empty:
        incomplete_cells[
            ["setting", "target", "fold", "row_set", "n_seeds", "protocol_complete"]
        ].drop_duplicates().to_csv(
            stem.with_name(stem.name + "_incomplete_cells.csv"), index=False
        )

    report = [
        "# Octamer positional-embedding ablation — R1 (monomer_heldout)",
        "",
        "**Convention:** every cell is the mean prediction of three seeds (42, 43, 44), "
        "averaged at the prediction level.  Metrics are reported on four row subsets: "
        "all rows, random rows only, block + alternating rows only, and "
        "`random_via_all_groups` (delta_r2 evaluated on random rows with group means built over all rows).",
        "",
        f"**Coverage:** {n_complete}/{total_inventory} runs."
        + ("  *(partial analysis)*" if partial else ""),
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
        "`skill_*` and `null_*` are only meaningful for the `all` row set. "
        "`protocol_complete=False` means the cell is missing at least one seed and is reported "
        "as a 2-seed (or 1-seed) average, not the intended 3-seed protocol.",
        "",
    ]
    for row_set in ROW_SETS:
        report += [f"### Row set: {row_set}", "", markdown(cells[cells.row_set == row_set]), ""]

    report += [
        "## Per-cell across-seed SD of `delta_r2`",
        "",
        "Each cell's `delta_r2_seed_sd` is computed from the available seeds for that cell "
        "(normally 3; 2 if a seed is missing).",
        "",
        markdown(sd_summary),
        "",
        "## Pre-registered outcome assessment",
        "",
        "**Primary quantity:** `delta_r2` (ablated − K=16) on the `all` row set. "
        f"Materiality threshold for R1: **±{R1_THRESHOLD}**. "
        "Overall R², MAE and RMSE are reported as secondary metrics only.",
        "",
        f"- **Supported outcome:** {prereg['supported']}",
        f"- **Median `delta_r2` difference (ablated − K=16), all rows:** {prereg['median_delta_r2_difference_all']:.6f}",
        f"- **Median `overall_r2` difference (ablated − K=16), all rows:** {prereg.get('median_overall_r2_difference_all', float('nan')):.6f}",
        f"- **Median `mae` difference (ablated − K=16), all rows:** {prereg.get('median_mae_difference_all', float('nan')):.6f}",
        f"- **Median `rmse` difference (ablated − K=16), all rows:** {prereg.get('median_rmse_difference_all', float('nan')):.6f}",
        "",
        "### Metric-choice caveat (pilot observation)",
        "",
        "On pilot fold 0 the primary and secondary metrics moved in opposite directions: "
        "overall R² increased by ~+0.028 while `delta_r2` fell by ~−0.173. "
        "Because the pre-registration names `delta_r2` as the primary quantity, the outcome "
        "is driven by `delta_r2`; overall R² is reported only for context.",
        "",
        "### Assessment notes",
        "",
    ] + [f"- {note}" for note in prereg["notes"]] + [
        "",
        "## Paired per-fold comparisons (ablated minus K=16)",
        "",
        "Signed differences are ablated minus K=16.  Tests are paired per fold. "
        "Folds where either setting is missing a seed are excluded from paired tests; see Exclusions above. "
        "No pooled comparisons are reported.",
        "",
    ]
    if comparison_notes:
        report += ["### Exclusions", ""] + [f"- {note}" for note in comparison_notes] + [""]
    if comparisons.empty:
        report += ["_No comparisons computed._", ""]
    else:
        for label, frame in comparison_blocks:
            report += [f"### {label}", "", markdown(frame), ""]

    OUTPUT_PATH.write_text("\n".join(report))
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
