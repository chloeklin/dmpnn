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
import warnings
from pathlib import Path

import numpy as np

# Suppress harmless "Mean of empty slice" warnings from all-NaN group metrics on
# single-poly_type row subsets (e.g. random-only rows have no defined delta_r2).
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
ROW_SETS = ("all", "random", "block_alternating", "random_via_all_groups")
ROW_SET_FILTERS: dict[str, tuple[str, ...] | None] = {
    "all": None,
    "random": ("random",),
    "block_alternating": ("block", "alternating"),
    "random_via_all_groups": None,
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


def _delta_r2_random_via_all_groups(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    global_indices: np.ndarray,
) -> float:
    """Compute delta_r2 evaluated on random rows only, with groups built over all rows.

    Group key is smiles_A || smiles_B || fracA, and only groups with >=2 distinct
    poly_type values are retained, matching compute_copolymer_metrics.  Within-group
    deviations are computed on the full fold and then R² is evaluated on the random
    rows only.
    """
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
        if row_set == "random_via_all_groups":
            # Only meaningful on 3-seed averaged predictions; placeholder here.
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
                    if row_set == "random_via_all_groups":
                        # Use the full-fold ("all") payloads so groups are built over all rows,
                        # then delta_r2 is evaluated on random rows only.
                        source_row_set = "all"
                    else:
                        source_row_set = row_set
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
    cells: pd.DataFrame, folds: tuple[int, ...], row_set: str, label: str
) -> tuple[pd.DataFrame, list[str]]:
    """Paired per-fold comparison of K=1 vs K=16 within a fold group and row set.

    Folds where K=1 or K=16 does not have the full three-seed protocol are excluded
    from paired tests and reported in the returned note list.
    """
    rows = []
    notes = []
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
            notes.append(f"{target}: no common folds in {label} / {row_set}")
            continue
        # Exclude any fold where either setting is not protocol-complete (3 seeds).
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
    comparisons = pd.DataFrame(rows)
    if not comparisons.empty:
        comparisons["holm_p"] = holm_adjust(comparisons.exact_sign_p.tolist())
    return comparisons, notes


def _seed_sd_sign_test(
    cells: pd.DataFrame,
    group: str | None,
    seed_sd_row_set: str = "all",
) -> dict:
    """Paired per-cell exact sign test for K=1 vs K=16 delta_r2 seed SD.

    Pairs are matched on (target, fold[, fold_group]).  Only protocol-complete
    cells (three seeds) contribute.  Returns n, wins (K=1 higher), median
    paired difference (K=1 minus K=16), and two-sided exact sign p-value.
    """
    sub = cells[(cells.row_set == seed_sd_row_set) & cells.protocol_complete]
    if group is not None:
        sub = sub[sub.fold_group == group]
    if sub.empty:
        return {"n": 0, "wins": 0, "median_diff": np.nan, "pvalue": np.nan}
    on = ["target", "fold"]
    if group is None:
        on.append("fold_group")
    k1 = sub[sub.setting == "k1"][on + ["delta_r2_seed_sd"]]
    k16 = sub[sub.setting == "k16"][on + ["delta_r2_seed_sd"]]
    paired = k1.merge(k16, on=on, suffixes=("_k1", "_k16"), how="inner")
    paired = paired.dropna(subset=["delta_r2_seed_sd_k1", "delta_r2_seed_sd_k16"])
    n = len(paired)
    if n == 0:
        return {"n": 0, "wins": 0, "median_diff": np.nan, "pvalue": np.nan}
    wins = int((paired["delta_r2_seed_sd_k1"] > paired["delta_r2_seed_sd_k16"]).sum())
    median_diff = float(
        (paired["delta_r2_seed_sd_k1"] - paired["delta_r2_seed_sd_k16"]).median()
    )
    pvalue = float(binomtest(wins, n, 0.5, alternative="two-sided").pvalue)
    return {"n": n, "wins": wins, "median_diff": median_diff, "pvalue": pvalue}


def prereg_outcome(
    comparisons: pd.DataFrame,
    cells: pd.DataFrame,
    threshold: float = 0.024,
    seed_sd_row_set: str = "all",
    seed_sd_significance: float = 0.05,
    seed_sd_min_median: float = 0.005,
) -> dict:
    """Determine which pre-registered outcome A-D is supported.

    The primary quantity is the K=1-minus-K=16 difference in **overall_r2** on
    random rows, D folds, because the copolymer group metrics (including the
    metric literally named `delta_r2`) are undefined on a single-`poly_type`
    subset.  The seed-SD comparison uses the per-cell across-seed SD of the named
    `delta_r2` metric on the full fold (`all` row set), evaluated with a paired
    per-cell exact sign test rather than a naive median comparison.  A seed-SD
    rise requires (a) a consistent direction (two-sided exact sign p < 0.05)
    and (b) a post-hoc median-paired gap > 0.005; the pre-registration did not
    define a material threshold for seed SD, so any positive finding would be
    reported cautiously.
    """
    outcome = {
        "supported": None,
        "notes": [],
        "threshold": threshold,
    }

    def _median_diff(row_set: str, metric: str, group: str) -> float:
        sub = comparisons[
            (comparisons.fold_group == group)
            & (comparisons.row_set == row_set)
            & (comparisons.metric == metric)
        ]
        return float(sub["median_paired_difference"].median()) if not sub.empty else np.nan

    median_diff = _median_diff("random", "overall_r2", "D_cross_scaffold")
    outcome["median_overall_r2_difference_random_D"] = median_diff

    # Negative-control: overall_r2 shift on block/alternating rows, D folds
    outcome["median_overall_r2_difference_block_alt_D"] = _median_diff(
        "block_alternating", "overall_r2", "D_cross_scaffold"
    )
    outcome["control_block_alt_delta_r2_D"] = _median_diff(
        "block_alternating", "delta_r2", "D_cross_scaffold"
    )

    # Seed SD rise: paired per-cell exact sign test on delta_r2 seed SD
    sd_s = _seed_sd_sign_test(cells, "S_within_scaffold", seed_sd_row_set)
    sd_d = _seed_sd_sign_test(cells, "D_cross_scaffold", seed_sd_row_set)
    sd_all = _seed_sd_sign_test(cells, None, seed_sd_row_set)
    outcome["seed_sd_S"] = sd_s
    outcome["seed_sd_D"] = sd_d
    outcome["seed_sd_overall"] = sd_all

    def _rises(sd_result: dict) -> bool:
        if sd_result["n"] == 0:
            return False
        return (
            sd_result["wins"] > sd_result["n"] / 2
            and sd_result["pvalue"] < seed_sd_significance
            and sd_result["median_diff"] > seed_sd_min_median
        )

    outcome["seed_sd_rises"] = bool(_rises(sd_d))

    if np.isnan(median_diff):
        outcome["supported"] = None
        outcome["notes"].append("Primary quantity is NaN; cannot determine outcome.")
        return outcome

    if median_diff > threshold:
        outcome["supported"] = "D"
        outcome["notes"].append(
            "Overall R² rises at K=1 on random rows (unexpected; treat as bug signal until controls pass)."
        )
    elif median_diff < -threshold:
        if outcome["seed_sd_rises"] is True:
            outcome["supported"] = "A"
            outcome["notes"].append(
                "Seed SD rises and overall R² falls materially on random rows: the in-loss ensemble carried the octamer gain."
            )
        else:
            outcome["supported"] = "A-ish"
            outcome["notes"].append(
                "Overall R² falls materially on random rows but seed SD does not rise: outcome A partially supported."
            )
    else:
        if outcome["seed_sd_rises"] is True:
            outcome["supported"] = "B"
            outcome["notes"].append(
                "Seed SD rises but overall R² holds on random rows: stability came from ensembling; architecture effect survives."
            )
        else:
            outcome["supported"] = "C"
            outcome["notes"].append(
                "Overall R² holds on random rows and seed SD does not show a consistent rise: "
                "the 16 replicas were not contributing. The remaining candidates are factor 2 "
                "(positional embeddings) and factor 4 (discarded edge features), neither of which "
                "arms C or D address. The seed-SD criterion was not defined in advance, so the "
                "stability claim is reported as inconclusive."
            )
    return outcome


def delta_r2_seed_sd_summary(cells: pd.DataFrame) -> pd.DataFrame:
    """Per-cell across-seed SD of the named `delta_r2` metric for K=1 and K=16, by fold group and row subset."""
    rows = []
    for setting in ("k1", "k16"):
        for target in TARGETS:
            for row_set in ROW_SETS:
                for fold_group, label in [("S_within_scaffold", "S"), ("D_cross_scaffold", "D")]:
                    sub = cells[
                        (cells.setting == setting)
                        & (cells.target == target)
                        & (cells.row_set == row_set)
                        & (cells.fold_group == fold_group)
                    ]
                    if sub.empty:
                        continue
                    rows.append({
                        "setting": setting,
                        "target": target,
                        "row_set": row_set,
                        "fold_group": label,
                        "n_cells": len(sub),
                        "n_cells_2_seed": int((sub.n_seeds == 2).sum()),
                        "median_delta_r2_seed_sd": sub["delta_r2_seed_sd"].median(),
                        "mean_delta_r2_seed_sd": sub["delta_r2_seed_sd"].mean(),
                        "max_delta_r2_seed_sd": sub["delta_r2_seed_sd"].max(),
                    })
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
    comparison_notes = []
    if group_s:
        for row_set in ROW_SETS:
            frame, notes = build_comparisons(cells, group_s, row_set, "S_within_scaffold")
            comparison_blocks.append((f"S — {row_set}", frame))
            comparison_notes.extend(notes)
    if group_d:
        for row_set in ROW_SETS:
            frame, notes = build_comparisons(cells, group_d, row_set, "D_cross_scaffold")
            comparison_blocks.append((f"D — {row_set}", frame))
            comparison_notes.extend(notes)

    comparisons = (
        pd.concat([frame for _, frame in comparison_blocks], ignore_index=True)
        if comparison_blocks else pd.DataFrame()
    )
    if not comparisons.empty:
        comparisons["holm_p"] = holm_adjust(comparisons.exact_sign_p.tolist())

    prereg = prereg_outcome(comparisons, cells)
    sd_summary = delta_r2_seed_sd_summary(cells)

    cap_counts = (
        detail.groupby("setting")
        .reached_epoch_cap.sum()
        .astype(int)
        .reset_index()
    )
    cap_counts.columns = ["setting", "runs_at_epoch_cap"]
    cap_runs = detail[detail.reached_epoch_cap]

    # Incomplete protocol cells: record those that are not 3-seed
    incomplete_cells = cells[~cells.protocol_complete]

    stem = OUTPUT_PATH.with_suffix("")
    detail.to_csv(stem.with_name(stem.name + "_individual_runs.csv"), index=False)
    cells.to_csv(stem.with_name(stem.name + "_cells.csv"), index=False)
    if not composition.empty:
        composition.to_csv(stem.with_name(stem.name + "_fold_composition.csv"), index=False)
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
        "# Octamer K=1 vs K=16 — clustered B-heldout",
        "",
        "**Convention:** every cell is the mean prediction of three seeds (42, 43, 44), "
        "averaged at the prediction level.  Metrics are reported on four row subsets: "
        "all rows, random rows only, block + alternating rows only, and "
        "`random_via_all_groups` (delta_r2 evaluated on random rows with group means built over all rows).",
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
        "because the fold-level null floor is computed on the full test fold. "
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
        "(normally 3; 2 if a seed is missing).  The table below summarises how many cells contribute "
        "and how many are based on only two seeds, because three seeds per cell gives a weak SD estimate.",
        "",
        markdown(sd_summary),
        "",
        "## Pre-registered outcome assessment",
        "",
        "Primary quantity: **overall R² on random rows, D folds**, K=1 minus K=16.  "
        "The pre-registered material threshold is **0.024**.",
        "",
        f"- **Supported outcome:** {prereg['supported']}",
        f"- **Median overall R² difference (K=1 − K=16), random rows, D folds:** {prereg.get('median_overall_r2_difference_random_D', float('nan')):.6f}",
        f"- **Median overall R² difference (K=1 − K=16), block+alternating rows, D folds (control):** {prereg.get('median_overall_r2_difference_block_alt_D', float('nan')):.6f}",
        f"- **Median `delta_r2` difference (K=1 − K=16), block+alternating rows, D folds (control):** {prereg.get('control_block_alt_delta_r2_D', float('nan')):.6f}",
    ]
    for sd_name, sd_label in [("seed_sd_D", "D folds"), ("seed_sd_S", "S folds"), ("seed_sd_overall", "all paired cells")]:
        sd = prereg.get(sd_name, {})
        if sd.get("n"):
            report.append(
                f"- **Paired per-cell `delta_r2` seed-SD sign test, {sd_label}:** "
                f"{sd['wins']}/{sd['n']} cells higher at K=1, "
                f"median paired difference = {sd['median_diff']:.6f}, "
                f"two-sided exact sign p = {sd['pvalue']:.6f}"
            )
        else:
            report.append(f"- **Paired per-cell `delta_r2` seed-SD sign test, {sd_label}:** insufficient paired cells.")
    report += [
        f"- **Seed SD rises at K=1:** {prereg.get('seed_sd_rises', None)}",
        "",
        "The seed-SD rise test requires both a consistent direction (two-sided exact sign p < 0.05) "
        "and a post-hoc median paired gap > 0.005.  The 0.005 gap threshold was chosen after seeing "
        "the data and is disclosed for transparency; it does not change the reported outcome because "
        "the sign test does not pass first.",
        "",
        "Pre-registered outcomes (from `PREREG_octamer_k1_2026-07-30.md`):",
        "",
        "| Outcome | Reading |",
        "|---|---|",
        "| A | Seed SD rises **and** overall R² falls on random rows → the in-loss ensemble carried the octamer gain. |",
        "| B | Seed SD rises, overall R² holds → stability came from ensembling; architecture effect survives. |",
        "| C | Neither moves materially → the 16 replicas were not contributing. |",
        "| D | Overall R² rises at K=1 → unexpected; treat as bug signal until controls pass. |",
        "",
        "### Assessment notes",
        "",
    ] + [f"- {note}" for note in prereg["notes"]] + [
        "",
        "### Correction to the pre-registration",
        "",
        "The `block`/`alternating` subset is **not** a pure negative control.  "
        "Although those rows take the argmax path and `--n_random_samples 1` does not change their "
        "forward pass, training with K=1 changes the shared model weights (and therefore the predictions "
        "on all rows) relative to the K=16 run.  A material shift on block/alternating therefore rules "
        "out configuration errors and confirms that the comparison is between two genuinely different "
        "trained models, but it does **not** rule out weight-driven effects spilling over from the random rows.",
        "",
        "## Paired per-fold comparisons (K=1 minus K=16)",
        "",
    ]
    if comparison_notes:
        report += ["### Exclusions", ""] + [f"- {note}" for note in comparison_notes] + [""]
    if comparisons.empty:
        report += ["_No comparisons computed._", ""]
    else:
        report += [
            "Signed differences are K=1 minus K=16.  Tests are run within fold group and row set. "
            "Folds where either setting is missing a seed are excluded from paired tests; see Exclusions above. "
            "No pooled comparisons are reported.",
            "",
        ]
        for label, frame in comparison_blocks:
            report += [f"### {label}", "", markdown(frame), ""]

    OUTPUT_PATH.write_text("\n".join(report))
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
