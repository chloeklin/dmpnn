"""Analyze wDMPNN original-paper reproduction on R1 (monomer_heldout).

Loads predictions from predictions/wdmpnn_original/ea_ip_lomo/ (__orig suffix) and
compares them against the regen_v1 R1 models in predictions/regen_v1/ea_ip_lomo/.

Every comparison is paired per-fold, averaged over the three seeds at the prediction
level.  The protocol-variant labels are explicit: wdmpnn-original is
`protocol_variant='original_paper'`; every other model is `regen_v1`.  This is the
protocol-parity test and must never be pooled with the regen_v1 wDMPNN rows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation.metrics import compute_copolymer_metrics
from analyze_regen_v1 import (
    markdown,
    holm_adjust,
    load_metrics,
    null_floors,
)

DATA_PATH = ROOT / "data" / "ea_ip.csv"
WDMPNN_ORIG_DIR = ROOT / "predictions" / "wdmpnn_original" / "ea_ip_lomo"
REGEN_DIR = ROOT / "predictions" / "regen_v1" / "ea_ip_lomo"
OUTPUT_PATH = ROOT / "analysis" / "model_diagnostics" / "_wdmpnn_original_results.md"

TARGETS = {"EA": "EA_vs_SHE_eV", "IP": "IP_vs_SHE_eV"}
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
SU_PER_GPU_HOUR = 36.0

WDMPNN_ORIG_MODEL = "wdmpnn_original"
REGEN_MODELS = ("wdmpnn", "hpg_hier", "hpg_hier_octamer", "hpg_hier_junction", "hpg_hier_junction1")


def wdmpnn_orig_path(target: str, fold: int, seed: int) -> Path:
    return WDMPNN_ORIG_DIR / f"ea_ip__{TARGETS[target]}__wdmpnn__monomer_heldout__fold{fold}__s{seed}__orig.npz"


def regen_path(model: str, target: str, fold: int, seed: int) -> Path:
    return REGEN_DIR / f"ea_ip__{TARGETS[target]}__{model}__monomer_heldout__fold{fold}__s{seed}.npz"


def load_wdmpnn_orig_run(df: pd.DataFrame, target: str, fold: int, seed: int) -> tuple[dict, dict]:
    path = wdmpnn_orig_path(target, fold, seed)
    metrics, _, payload = load_metrics(df, path)
    sidecar = json.loads(path.with_suffix(".config.json").read_text())
    return metrics, sidecar, payload


def load_regen_run(df: pd.DataFrame, model: str, target: str, fold: int, seed: int) -> tuple[dict, dict, dict]:
    path = regen_path(model, target, fold, seed)
    metrics, _, payload = load_metrics(df, path)
    sidecar = json.loads(path.with_suffix(".config.json").read_text())
    return metrics, sidecar, payload


def assert_wdmpnn_orig_sidecar(sidecar: dict, path: Path) -> None:
    rc = sidecar.get("resolved_config", {})
    cli = sidecar.get("cli_args", {})
    errors = []
    if sidecar.get("epochs_actually_run") != 30:
        errors.append(f"epochs_actually_run={sidecar.get('epochs_actually_run')} != 30")
    if cli.get("batch_size") != 50 and rc.get("batch_size") != 50:
        errors.append(f"batch_size != 50")
    if cli.get("patience") != 30 and rc.get("patience") != 30:
        errors.append(f"patience != 30")
    if cli.get("protocol_variant") != "original_paper" and rc.get("protocol_variant") != "original_paper":
        errors.append(f"protocol_variant != 'original_paper'")
    if cli.get("frozen_protocol") is not True and rc.get("frozen_protocol") is not True:
        errors.append(f"frozen_protocol != true")
    if errors:
        raise AssertionError(f"{path}: {'; '.join(errors)}")


def seed_average_metrics(df: pd.DataFrame, payloads: list[dict]) -> tuple[dict, pd.DataFrame]:
    first = payloads[0]
    if any(
        not np.array_equal(first["indices"], p["indices"])
        or not np.array_equal(first["y_true"], p["y_true"])
        for p in payloads[1:]
    ):
        raise AssertionError("Rows differ across seeds")
    averaged = np.mean(np.stack([p["y_pred"] for p in payloads]), axis=0)
    metrics, group_means = compute_copolymer_metrics(df, first["y_true"], averaged, first["indices"])
    return metrics, group_means


def build_cells(df: pd.DataFrame, detail: pd.DataFrame, arrays: dict, model: str) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        for fold in FOLDS:
            subset = detail[(detail.model == model) & (detail.target == target) & (detail.fold == fold)]
            if subset.empty:
                continue
            seeds = subset.seed.tolist()
            payloads = [arrays[(model, target, fold, seed)] for seed in seeds]
            averaged_metrics, _ = seed_average_metrics(df, payloads)
            row = {"model": model, "target": target, "fold": fold, "n_seeds": len(seeds)}
            for metric in METRICS:
                row[metric] = averaged_metrics[metric]
                row[f"{metric}_seed_mean"] = float(subset[metric].mean())
                row[f"{metric}_seed_sd"] = float(subset[metric].std(ddof=1))
            rows.append(row)
    cells = pd.DataFrame(rows)
    cells = cells.merge(null_floors(), on=["target", "fold"], validate="many_to_one")
    cells["beats_null_floor"] = cells.group_mean_r2 > cells.null_group_mean_r2
    cells["skill_group_mean"] = (cells.group_mean_r2 - cells.null_group_mean_r2) / (1.0 - cells.null_group_mean_r2)
    cells["skill_overall"] = (cells.overall_r2 - cells.null_overall_r2) / (1.0 - cells.null_overall_r2)
    cells["null_floor_headroom_used"] = cells["skill_group_mean"]
    return cells


def build_comparisons(wdmpnn_cells: pd.DataFrame, regen_cells: pd.DataFrame) -> pd.DataFrame:
    """Candidate = regen_v1 model, reference = wdmpnn-original."""
    rows = []
    for target in TARGETS:
        ref = wdmpnn_cells[wdmpnn_cells.target == target].set_index("fold")
        for model in REGEN_MODELS:
            cand = regen_cells[(regen_cells.model == model) & (regen_cells.target == target)].set_index("fold")
            common = ref.index.intersection(cand.index)
            if len(common) == 0:
                continue
            ref_sub = ref.loc[common]
            cand_sub = cand.loc[common]
            for metric in COMPARISON_METRICS:
                differences = cand_sub[metric] - ref_sub[metric]
                wins = int((differences < 0).sum()) if metric == "mae" else int((differences > 0).sum())
                losses = int((differences > 0).sum()) if metric == "mae" else int((differences < 0).sum())
                non_ties = wins + losses
                p_value = float(binomtest(wins, non_ties, 0.5).pvalue) if non_ties else 1.0
                seed_sd = np.maximum(
                    cand_sub[f"{metric}_seed_sd"],
                    ref_sub[f"{metric}_seed_sd"],
                )
                rows.append({
                    "ref_model": WDMPNN_ORIG_MODEL,
                    "ref_protocol_variant": "original_paper",
                    "cand_model": model,
                    "cand_protocol_variant": "regen_v1",
                    "target": target,
                    "metric": metric,
                    "n_folds": len(common),
                    "median_paired_difference": differences.median(),
                    "wins": wins,
                    "losses": losses,
                    "exact_sign_p": p_value,
                    "min_attainable_p": float(binomtest(len(common), len(common), 0.5).pvalue) if len(common) else np.nan,
                    "folds_smaller_than_measured_seed_sd": int((differences.abs() < seed_sd).sum()),
                })
    comparisons = pd.DataFrame(rows)
    if not comparisons.empty:
        comparisons["holm_p"] = holm_adjust(comparisons.exact_sign_p.tolist())
    return comparisons


def cost_table(wdmpnn_detail: pd.DataFrame, regen_detail: pd.DataFrame) -> pd.DataFrame:
    all_detail = pd.concat([wdmpnn_detail, regen_detail], ignore_index=True)
    rows = []
    for model in [WDMPNN_ORIG_MODEL] + list(REGEN_MODELS):
        sub = all_detail[all_detail.model == model]
        if sub.empty:
            continue
        for target in TARGETS:
            tsub = sub[sub.target == target]
            if tsub.empty:
                continue
            wall = tsub.wall_time_seconds.median()
            epochs = tsub.epochs.median()
            seconds_per_epoch = (tsub.wall_time_seconds / tsub.epochs).median()
            su_per_run = wall / 3600.0 * SU_PER_GPU_HOUR
            rows.append({
                "model": model,
                "target": target,
                "runs": len(tsub),
                "epochs_actually_run_median": epochs,
                "median_wall_time_seconds": wall,
                "median_seconds_per_epoch": seconds_per_epoch,
                "median_su_per_run": su_per_run,
            })
    return pd.DataFrame(rows)


def main() -> None:
    partial = "--partial" in sys.argv
    df = pd.read_csv(DATA_PATH)

    # Inventory wdmpnn-original
    wdmpnn_inventory = []
    for target in TARGETS:
        for fold in FOLDS:
            for seed in SEEDS:
                path = wdmpnn_orig_path(target, fold, seed)
                wdmpnn_inventory.append({
                    "target": target, "fold": fold, "seed": seed,
                    "available": path.is_file(),
                    "sidecar": path.with_suffix(".config.json").is_file(),
                })
    wdmpnn_inventory = pd.DataFrame(wdmpnn_inventory)
    wdmpnn_inventory["complete"] = wdmpnn_inventory.available & wdmpnn_inventory.sidecar

    # Inventory regen_v1 R1 models
    regen_inventory = []
    for model in REGEN_MODELS:
        for target in TARGETS:
            for fold in FOLDS:
                for seed in SEEDS:
                    path = regen_path(model, target, fold, seed)
                    regen_inventory.append({
                        "model": model, "target": target, "fold": fold, "seed": seed,
                        "available": path.is_file(),
                        "sidecar": path.with_suffix(".config.json").is_file(),
                    })
    regen_inventory = pd.DataFrame(regen_inventory)
    regen_inventory["complete"] = regen_inventory.available & regen_inventory.sidecar

    wdmpnn_complete = int(wdmpnn_inventory.complete.sum())
    wdmpnn_total = len(wdmpnn_inventory)
    regen_complete = int(regen_inventory.complete.sum())
    regen_total = len(regen_inventory)

    if wdmpnn_complete != wdmpnn_total and not partial:
        pending = wdmpnn_inventory[~wdmpnn_inventory.complete]
        report = [
            "# wDMPNN original-paper analysis — R1 monomer_heldout",
            "",
            f"## Status: pending — {wdmpnn_complete}/{wdmpnn_total} wDMPNN-original runs complete",
            "",
            "Re-run with `--partial` to analyse whatever has landed.",
            "",
            "## Missing wDMPNN-original cells",
            "",
            markdown(pending),
            "",
        ]
        OUTPUT_PATH.write_text("\n".join(report))
        print(f"Wrote pending report: {OUTPUT_PATH}")
        return

    if regen_complete != regen_total and not partial:
        # We can still analyse wdmpnn-original if regen comparators are incomplete,
        # but comparisons will be partial.
        print(f"Warning: only {regen_complete}/{regen_total} regen_v1 R1 comparator runs available")

    # Load wdmpnn-original
    wdmpnn_run_rows = []
    wdmpnn_arrays = {}
    for row in wdmpnn_inventory[wdmpnn_inventory.complete].itertuples(index=False):
        metrics, sidecar, payload = load_wdmpnn_orig_run(df, row.target, row.fold, row.seed)
        assert_wdmpnn_orig_sidecar(sidecar, wdmpnn_orig_path(row.target, row.fold, row.seed))
        wdmpnn_run_rows.append({
            "model": WDMPNN_ORIG_MODEL,
            "target": row.target,
            "fold": row.fold,
            "seed": row.seed,
            **metrics,
            "epochs": sidecar["epochs_actually_run"],
            "best_epoch": sidecar["best_epoch"],
            "wall_time_seconds": sidecar["wall_time_seconds"],
        })
        wdmpnn_arrays[(WDMPNN_ORIG_MODEL, row.target, row.fold, row.seed)] = payload
    wdmpnn_detail = pd.DataFrame(wdmpnn_run_rows)

    # Load regen_v1 comparators
    regen_run_rows = []
    regen_arrays = {}
    for row in regen_inventory[regen_inventory.complete].itertuples(index=False):
        metrics, sidecar, payload = load_regen_run(df, row.model, row.target, row.fold, row.seed)
        regen_run_rows.append({
            "model": row.model,
            "target": row.target,
            "fold": row.fold,
            "seed": row.seed,
            **metrics,
            "epochs": sidecar["epochs_actually_run"],
            "best_epoch": sidecar["best_epoch"],
            "wall_time_seconds": sidecar["wall_time_seconds"],
        })
        regen_arrays[(row.model, row.target, row.fold, row.seed)] = payload
    regen_detail = pd.DataFrame(regen_run_rows)

    # Build cells (seed-averaged)
    wdmpnn_cells = build_cells(df, wdmpnn_detail, wdmpnn_arrays, WDMPNN_ORIG_MODEL)
    regen_cells = pd.concat([
        build_cells(df, regen_detail[regen_detail.model == m], regen_arrays, m)
        for m in REGEN_MODELS
        if m in regen_detail.model.values
    ], ignore_index=True) if not regen_detail.empty else pd.DataFrame()

    comparisons = build_comparisons(wdmpnn_cells, regen_cells)
    costs = cost_table(wdmpnn_detail, regen_detail)

    # Combine all cells for a summary table
    all_cells = pd.concat([wdmpnn_cells, regen_cells], ignore_index=True)

    stem = OUTPUT_PATH.with_suffix("")
    wdmpnn_detail.to_csv(stem.with_name(stem.name + "_individual_runs.csv"), index=False)
    regen_detail.to_csv(stem.with_name(stem.name + "_regen_individual_runs.csv"), index=False)
    all_cells.to_csv(stem.with_name(stem.name + "_cells.csv"), index=False)
    comparisons.to_csv(stem.with_name(stem.name + "_comparisons.csv"), index=False)
    costs.to_csv(stem.with_name(stem.name + "_cost.csv"), index=False)

    report = [
        "# wDMPNN original-paper analysis — R1 monomer_heldout",
        "",
        "**Convention:** all figures are the mean prediction of three seeds (42, 43, 44), averaged at the prediction level.",
        "",
        f"**Coverage:** {wdmpnn_complete}/{wdmpnn_total} wDMPNN-original runs; {regen_complete}/{regen_total} regen_v1 R1 comparator runs.",
        "",
        "## ⚠️ Protocol-parity test",
        "",
        "This comparison crosses protocol variants by design.  The left-hand side is the original-paper wDMPNN run "
        "(`protocol_variant='original_paper'`, batch_size=50, epochs=30, patience=30, no early-stop possibility).  "
        "The right-hand side is the frozen-regeneration protocol (`protocol_variant='regen_v1'`).  "
        "Treat the wDMPNN-original rows as a protocol-parity probe, not as additional replicates to pool with the regen_v1 wDMPNN rows.",
        "",
        "## Asserted run invariants",
        "",
        "All wDMPNN-original runs were asserted to have:",
        "- `epochs_actually_run == 30`",
        "- `batch_size == 50`",
        "- `patience == 30`",
        "- `protocol_variant == 'original_paper'`",
        "- `frozen_protocol == true`",
        "",
        "The script failed loudly if any of these were violated.",
        "",
        "## Three-seed averaged cells",
        "",
        "`skill_*` is the skill score against the fold-specific A-blind null floor.",
        "",
        markdown(all_cells),
        "",
        "## Paired per-fold comparisons: regen_v1 model minus wDMPNN-original",
        "",
        "Signed differences are **candidate (regen_v1) minus reference (wDMPNN-original)**.  "
        "For metrics where higher is better (R², ordering) a positive median means regen_v1 is ahead.  "
        "For MAE/RMSE a negative median means regen_v1 is ahead.  `ref_protocol_variant` is always `original_paper`; "
        "`cand_protocol_variant` is always `regen_v1`.",
        "",
        markdown(comparisons),
        "",
        "## Cost table",
        "",
        "Median wall time, seconds per epoch, and SU/run at 36 SU per GPU-hour.  "
        "wDMPNN-original uses `num_workers=4`; HPG models use `num_workers=0`, so per-epoch wall times are not a clean architectural comparison.",
        "",
        markdown(costs),
        "",
        "## Completeness",
        "",
        f"- wDMPNN-original: {wdmpnn_complete}/{wdmpnn_total}",
        f"- regen_v1 R1 comparators: {regen_complete}/{regen_total}",
        "",
    ]
    OUTPUT_PATH.write_text("\n".join(report))
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
