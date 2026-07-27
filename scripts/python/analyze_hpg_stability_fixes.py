from __future__ import annotations

import json
from math import comb, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from aggregate_lomo_seeds import frame_from_npz, metric_row

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "ea_ip.csv"
CURRENT_DIR = ROOT / "predictions" / "noise_floor" / "ea_ip_lomo"
FIX_DIR = ROOT / "predictions" / "stability_fixes" / "ea_ip_lomo"
OUTPUT_PATH = ROOT / "analysis" / "model_diagnostics" / "_training_stability_stepc_results.md"
BASE = "ea_ip__EA_vs_SHE_eV__hpg_hier__monomer_heldout__fold{fold}__s42__repeat{repeat}"
FIXES = ("best_checkpoint", "row_val_best")
METRICS = ("mae", "group_mean_r2", "delta_r2")
POWER_EFFECT_EV = 0.03
POWER_TARGET = 0.80


def markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False):
        cells = [f"{value:.8f}" if isinstance(value, (float, np.floating)) else str(value) for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def sign_test_power(per_run_sd: float, effect: float, runs_per_cell: int) -> float:
    direction_probability = norm.cdf(effect * sqrt(runs_per_cell) / (sqrt(2.0) * per_run_sd))
    return float(sum(
        comb(9, wins) * direction_probability ** wins * (1.0 - direction_probability) ** (9 - wins)
        for wins in (0, 1, 8, 9)
    ))


def required_runs(per_run_sd: float, effect: float = POWER_EFFECT_EV, target: float = POWER_TARGET) -> int:
    for runs in range(1, 1001):
        if sign_test_power(per_run_sd, effect, runs) >= target:
            return runs
    raise ValueError(f"More than 1000 runs required for SD={per_run_sd}")


def artifact_paths(config: str, fold: int, repeat: int) -> tuple[Path, Path]:
    stem = BASE.format(fold=fold, repeat=repeat)
    if config == "current":
        root = CURRENT_DIR
    else:
        root = FIX_DIR
        stem += f"__{config}"
    return root / f"{stem}.npz", root / f"{stem}.config.json"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    rows = []
    environments = []
    for config in ("current", *FIXES):
        for fold in (0, 1):
            for repeat in (1, 2, 3):
                npz_path, sidecar_path = artifact_paths(config, fold, repeat)
                if not npz_path.exists() or not sidecar_path.exists():
                    raise SystemExit(f"Missing artifact: {npz_path} or {sidecar_path}")
                with np.load(npz_path, allow_pickle=True) as archive:
                    metrics, _ = metric_row(frame_from_npz(df, archive))
                    y_true = archive["y_true"].astype(float)
                    final_predictions = archive["y_pred_final"].astype(float) if "y_pred_final" in archive.files else np.asarray([])
                if config != "current" and final_predictions.shape != y_true.shape:
                    raise AssertionError(f"Missing same-run final predictions in {npz_path}")
                final_mae = float(np.mean(np.abs(y_true - final_predictions))) if final_predictions.size else np.nan
                sidecar = json.loads(sidecar_path.read_text())
                environment = sidecar["runtime_environment"]
                if environment["accelerator"] != "cuda" or "V100" not in str(environment["device_name"]):
                    raise AssertionError(f"Expected V100 CUDA artifact, got {environment}")
                environments.append(environment)
                curve = sidecar.get("validation_loss_curve")
                rows.append({
                    "config": config,
                    "fold": fold,
                    "repeat": repeat,
                    **{metric: metrics[metric] for metric in METRICS},
                    "epochs": sidecar["epochs_actually_run"],
                    "best_epoch": sidecar.get("best_epoch"),
                    "best_val_loss": sidecar["best_val_loss"],
                    "final_model_mae": final_mae,
                    "final_minus_best_mae": final_mae - metrics["mae"] if final_predictions.size else np.nan,
                    "wall_time_seconds": sidecar["wall_time_seconds"],
                    "validation_curve_epochs": None if curve is None else len(curve),
                })
    detail = pd.DataFrame(rows)
    summary = detail.groupby(["config", "fold"], as_index=False).agg(
        mae_mean=("mae", "mean"), mae_sd=("mae", "std"),
        group_mean_r2_mean=("group_mean_r2", "mean"), group_mean_r2_sd=("group_mean_r2", "std"),
        delta_r2_mean=("delta_r2", "mean"), delta_r2_sd=("delta_r2", "std"),
        checkpoint_mae_gap_mean=("final_minus_best_mae", "mean"),
        checkpoint_mae_gap_sd=("final_minus_best_mae", "std"),
        wall_time_mean_seconds=("wall_time_seconds", "mean"),
    )
    summary["fold1_success"] = np.where(summary.fold == 1, summary.mae_sd < 0.02, np.nan)
    fixes_fold1 = summary[(summary.config != "current") & (summary.fold == 1)]
    successful = fixes_fold1[fixes_fold1.mae_sd < 0.02].config.tolist()
    checkpoint_summary = detail[detail.config != "current"].groupby("config", as_index=False).agg(
        runs=("final_minus_best_mae", "count"),
        final_minus_best_mae_mean=("final_minus_best_mae", "mean"),
        final_minus_best_mae_median=("final_minus_best_mae", "median"),
        final_minus_best_mae_min=("final_minus_best_mae", "min"),
        final_minus_best_mae_max=("final_minus_best_mae", "max"),
    )
    power_rows = []
    for per_run_sd in (0.005, 0.010, 0.015, 0.020, 0.030):
        runs = required_runs(per_run_sd)
        power_rows.append({
            "per_run_sd_eV": per_run_sd,
            "runs_per_model_cell": runs,
            "exact_power": sign_test_power(per_run_sd, POWER_EFFECT_EV, runs),
            "jobs_for_144_cells": 144 * runs,
        })
    power = pd.DataFrame(power_rows)
    environment = environments[0]
    detail.to_csv(OUTPUT_PATH.with_suffix(".csv"), index=False)
    report = [
        "# HPG-hier stability fixes: Step C results",
        "",
        f"All artifacts were verified as CUDA V100 runs. Representative environment: `{environment['device_name']}`, driver `{environment['driver_version']}`, torch `{environment['torch_version']}`, torch CUDA `{environment['torch_cuda_version']}`, deterministic cuDNN `{environment['cudnn_deterministic']}`, global deterministic algorithms `{environment['deterministic_algorithms_enabled']}`.",
        "",
        "## Primary comparison",
        "",
        markdown(summary),
        "",
        f"**Success criterion:** fold-1 MAE SD < 0.02 eV. Successful fixes: {', '.join(successful) if successful else 'none'}.",
        "",
        "If `best_checkpoint` succeeds, retain the current single-monomer validation design. Use `row_val_best` only if the bug fix alone fails and row-level validation succeeds. Fixed 60 epochs remains a fallback only if neither arm succeeds.",
        "",
        "## Best-checkpoint versus final-model MAE",
        "",
        "Positive values mean the final patience-expired model was worse than its same-run best checkpoint.",
        "",
        markdown(checkpoint_summary),
        "",
        "## Nine-fold paired sign-test power",
        "",
        f"Exact two-sided sign-test power for nine folds, a common same-sign effect of {POWER_EFFECT_EV:.3f} eV per fold, independent equal per-model run SD, and alpha 0.05 (rejection at at least 8 of 9 signs):",
        "",
        markdown(power),
        "",
        "## Per-run diagnostics",
        "",
        markdown(detail),
        "",
    ]
    OUTPUT_PATH.write_text("\n".join(report))
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
