from __future__ import annotations

import json
import warnings
from math import comb, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, pearsonr

from aggregate_lomo_seeds import frame_from_npz, metric_row

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "ea_ip.csv"
CURRENT_DIR = ROOT / "predictions" / "noise_floor" / "ea_ip_lomo"
FIX_DIR = ROOT / "predictions" / "stability_fixes" / "ea_ip_lomo"
OUTPUT_PATH = ROOT / "analysis" / "model_diagnostics" / "_training_stability_stepc_results.md"
BASE = "ea_ip__EA_vs_SHE_eV__hpg_hier__monomer_heldout__fold{fold}__s42__repeat{repeat}"
FIXES = ("best_checkpoint", "row_val_best", "arm_c")
METRICS = ("mae", "group_mean_r2", "delta_r2", "ordering", "overall_r2")
POWER_EFFECT_EV = 0.03
POWER_TARGET = 0.80
DELTA_R2_SD_THRESHOLD = 0.083
N_REPEATS = 3
REQUIRED_CONFIGS = frozenset({"current", "best_checkpoint", "row_val_best"})


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


def sd_ci_95(s: float, n: int = N_REPEATS) -> tuple[float, float]:
    """95% CI for the true population SD given observed sample SD s from n observations."""
    df = n - 1
    return (
        s * sqrt(df / chi2.ppf(0.975, df)),
        s * sqrt(df / chi2.ppf(0.025, df)),
    )


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
    prediction_frames: dict[tuple[str, int, int], pd.DataFrame] = {}
    pending_configs: set[str] = set()
    for config in ("current", *FIXES):
        config_missing = False
        for fold in (0, 1):
            if config_missing:
                break
            for repeat in (1, 2, 3):
                npz_path, sidecar_path = artifact_paths(config, fold, repeat)
                if not npz_path.exists() or not sidecar_path.exists():
                    if config in REQUIRED_CONFIGS:
                        raise SystemExit(f"Missing artifact: {npz_path} or {sidecar_path}")
                    config_missing = True
                    break
                with np.load(npz_path, allow_pickle=True) as archive:
                    prediction_frame = frame_from_npz(df, archive)
                    metrics, _ = metric_row(prediction_frame)
                    y_true = archive["y_true"].astype(float)
                    final_predictions = archive["y_pred_final"].astype(float) if "y_pred_final" in archive.files else np.asarray([])
                prediction_frames[(config, fold, repeat)] = prediction_frame
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
        if config_missing:
            pending_configs.add(config)
            warnings.warn(f"Config '{config}' has missing artifacts; it will be absent from the report.")
    detail = pd.DataFrame(rows)
    correlation_rows = []
    correlation_scopes = [("fold_0", detail[detail.fold == 0]), ("fold_1", detail[detail.fold == 1]), ("pooled", detail)]
    for scope, values in correlation_scopes:
        r, p = pearsonr(values["best_val_loss"], values["mae"])
        correlation_rows.append({
            "scope": scope,
            "runs": len(values),
            "pearson_r": float(r),
            "p_value": float(p),
        })
    correlations = pd.DataFrame(correlation_rows)
    pooled_correlation = float(correlations.loc[correlations.scope == "pooled", "pearson_r"].iloc[0])
    selection_uninformative = pooled_correlation <= 0.0 or abs(pooled_correlation) < 0.2
    uninformative_scopes = correlations.loc[
        (correlations.pearson_r <= 0.0) | (correlations.pearson_r.abs() < 0.2), "scope"
    ].tolist()

    averaged_rows = []
    for config in detail.config.unique():
        for fold in (0, 1):
            frames = [prediction_frames[(config, fold, repeat)] for repeat in (1, 2, 3)]
            reference = frames[0]
            for repeat, frame in enumerate(frames[1:], start=2):
                if not reference[["smiles_A", "smiles_B", "fracA", "poly_type", "y_true"]].equals(
                    frame[["smiles_A", "smiles_B", "fracA", "poly_type", "y_true"]]
                ):
                    raise AssertionError(f"Prediction rows differ for {config} fold {fold}, repeat {repeat}")
            averaged = reference.copy()
            averaged["y_pred"] = np.mean(np.stack([frame.y_pred.to_numpy() for frame in frames]), axis=0)
            averaged_metrics, _ = metric_row(averaged)
            individual = detail[(detail.config == config) & (detail.fold == fold)]
            row = {"config": config, "fold": fold, "repeats_averaged": len(frames)}
            for metric in METRICS:
                individual_mean = float(individual[metric].mean())
                row[f"averaged_{metric}"] = averaged_metrics[metric]
                row[f"individual_mean_{metric}"] = individual_mean
                row[f"averaged_minus_individual_{metric}"] = averaged_metrics[metric] - individual_mean
            averaged_rows.append(row)
    repeat_averaged = pd.DataFrame(averaged_rows)
    mae_averaging_wins = int((repeat_averaged.averaged_minus_individual_mae < 0.0).sum())
    higher_is_better = ("group_mean_r2", "delta_r2", "ordering", "overall_r2")
    score_averaging_wins = sum(
        int((repeat_averaged[f"averaged_minus_individual_{metric}"] > 0.0).sum())
        for metric in higher_is_better
    )
    score_comparisons = len(repeat_averaged) * len(higher_is_better)

    summary = detail.groupby(["config", "fold"], as_index=False).agg(
        mae_mean=("mae", "mean"), mae_sd=("mae", "std"),
        group_mean_r2_mean=("group_mean_r2", "mean"), group_mean_r2_sd=("group_mean_r2", "std"),
        delta_r2_mean=("delta_r2", "mean"), delta_r2_sd=("delta_r2", "std"),
        ordering_mean=("ordering", "mean"), ordering_sd=("ordering", "std"),
        overall_r2_mean=("overall_r2", "mean"), overall_r2_sd=("overall_r2", "std"),
        checkpoint_mae_gap_mean=("final_minus_best_mae", "mean"),
        checkpoint_mae_gap_sd=("final_minus_best_mae", "std"),
        wall_time_mean_seconds=("wall_time_seconds", "mean"),
    )
    summary["fold1_mae_ok"] = np.where(summary.fold == 1, summary.mae_sd < 0.02, np.nan)
    summary["fold1_delta_r2_ok"] = np.where(summary.fold == 1, summary.delta_r2_sd < DELTA_R2_SD_THRESHOLD, np.nan)
    summary["fold1_success"] = np.where(
        summary.fold == 1,
        (summary.mae_sd < 0.02) & (summary.delta_r2_sd < DELTA_R2_SD_THRESHOLD),
        np.nan,
    )
    fixes_fold1 = summary[(summary.config != "current") & (summary.fold == 1)]
    successful = fixes_fold1[
        (fixes_fold1.mae_sd < 0.02) & (fixes_fold1.delta_r2_sd < DELTA_R2_SD_THRESHOLD)
    ].config.tolist()
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
    correlations.to_csv(OUTPUT_PATH.with_name(f"{OUTPUT_PATH.stem}_validation_correlation.csv"), index=False)
    repeat_averaged.to_csv(OUTPUT_PATH.with_name(f"{OUTPUT_PATH.stem}_repeat_averaged.csv"), index=False)
    ex_sd = 0.020
    ex_lo, ex_hi = sd_ci_95(ex_sd)
    pending_section: list[str] = []
    if pending_configs:
        pending_section = [
            f"> **Pending:** {', '.join(sorted(pending_configs))} — artifacts not yet collected on Gadi V100.",
            "",
        ]
    report = [
        "# HPG-hier stability fixes: Step C results",
        "",
        f"All artifacts were verified as CUDA V100 runs. Representative environment: `{environment['device_name']}`, driver `{environment['driver_version']}`, torch `{environment['torch_version']}`, torch CUDA `{environment['torch_cuda_version']}`, deterministic cuDNN `{environment['cudnn_deterministic']}`, global deterministic algorithms `{environment['deterministic_algorithms_enabled']}`.",
        "",
        *pending_section,
        "## Primary comparison",
        "",
        markdown(summary),
        "",
        f"**Success criteria (co-equal):** fold-1 MAE SD < 0.02 eV AND fold-1 delta-R\u00b2 SD < {DELTA_R2_SD_THRESHOLD:.3f} (current baseline). Ordering SD is reported but has no fixed threshold. Successful fixes (both criteria met): {', '.join(successful) if successful else 'none'}.",
        "",
        "If `best_checkpoint` succeeds, retain the current single-monomer validation design. Use `row_val_best` only if the bug fix alone fails and row-level validation succeeds. `arm_c` (epoch floor 40, patience 30) targets runs that stopped at the first noisy minimum; it succeeds if both MAE SD and delta-R\u00b2 SD fall below their respective thresholds. Fixed 60 epochs remains a fallback only if none of the above succeed.",
        "",
        "## Validation loss versus test MAE",
        "",
        "Pearson correlations use all three configurations (`current`, `best_checkpoint`, and `row_val_best`): nine runs per fold and 18 runs pooled.",
        "",
        markdown(correlations),
        "",
        (f"**Validation-based model selection is uninformative under this design for {', '.join(uninformative_scopes)}:** the correlation is near zero or negative."
         if uninformative_scopes else
         "No fold or pooled correlation is near zero or negative."),
        ("**Validation-based model selection is uninformative under this design in the pooled analysis:** the pooled correlation is near zero or negative."
         if selection_uninformative else
         "The pooled correlation is positive and not near zero, so the pooled 18-run diagnostic does not support the same conclusion."),
        "",
        "## Three-repeat prediction averaging",
        "",
        "For each configuration and fold, `y_pred` was averaged row-wise across the three repeats before recomputing every metric. `averaged_minus_individual_*` is the averaged-prediction metric minus the mean metric of the three individual runs; positive is better for R² and ordering, while negative is better for MAE.",
        "",
        markdown(repeat_averaged),
        "",
        f"Repeat averaging lowered MAE in {mae_averaging_wins}/{len(repeat_averaged)} configuration-fold cells and improved the higher-is-better metrics in {score_averaging_wins}/{score_comparisons} comparisons. " + ("Averaging consistently recovers predictive performance that single-run validation-based selection does not reliably deliver." if mae_averaging_wins == len(repeat_averaged) and score_averaging_wins == score_comparisons else "Averaging helps in most comparisons, but does not uniformly recover predictive performance."),
        "",
        "## Small-sample caveat",
        "",
        f"Every SD in this report is estimated from n={N_REPEATS} repeats. The 95% confidence interval for the true population SD \u03c3 given an observed sample SD s is approximately [0.52\u2009s,\u20096.28\u2009s] (chi-squared distribution, {N_REPEATS - 1} degrees of freedom). For example, an observed SD of {ex_sd:.3f}\u2009eV corresponds to a 95% CI of [{ex_lo:.3f}, {ex_hi:.3f}]\u2009eV. **All threshold comparisons should be treated as provisional.**",
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
