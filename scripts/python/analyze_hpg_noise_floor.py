from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from aggregate_lomo_seeds import frame_from_npz, metric_row

ROOT = Path(__file__).resolve().parents[2]
PREDICTION_DIR = ROOT / "predictions" / "noise_floor" / "ea_ip_lomo"
DATA_PATH = ROOT / "data" / "ea_ip.csv"
OUTPUT_PATH = ROOT / "analysis" / "model_diagnostics" / "_noise_floor_results.md"
BASE = "ea_ip__EA_vs_SHE_eV__hpg_hier__monomer_heldout__fold{fold}__s42__repeat{repeat}"
OCTAMER_MAE_DIFFERENCES = np.asarray([0.071, 0.183, 0.083, 0.010, -0.035, 0.025, 0.002, 0.100, 0.051])
METRICS = ["group_mean_r2", "delta_r2", "ordering", "overall_r2", "mae"]


def markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False):
        cells = []
        for value in row:
            cells.append(f"{value:.8f}" if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    rows = []
    predictions: dict[int, list[np.ndarray]] = {0: [], 1: []}
    environments = []
    for fold in (0, 1):
        for repeat in (1, 2, 3):
            stem = BASE.format(fold=fold, repeat=repeat)
            npz_path = PREDICTION_DIR / f"{stem}.npz"
            config_path = PREDICTION_DIR / f"{stem}.config.json"
            if not npz_path.exists() or not config_path.exists():
                raise SystemExit(f"Missing noise-floor artifact: {npz_path} or {config_path}")
            with np.load(npz_path, allow_pickle=True) as archive:
                frame = frame_from_npz(df, archive)
                metrics, _ = metric_row(frame)
                predictions[fold].append(archive["y_pred"].astype(float).ravel())
            config = json.loads(config_path.read_text())
            environment = config["runtime_environment"]
            if environment["accelerator"] != "cuda" or "V100" not in str(environment["device_name"]):
                raise AssertionError(f"Expected Gadi V100 CUDA run, got {environment}")
            environments.append(environment)
            rows.append({
                "fold": fold,
                "repeat": repeat,
                **{name: metrics[name] for name in METRICS},
                "wall_time_seconds": config["wall_time_seconds"],
                "accelerator": environment["accelerator"],
                "device": environment["device_name"],
                "driver": environment["driver_version"],
                "torch": environment["torch_version"],
                "torch_cuda": environment["torch_cuda_version"],
                "deterministic_requested": environment["deterministic_kernels_requested"],
                "deterministic_enabled": environment["deterministic_algorithms_enabled"],
            })
    detail = pd.DataFrame(rows)
    environment_keys = [
        "accelerator", "device_name", "driver_version", "torch_version", "torch_cuda_version",
        "cudnn_version", "deterministic_kernels_requested", "deterministic_algorithms_enabled",
        "cudnn_deterministic", "cudnn_benchmark",
    ]
    unique_environments = {
        tuple((key, json.dumps(environment.get(key), sort_keys=True)) for key in environment_keys)
        for environment in environments
    }
    if len(unique_environments) != 1:
        raise AssertionError("Noise-floor runtime environments differ across repeats")
    summary_rows = []
    for fold in (0, 1):
        selected = detail[detail.fold == fold]
        pairwise = [float(np.max(np.abs(predictions[fold][left] - predictions[fold][right]))) for left, right in combinations(range(3), 2)]
        row = {"fold": fold}
        for metric in METRICS:
            row[f"{metric}_mean"] = selected[metric].mean()
            row[f"{metric}_sd"] = selected[metric].std(ddof=1)
        row["max_pairwise_prediction_difference_eV"] = max(pairwise)
        row["wall_time_mean_seconds"] = selected.wall_time_seconds.mean()
        row["wall_time_sd_seconds"] = selected.wall_time_seconds.std(ddof=1)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    mae_band = float(summary.mae_sd.max())
    inside_all = np.abs(OCTAMER_MAE_DIFFERENCES) <= mae_band
    matched_inside = [abs(OCTAMER_MAE_DIFFERENCES[fold]) <= summary.loc[summary.fold == fold, "mae_sd"].iloc[0] for fold in (0, 1)]
    environment = environments[0]
    detail.to_csv(OUTPUT_PATH.with_suffix(".csv"), index=False)
    report = [
        "# HPG-hier Gadi V100 noise floor",
        "",
        "## Runtime environment",
        "",
        f"All six runs used accelerator `{environment['accelerator']}`, device `{environment['device_name']}`, driver `{environment['driver_version']}`, torch `{environment['torch_version']}`, torch CUDA `{environment['torch_cuda_version']}`, cuDNN `{environment['cudnn_version']}`, deterministic kernels requested `{environment['deterministic_kernels_requested']}`, deterministic algorithms enabled `{environment['deterministic_algorithms_enabled']}`, cuDNN deterministic `{environment['cudnn_deterministic']}`, and cuDNN benchmark `{environment['cudnn_benchmark']}`.",
        "",
        "Every run used current code, A-heldout EA, seed 42, and changed only the independent process/repeat label. SD is the sample SD across three repeats.",
        "",
        "## Per-run metrics and V100 wall time",
        "",
        markdown(detail),
        "",
        "## Per-fold noise floor",
        "",
        markdown(summary),
        "",
        "## Octamer-versus-baseline EA MAE differences",
        "",
        f"Only folds 0 and 1 have fold-matched noise estimates. {sum(matched_inside)} of those 2 differences are within their own fold's one-SD MAE noise band.",
        "",
        f"For an explicit all-nine sensitivity count, using the larger measured fold-0/fold-1 MAE SD ({mae_band:.8f} eV) as a common conservative one-SD band places {int(inside_all.sum())} of 9 recorded differences inside the band. This extrapolation is not a fold-matched estimate for folds 2-8.",
        "",
        "Recorded signed differences: `" + ", ".join(f"{value:+.3f}" for value in OCTAMER_MAE_DIFFERENCES) + "` eV.",
        "",
        "The measured SD columns are required context for future model-comparison tables.",
        "",
    ]
    OUTPUT_PATH.write_text("\n".join(report))
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
