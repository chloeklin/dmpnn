from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.metrics import compute_overall_mae


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_root", type=Path, default=ROOT / "predictions" / "regen_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "analysis" / "model_diagnostics" / "_regen_v1_checkpoint_gap.csv")
    args = parser.parse_args()
    rows = []
    for path in sorted(args.prediction_root.rglob("*.npz")):
        sidecar_path = path.with_suffix(".config.json")
        if not sidecar_path.is_file():
            raise SystemExit(f"Missing provenance sidecar: {sidecar_path}")
        with np.load(path, allow_pickle=True) as archive:
            if "y_pred_final" not in archive.files:
                raise SystemExit(f"Missing y_pred_final: {path}")
            y_true = archive["y_true"].astype(float).ravel()
            y_best = archive["y_pred"].astype(float).ravel()
            y_final = archive["y_pred_final"].astype(float).ravel()
            model = str(archive["model"].item())
            target = str(archive["target"].item())
            fold = int(archive["fold"].item())
            seed = int(archive["seed"].item())
        if y_true.shape != y_best.shape or y_true.shape != y_final.shape:
            raise SystemExit(f"Prediction shape mismatch: {path}")
        sidecar = json.loads(sidecar_path.read_text())
        rows.append({
            "model": model,
            "target": target,
            "fold": fold,
            "seed": seed,
            "best_mae": compute_overall_mae(y_true, y_best),
            "final_mae": compute_overall_mae(y_true, y_final),
            "final_minus_best_mae": compute_overall_mae(y_true, y_final) - compute_overall_mae(y_true, y_best),
            "epochs_actually_run": sidecar["epochs_actually_run"],
        })
    if not rows:
        raise SystemExit(f"No prediction NPZs found under {args.prediction_root}")
    detail = pd.DataFrame(rows)
    summary = detail.groupby("model", as_index=False).agg(
        cells=("final_minus_best_mae", "count"),
        final_minus_best_mae_mean=("final_minus_best_mae", "mean"),
        final_minus_best_mae_sd=("final_minus_best_mae", "std"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output.with_name(args.output.stem + "_detail.csv"), index=False)
    summary.to_csv(args.output, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
