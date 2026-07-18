from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from evaluation.naming import make_prediction_filename, split_subdir
from run_stage2d_generalization import build_group_keys

ROOT_DIR = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="hpg_sum,hpg_frac")
    parser.add_argument("--split", default="group_disjoint")
    parser.add_argument("--target", default="EA vs SHE (eV)")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--assert-nonzero-seed-std", action="store_true")
    parser.add_argument("--seeds", default=None)
    args = parser.parse_args()
    df = pd.read_csv(ROOT_DIR / "data" / "ea_ip.csv")
    group_keys = build_group_keys(df)
    rows = []
    seeds = [args.seed] if args.seeds is None else [int(value) for value in args.seeds.split(",")]
    for seed in seeds:
        for model in args.models.split(","):
            path = ROOT_DIR / "predictions" / split_subdir(args.split) / make_prediction_filename(
                args.target, model, args.split, args.fold, seed=seed
            )
            data = np.load(path)
            y_true = data["y_true"].reshape(-1)
            y_pred = data["y_pred"].reshape(-1)
            indices = data["test_indices"].astype(int)
            groups = pd.DataFrame({"key": group_keys[indices], "y_true": y_true, "y_pred": y_pred}).groupby("key").mean()
            rows.append({
                "seed": seed,
                "model": model,
                "split": args.split,
                "target": args.target,
                "fold": args.fold,
                "overall_r2": r2_score(y_true, y_pred),
                "group_mean_r2": r2_score(groups["y_true"], groups["y_pred"]),
            })
    result = pd.DataFrame(rows)
    grouped = result.groupby(["model", "split", "target", "fold"], as_index=False).agg(
        overall_r2_mean=("overall_r2", "mean"),
        overall_r2_seed_std=("overall_r2", lambda values: values.std(ddof=0)),
        group_mean_r2_mean=("group_mean_r2", "mean"),
        group_mean_r2_seed_std=("group_mean_r2", lambda values: values.std(ddof=0)),
    )
    output = ROOT_DIR / "analysis" / "model_diagnostics" / "aggregate" / "hpg_gate_seed_metrics.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output, index=False)
    print(result.to_string(index=False))
    print(grouped.to_string(index=False))
    if args.assert_nonzero_seed_std:
        if not (grouped["overall_r2_seed_std"] > 0).all():
            raise AssertionError(f"Expected non-zero per-fold seed standard deviation, got {grouped.to_dict('records')}")


if __name__ == "__main__":
    main()
