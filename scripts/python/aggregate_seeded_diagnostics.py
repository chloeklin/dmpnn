from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT_DIR / "analysis" / "model_diagnostics" / "aggregate"
METRIC_FILES = {
    "02_variance_geometry/variance_geometry.csv",
    "02_variance_geometry/model_error_decomposition.csv",
    "03_group_mean_prediction/group_mean_metrics.csv",
    "04_architecture_calibration/calibration_metrics.csv",
    "05_architecture_ordering/ordering_metrics.csv",
    "06_effect_magnitude/effect_magnitude_metrics.csv",
    "09_per_fold_case_studies/fold_case_summaries.csv",
    "10_summary/statistical_comparisons.csv",
    "11_pathological_folds/pathological_fold_metrics.csv",
}
KEY_COLUMNS = ["model", "split", "target", "fold"]


def _load_seeded(relpath: str, seeds: list[int]) -> pd.DataFrame:
    frames = []
    for seed in seeds:
        path = ROOT_DIR / "analysis" / "model_diagnostics" / f"seed_{seed}" / relpath
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["seed"] = seed
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _aggregate_metrics(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    key_columns = [column for column in KEY_COLUMNS if column in frame.columns]
    numeric = [
        column for column in frame.select_dtypes(include=np.number).columns
        if column not in {"seed", "fold"}
    ]
    if not key_columns or not numeric:
        return pd.DataFrame(), []
    seed_fold = frame.groupby(key_columns + ["seed"], as_index=False)[numeric].mean()
    mean_fold = seed_fold.groupby(key_columns, as_index=False)[numeric].mean()
    spread = seed_fold.groupby(key_columns, as_index=False)[numeric].std(ddof=0)
    spread = spread.rename(columns={column: f"{column}_seed_std" for column in numeric})
    return mean_fold.merge(spread, on=key_columns, how="left"), numeric


def _wilcoxon_rows(mean_fold: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    if not {"model", "split", "target", "fold"}.issubset(mean_fold.columns):
        return pd.DataFrame()
    rows = []
    for (split, target), subset in mean_fold.groupby(["split", "target"]):
        pivotable = subset.drop_duplicates(["model", "fold"])
        for metric in metrics:
            values = pivotable.pivot(index="fold", columns="model", values=metric)
            for model_a, model_b in combinations(values.columns, 2):
                paired = values[[model_a, model_b]].dropna()
                differences = paired[model_a] - paired[model_b]
                try:
                    pvalue = float(wilcoxon(differences, alternative="two-sided").pvalue)
                except ValueError:
                    pvalue = np.nan
                rows.append({
                    "split": split,
                    "target": target,
                    "metric": metric,
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_folds": len(paired),
                    "mean_difference": float(differences.mean()) if len(differences) else np.nan,
                    "median_difference": float(differences.median()) if len(differences) else np.nan,
                    "wilcoxon_p": pvalue,
                })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,43,44")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_tests = []
    for relpath in sorted(METRIC_FILES):
        frame = _load_seeded(relpath, seeds)
        if frame.empty:
            continue
        mean_fold, metrics = _aggregate_metrics(frame)
        if mean_fold.empty:
            continue
        stem = relpath.replace("/", "__").removesuffix(".csv")
        mean_fold.to_csv(OUT_DIR / f"{stem}__mean_folds.csv", index=False)
        tests = _wilcoxon_rows(mean_fold, metrics)
        if not tests.empty:
            tests.insert(0, "source", relpath)
            all_tests.append(tests)
    if all_tests:
        pd.concat(all_tests, ignore_index=True).to_csv(OUT_DIR / "paired_wilcoxon_across_folds.csv", index=False)


if __name__ == "__main__":
    main()
