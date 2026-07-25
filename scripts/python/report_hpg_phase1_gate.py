from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from analysis.diagnostics.ordering import _group_ordering_metrics
PREDICTION_DIR = ROOT_DIR / "predictions" / "ea_ip_group"
MODELS = ["hpg_hier", "hpg_hier_wedge", "hpg_hier_octamer", "hpg_hier_junction"]
TARGETS = {"EA": "EA_vs_SHE_eV", "IP": "IP_vs_SHE_eV"}


def main() -> None:
    df = pd.read_csv(ROOT_DIR / "data" / "ea_ip.csv")
    rows = []
    for model in MODELS:
        for target, token in TARGETS.items():
            path = PREDICTION_DIR / f"ea_ip__{token}__{model}__group_disjoint__fold0__s42.npz"
            prediction = np.load(path, allow_pickle=True)
            y_true = prediction["y_true"].astype(np.float64).reshape(-1)
            y_pred = prediction["y_pred"].astype(np.float64).reshape(-1)
            test_indices = prediction["test_indices"].astype(int).reshape(-1)
            fold_df = df.iloc[test_indices][["smiles_A", "smiles_B", "fracA", "poly_type"]].copy()
            fold_df["y_true"] = y_true
            fold_df["y_pred"] = y_pred
            fold_df["group_key"] = (
                fold_df["smiles_A"].astype(str) + "||" +
                fold_df["smiles_B"].astype(str) + "||" +
                fold_df["fracA"].astype(str)
            )
            n_arch = fold_df.groupby("group_key")["poly_type"].nunique()
            matched = fold_df[fold_df["group_key"].isin(n_arch[n_arch >= 2].index)].copy()
            means = matched.groupby("group_key")[["y_true", "y_pred"]].mean()
            matched["delta_true"] = matched["y_true"] - matched.groupby("group_key")["y_true"].transform("mean")
            matched["delta_pred"] = matched["y_pred"] - matched.groupby("group_key")["y_pred"].transform("mean")
            pairwise = [
                _group_ordering_metrics(group_df)["pairwise_acc"]
                for _, group_df in matched.groupby("group_key")
            ]
            rows.append({
                "model": model,
                "target": target,
                "n_samples": len(y_true),
                "finite": bool(np.isfinite(y_true).all() and np.isfinite(y_pred).all()),
                "overall_r2": r2_score(y_true, y_pred),
                "overall_mae_eV": mean_absolute_error(y_true, y_pred),
                "group_mean_r2": r2_score(means["y_true"], means["y_pred"]),
                "architecture_delta_r2": r2_score(matched["delta_true"], matched["delta_pred"]),
                "pairwise_ordering_accuracy": float(np.mean(pairwise)),
                "n_matched_groups": len(means),
                "n_matched_samples": len(matched),
            })
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
