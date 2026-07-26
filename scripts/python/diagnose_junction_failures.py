from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

CASES = [("IP", "IP_vs_SHE_eV", 5, "bithiophene"), ("EA", "EA_vs_SHE_eV", 6, "benzothiadiazole")]
MODELS = ["hpg_hier", "hpg_hier_junction"]


def fold_frame(df: pd.DataFrame, model: str, token: str, fold: int) -> pd.DataFrame:
    path = ROOT_DIR / "predictions" / "ea_ip_lomo" / f"ea_ip__{token}__{model}__monomer_heldout__fold{fold}__s42.npz"
    prediction = np.load(path, allow_pickle=True)
    indices = prediction["test_indices"].astype(int).reshape(-1)
    frame = df.iloc[indices][["smiles_A", "smiles_B", "fracA", "poly_type"]].copy().reset_index(drop=True)
    frame["y_true"] = prediction["y_true"].astype(float).reshape(-1)
    frame["y_pred"] = prediction["y_pred"].astype(float).reshape(-1)
    frame["group_key"] = frame["smiles_A"].astype(str) + "||" + frame["smiles_B"].astype(str) + "||" + frame["fracA"].astype(str)
    return frame


def parity_axis(axis, frame: pd.DataFrame, title: str) -> None:
    axis.scatter(frame["y_true"], frame["y_pred"], s=8, alpha=0.2)
    lo = min(frame["y_true"].min(), frame["y_pred"].min()) - 0.05
    hi = max(frame["y_true"].max(), frame["y_pred"].max()) + 0.05
    axis.plot([lo, hi], [lo, hi], "--", color="black", linewidth=0.8)
    axis.set(xlim=(lo, hi), ylim=(lo, hi), aspect="equal", title=title, xlabel="True (eV)", ylabel="Predicted (eV)")


def main() -> None:
    df = pd.read_csv(ROOT_DIR / "data" / "ea_ip.csv")
    output_dir = ROOT_DIR / "analysis" / "model_diagnostics" / "junction_n2_failures"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for target, token, fold, monomer in CASES:
        target_column = f"{target} vs SHE (eV)"
        frames = {model: fold_frame(df, model, token, fold) for model in MODELS}
        test_indices = np.load(ROOT_DIR / "predictions" / "ea_ip_lomo" / f"ea_ip__{token}__hpg_hier__monomer_heldout__fold{fold}__s42.npz", allow_pickle=True)["test_indices"].astype(int)
        train_indices = np.setdiff1d(np.arange(len(df)), test_indices)
        y_train = df.iloc[train_indices][target_column].to_numpy(dtype=float)
        y_test = frames["hpg_hier"]["y_true"].to_numpy(dtype=float)
        for model, frame in frames.items():
            matched_counts = frame.groupby("group_key")["poly_type"].nunique()
            matched = frame[frame["group_key"].isin(matched_counts[matched_counts >= 2].index)].copy()
            grouped = matched.groupby("group_key")[["y_true", "y_pred"]].mean()
            matched["delta_true"] = matched["y_true"] - matched.groupby("group_key")["y_true"].transform("mean")
            matched["delta_pred"] = matched["y_pred"] - matched.groupby("group_key")["y_pred"].transform("mean")
            rows.append({
                "target": target,
                "fold": fold,
                "held_out_monomer": monomer,
                "model": model,
                "train_mean": y_train.mean(),
                "train_std": y_train.std(),
                "test_mean": y_test.mean(),
                "test_std": y_test.std(),
                "mean_shift": y_test.mean() - y_train.mean(),
                "overall_r2": r2_score(frame["y_true"], frame["y_pred"]),
                "overall_mae": mean_absolute_error(frame["y_true"], frame["y_pred"]),
                "overall_bias": (frame["y_pred"] - frame["y_true"]).mean(),
                "group_mean_r2": r2_score(grouped["y_true"], grouped["y_pred"]),
                "group_mean_mae": mean_absolute_error(grouped["y_true"], grouped["y_pred"]),
                "group_bias": (grouped["y_pred"] - grouped["y_true"]).mean(),
                "delta_r2": r2_score(matched["delta_true"], matched["delta_pred"]),
            })
        figure, axes = plt.subplots(2, 2, figsize=(9, 9))
        for col, model in enumerate(MODELS):
            parity_axis(axes[0, col], frames[model], f"{model}: overall")
            group_frame = frames[model].groupby("group_key")[["y_true", "y_pred"]].mean().reset_index()
            parity_axis(axes[1, col], group_frame, f"{model}: group means")
        figure.suptitle(f"{target} LOMO fold {fold}: {monomer}")
        figure.tight_layout()
        figure.savefig(output_dir / f"{target.lower()}_fold{fold}_{monomer.replace(' ', '_')}_parity.png", dpi=200)
        plt.close(figure)
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "target_shift_and_parity_metrics.csv", index=False)
    print(result.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
