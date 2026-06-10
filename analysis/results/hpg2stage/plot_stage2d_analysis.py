"""HPG2Stage Stage 2D analysis — architecture-aware polymer modeling.

Generates publication-quality figures comparing the Frac → 2D-0 → 2D-1
progression and quantifying how architecture-aware modeling improves
copolymer property prediction.

Expected input:
    - Results CSVs in results/HPG2Stage/ with stage2d_* naming
    - Prediction .npz files in predictions/HPG2Stage/ with test_ids

Usage:
    python plot_stage2d_analysis.py [--results_dir PATH] [--pred_dir PATH] [--out_dir PATH]
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════════

# Project root (two levels up from this script's directory)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

RESULTS_DIR_DEFAULT = PROJECT_ROOT / "results" / "HPG2Stage"
PRED_DIR_DEFAULT = PROJECT_ROOT / "predictions" / "HPG2Stage"
OUT_DIR_DEFAULT = Path(__file__).resolve().parent / "figures_stage2d"

# Model keys and display labels (ordered for plots)
MODEL_KEYS = [
    "stage2d_frac",
    "stage2d_2d0_fixed",
    "stage2d_2d0_arch",
    "stage2d_2d0_gate",
    "stage2d_2d1_fixed",
    "stage2d_2d1_arch",
    "stage2d_2d1_gate",
]

MODEL_DISPLAY = {
    "stage2d_frac":      "Frac",
    "stage2d_2d0_fixed": "2D0-fixed",
    "stage2d_2d0_arch":  "2D0-arch",
    "stage2d_2d0_gate":  "2D0-gate",
    "stage2d_2d1_fixed": "2D1-fixed",
    "stage2d_2d1_arch":  "2D1-arch",
    "stage2d_2d1_gate":  "2D1-gate",
}

# Regex to extract stage2d variant from filename (non-greedy, stop at __)
VARIANT_PATTERN = re.compile(r"copoly_(stage2d_\w+?)__")

# Target mapping
TARGET_RAW = {"EA": "EA vs SHE (eV)", "IP": "IP vs SHE (eV)"}
TARGET_SHORT = {v: k for k, v in TARGET_RAW.items()}

# Architecture labels
ARCH_LABEL_MAP = {"alternating": 0, "random": 1, "block": 2}
ARCH_DISPLAY = {0: "alt", 1: "rand", 2: "block"}

# ── Plotting style ──────────────────────────────────────────────────────────
FIGURE_DPI = 300
FONT_SIZE = 11
TITLE_SIZE = 13
TICK_SIZE = 9
LABEL_SIZE = 10
BAR_LABEL_SIZE = 8

# Color palette for models (colorblind-friendly progression)
COLORS = {
    "stage2d_frac":      "#999999",  # grey baseline
    "stage2d_2d0_fixed": "#4E79A7",
    "stage2d_2d0_arch":  "#59A14F",
    "stage2d_2d0_gate":  "#F28E2B",
    "stage2d_2d1_fixed": "#E15759",
    "stage2d_2d1_arch":  "#B07AA1",
    "stage2d_2d1_gate":  "#76B7B2",
}

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "figure.dpi": FIGURE_DPI,
})


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def _variant_from_filename(filename: str) -> Optional[str]:
    """Extract stage2d variant from result filename."""
    m = VARIANT_PATTERN.search(filename)
    return m.group(1) if m else None


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load per-fold result CSVs and return tidy DataFrame.

    Returns DataFrame with columns: model, target, fold, mae, rmse, r2
    """
    results_dir = Path(results_dir)
    frames: List[pd.DataFrame] = []

    for csv_path in sorted(results_dir.glob("*stage2d*.csv")):
        variant = _variant_from_filename(csv_path.stem)
        if variant is None:
            continue

        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        # Normalise columns
        col_map = {"test/rmse": "rmse", "test/r2": "r2", "test/mae": "mae", "split": "fold"}
        df = df.rename(columns=col_map)

        required = {"rmse", "r2", "mae", "fold"}
        if not required.issubset(df.columns):
            continue

        # Short target label
        if "target" in df.columns:
            df["target"] = df["target"].map(lambda t: TARGET_SHORT.get(t, t))
        else:
            # Infer from filename
            for raw, short in TARGET_SHORT.items():
                if raw in csv_path.stem:
                    df["target"] = short
                    break

        df["model"] = variant
        frames.append(df[["model", "target", "fold", "mae", "rmse", "r2"]])

    if not frames:
        raise FileNotFoundError(
            f"No stage2d result CSVs found in {results_dir}.\n"
            "Expected filenames containing 'copoly_stage2d_<variant>'."
        )

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["model", "target", "fold"], keep="last")
    return combined


def load_predictions(pred_dir: Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load .npz prediction files and return nested dict: model → target → DataFrame.

    Each DataFrame has columns: y_true, y_pred, test_id (optional).
    Multiple splits are concatenated.
    """
    pred_dir = Path(pred_dir)
    if not pred_dir.exists():
        return {}

    results: Dict[str, Dict[str, pd.DataFrame]] = {}

    for npz_path in sorted(pred_dir.glob("ea_ip*stage2d*.npz")):
        variant = _variant_from_filename(npz_path.stem)
        if variant is None:
            continue

        # Determine target from filename
        target = None
        for raw, short in TARGET_SHORT.items():
            if raw in npz_path.stem:
                target = short
                break
        if target is None:
            continue

        data = np.load(npz_path, allow_pickle=True)
        y_true = data["y_true"].flatten()
        y_pred = data["y_pred"].flatten()

        row = {"y_true": y_true, "y_pred": y_pred}
        if "test_ids" in data:
            ids = data["test_ids"]
            if len(ids) == len(y_true):
                row["test_id"] = ids

        df_split = pd.DataFrame(row)

        if variant not in results:
            results[variant] = {}
        if target not in results[variant]:
            results[variant][target] = df_split
        else:
            results[variant][target] = pd.concat(
                [results[variant][target], df_split], ignore_index=True
            )

    return results


def load_dataset_metadata(data_dir: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """Try to load ea_ip dataset to get architecture and monomer info."""
    if data_dir is None:
        data_dir = PROJECT_ROOT / "data"
    csv_path = data_dir / "ea_ip.csv"
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def build_summary_from_results(df_results: pd.DataFrame) -> pd.DataFrame:
    """Build summary table from per-fold results (mean across folds)."""
    rows = []
    for model in MODEL_KEYS:
        row = {"model": model}
        for target in ["EA", "IP"]:
            subset = df_results[(df_results["model"] == model) & (df_results["target"] == target)]
            if subset.empty:
                row[f"{target}_R2"] = np.nan
                row[f"{target}_MAE"] = np.nan
                row[f"{target}_RMSE"] = np.nan
            else:
                row[f"{target}_R2"] = subset["r2"].mean()
                row[f"{target}_MAE"] = subset["mae"].mean()
                row[f"{target}_RMSE"] = subset["rmse"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def build_summary_from_predictions(preds: Dict) -> pd.DataFrame:
    """Build summary table from prediction files (pooled across splits)."""
    rows = []
    for model in MODEL_KEYS:
        row = {"model": model}
        if model not in preds:
            for target in ["EA", "IP"]:
                row[f"{target}_R2"] = np.nan
                row[f"{target}_MAE"] = np.nan
                row[f"{target}_RMSE"] = np.nan
        else:
            for target in ["EA", "IP"]:
                if target in preds[model]:
                    df_p = preds[model][target]
                    yt = df_p["y_true"].values
                    yp = df_p["y_pred"].values
                    row[f"{target}_R2"] = compute_r2(yt, yp)
                    row[f"{target}_MAE"] = compute_mae(yt, yp)
                    row[f"{target}_RMSE"] = compute_rmse(yt, yp)
                else:
                    row[f"{target}_R2"] = np.nan
                    row[f"{target}_MAE"] = np.nan
                    row[f"{target}_RMSE"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  Architecture Deviation Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def _build_dataset_lookup(dataset_df: pd.DataFrame) -> Tuple[dict, dict, str, str, str, str]:
    """Build fast lookup tables from dataset: y_true → (group_key, arch_label).

    Matching strategy: join predictions to dataset rows via y_true value.
    EA/IP floats are unique enough (continuous) to serve as a reliable key.

    Returns (ea_lookup, ip_lookup, smiles_a_col, smiles_b_col, frac_col, arch_col)
    where each lookup maps rounded float → dict of metadata.
    """
    # Detect columns
    smiles_a_col = next((c for c in ["smiles_A", "SMILES_A", "monomer_A"] if c in dataset_df.columns), None)
    smiles_b_col = next((c for c in ["smiles_B", "SMILES_B", "monomer_B"] if c in dataset_df.columns), None)
    frac_col = next((c for c in ["fracA", "frac_A", "f_A"] if c in dataset_df.columns), None)
    arch_col = next((c for c in ["poly_type", "architecture", "arch"] if c in dataset_df.columns), None)

    ea_col = "EA vs SHE (eV)"
    ip_col = "IP vs SHE (eV)"

    ea_lookup: dict = {}
    ip_lookup: dict = {}

    for _, row in dataset_df.iterrows():
        group_key = None
        if smiles_a_col and smiles_b_col and frac_col:
            group_key = f"{row[smiles_a_col]}|{row[smiles_b_col]}|{row[frac_col]:.3f}"

        arch = row[arch_col] if arch_col else None

        if ea_col in dataset_df.columns and pd.notna(row[ea_col]):
            ea_lookup[round(float(row[ea_col]), 6)] = {"group": group_key, "arch": arch}
        if ip_col in dataset_df.columns and pd.notna(row[ip_col]):
            ip_lookup[round(float(row[ip_col]), 6)] = {"group": group_key, "arch": arch}

    return ea_lookup, ip_lookup, smiles_a_col, smiles_b_col, frac_col, arch_col


def compute_architecture_deviations(
    preds: Dict, dataset_df: Optional[pd.DataFrame]
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, pd.DataFrame]]]:
    """Compute architecture-induced deviations for each model.

    For each composition group (monomer_A, monomer_B, f_A, f_B):
        Δy = y - group_mean(y)

    Matching strategy: join predictions to dataset rows via y_true value
    (test_ids from npz are sequential fold indices, not global row indices).

    Returns:
        summary_df: DataFrame with model, EA_arch_R2, IP_arch_R2
        deviation_data: model → target → DataFrame with delta_true, delta_pred, architecture
    """
    if not preds or dataset_df is None:
        return pd.DataFrame(), {}

    ea_lookup, ip_lookup, smiles_a_col, smiles_b_col, frac_col, arch_col = \
        _build_dataset_lookup(dataset_df)

    if smiles_a_col is None or frac_col is None:
        print("  [warn] Cannot compute architecture deviations: missing composition columns")
        return pd.DataFrame(), {}

    lookups = {"EA": ea_lookup, "IP": ip_lookup}
    deviation_data: Dict[str, Dict[str, pd.DataFrame]] = {}
    summary_rows = []

    for model in MODEL_KEYS:
        if model == "stage2d_frac":
            continue
        if model not in preds:
            summary_rows.append({"model": model, "EA_arch_R2": np.nan, "IP_arch_R2": np.nan})
            continue

        model_devs: Dict[str, pd.DataFrame] = {}
        row = {"model": model}

        for target in ["EA", "IP"]:
            if target not in preds[model]:
                row[f"{target}_arch_R2"] = np.nan
                continue

            df_pred = preds[model][target].copy()
            df_pred["y_true"] = df_pred["y_true"].astype(float)
            df_pred["y_pred"] = df_pred["y_pred"].astype(float)

            # Match each prediction to dataset metadata via y_true value
            lut = lookups[target]
            groups, archs = [], []
            for yt in df_pred["y_true"].values:
                meta = lut.get(round(float(yt), 6), None)
                groups.append(meta["group"] if meta else None)
                archs.append(meta["arch"] if meta else None)

            df_pred["group"] = groups
            df_pred["architecture"] = archs

            # Drop rows that didn't match
            matched = df_pred.dropna(subset=["group"])
            match_rate = len(matched) / max(len(df_pred), 1)
            if match_rate < 0.5:
                print(f"  [warn] {model}/{target}: only {match_rate:.1%} rows matched dataset — skipping")
                row[f"{target}_arch_R2"] = np.nan
                continue
            if len(matched) < 10:
                row[f"{target}_arch_R2"] = np.nan
                continue

            # Compute group means and deviations
            group_mean_true = matched.groupby("group")["y_true"].transform("mean")
            group_mean_pred = matched.groupby("group")["y_pred"].transform("mean")
            matched = matched.copy()
            matched["delta_true"] = matched["y_true"].values - group_mean_true.values
            matched["delta_pred"] = matched["y_pred"].values - group_mean_pred.values

            # Only keep groups with >1 member (arch types differ, so ≥2 per group)
            group_counts = matched.groupby("group")["y_true"].transform("count")
            dev_df = matched[group_counts > 1].copy()

            if len(dev_df) < 10:
                row[f"{target}_arch_R2"] = np.nan
                continue

            dt = dev_df["delta_true"].values
            dp = dev_df["delta_pred"].values
            row[f"{target}_arch_R2"] = compute_r2(dt, dp)
            model_devs[target] = dev_df

        deviation_data[model] = model_devs
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    return summary_df, deviation_data


# ═══════════════════════════════════════════════════════════════════════════════
#  Figures
# ═══════════════════════════════════════════════════════════════════════════════

def _save_fig(fig, out_dir: Path, basename: str):
    """Save figure as both PNG and PDF."""
    fig.savefig(out_dir / f"{basename}.png", dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(out_dir / f"{basename}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {basename}.png / .pdf")


def figure1_model_comparison(summary: pd.DataFrame, out_dir: Path):
    """Figure 1: Main performance comparison (R² bar chart)."""
    print("\n[Figure 1] Model comparison (R²)")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Architecture-aware polymer modeling performance", fontsize=TITLE_SIZE, fontweight="bold")

    for idx, target in enumerate(["EA", "IP"]):
        ax = axes[idx]
        col = f"{target}_R2"
        models = [m for m in MODEL_KEYS if not np.isnan(summary.loc[summary["model"] == m, col].values[0])]
        values = [summary.loc[summary["model"] == m, col].values[0] for m in models]
        labels = [MODEL_DISPLAY[m] for m in models]
        colors = [COLORS[m] for m in models]

        bars = ax.bar(range(len(models)), values, color=colors, edgecolor="black", linewidth=0.5)

        # Value labels on bars
        for bar, val in zip(bars, values):
            y_pos = bar.get_height()
            offset = 0.01 if val >= 0 else -0.02
            va = "bottom" if val >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2, y_pos + offset,
                f"{val:.3f}", ha="center", va=va, fontsize=BAR_LABEL_SIZE, fontweight="bold"
            )

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("R²")
        ax.set_title(f"{target}")
        ax.axhline(0, color="black", linewidth=0.5, linestyle="-")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _save_fig(fig, out_dir, "figure1_model_comparison")


def figure2_delta_r2(summary: pd.DataFrame, out_dir: Path):
    """Figure 2: Improvement over Frac (ΔR²)."""
    print("\n[Figure 2] ΔR² improvement over Frac")

    frac_row = summary[summary["model"] == "stage2d_frac"]
    if frac_row.empty:
        print("  [skip] No Frac baseline found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Improvement over Frac baseline (ΔR²)", fontsize=TITLE_SIZE, fontweight="bold")

    non_frac_models = [m for m in MODEL_KEYS if m != "stage2d_frac"]

    for idx, target in enumerate(["EA", "IP"]):
        ax = axes[idx]
        col = f"{target}_R2"
        frac_val = frac_row[col].values[0]

        models = [m for m in non_frac_models
                  if not np.isnan(summary.loc[summary["model"] == m, col].values[0])]
        deltas = [summary.loc[summary["model"] == m, col].values[0] - frac_val for m in models]
        labels = [MODEL_DISPLAY[m] for m in models]
        colors = [COLORS[m] for m in models]

        bars = ax.bar(range(len(models)), deltas, color=colors, edgecolor="black", linewidth=0.5)

        for bar, val in zip(bars, deltas):
            y_pos = bar.get_height()
            offset = 0.002 if val >= 0 else -0.002
            va = "bottom" if val >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2, y_pos + offset,
                f"{val:+.4f}", ha="center", va=va, fontsize=BAR_LABEL_SIZE
            )

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("ΔR² (relative to Frac)")
        ax.set_title(f"{target}")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _save_fig(fig, out_dir, "figure2_delta_r2")


def figure3_architecture_deviation_r2(arch_summary: pd.DataFrame, out_dir: Path):
    """Figure 3: Architecture deviation R² — the most important figure."""
    print("\n[Figure 3] Architecture deviation R²")

    if arch_summary.empty:
        print("  [skip] No architecture deviation data available.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Prediction of architecture-induced deviations", fontsize=TITLE_SIZE, fontweight="bold")

    non_frac_models = [m for m in MODEL_KEYS if m != "stage2d_frac"]

    for idx, target in enumerate(["EA", "IP"]):
        ax = axes[idx]
        col = f"{target}_arch_R2"

        models = [m for m in non_frac_models
                  if m in arch_summary["model"].values and
                  not np.isnan(arch_summary.loc[arch_summary["model"] == m, col].values[0])]
        if not models:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{target}")
            continue

        values = [arch_summary.loc[arch_summary["model"] == m, col].values[0] for m in models]
        labels = [MODEL_DISPLAY[m] for m in models]
        colors = [COLORS[m] for m in models]

        # Highlight best
        best_idx = int(np.argmax(values))
        edge_colors = ["gold" if i == best_idx else "black" for i in range(len(models))]
        edge_widths = [2.5 if i == best_idx else 0.5 for i in range(len(models))]

        bars = ax.bar(range(len(models)), values, color=colors,
                      edgecolor=edge_colors, linewidth=edge_widths)

        for bar, val in zip(bars, values):
            y_pos = bar.get_height()
            offset = 0.01 if val >= 0 else -0.02
            va = "bottom" if val >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2, y_pos + offset,
                f"{val:.3f}", ha="center", va=va, fontsize=BAR_LABEL_SIZE, fontweight="bold"
            )

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("R² (architecture deviations)")
        ax.set_title(f"{target}")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _save_fig(fig, out_dir, "figure3_architecture_deviation_r2")


def figure4_deviation_scatter(
    arch_summary: pd.DataFrame,
    deviation_data: Dict[str, Dict[str, pd.DataFrame]],
    out_dir: Path,
):
    """Figure 4: True vs predicted architecture deviations (scatter)."""
    print("\n[Figure 4] Deviation scatter plots")

    if arch_summary.empty or not deviation_data:
        print("  [skip] No architecture deviation data available.")
        return

    # Identify best 2D0 and 2D1 models
    models_2d0 = [m for m in MODEL_KEYS if "2d0" in m]
    models_2d1 = [m for m in MODEL_KEYS if "2d1" in m]

    def _best_model(model_list: List[str], target: str) -> Optional[str]:
        col = f"{target}_arch_R2"
        best, best_val = None, -np.inf
        for m in model_list:
            row = arch_summary[arch_summary["model"] == m]
            if row.empty:
                continue
            val = row[col].values[0]
            if not np.isnan(val) and val > best_val:
                best_val = val
                best = m
        return best

    # We'll pick the best across both targets combined (average)
    def _best_model_overall(model_list: List[str]) -> Optional[str]:
        best, best_val = None, -np.inf
        for m in model_list:
            row = arch_summary[arch_summary["model"] == m]
            if row.empty:
                continue
            ea_val = row["EA_arch_R2"].values[0] if "EA_arch_R2" in row.columns else np.nan
            ip_val = row["IP_arch_R2"].values[0] if "IP_arch_R2" in row.columns else np.nan
            avg = np.nanmean([ea_val, ip_val])
            if avg > best_val:
                best_val = avg
                best = m
        return best

    best_2d0 = _best_model_overall(models_2d0)
    best_2d1 = _best_model_overall(models_2d1)

    if best_2d0 is None and best_2d1 is None:
        print("  [skip] Could not identify best models.")
        return

    plot_models = [(best_2d0, "Best 2D0"), (best_2d1, "Best 2D1")]
    plot_models = [(m, label) for m, label in plot_models if m is not None]

    n_cols = len(plot_models)
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 9))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle("True vs predicted architecture deviations", fontsize=TITLE_SIZE, fontweight="bold")

    for col_idx, (model, model_label) in enumerate(plot_models):
        if model not in deviation_data:
            continue

        for row_idx, target in enumerate(["EA", "IP"]):
            ax = axes[row_idx, col_idx]

            if target not in deviation_data[model]:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{target} — {model_label} ({MODEL_DISPLAY[model]})")
                continue

            dev_df = deviation_data[model][target]
            dt = dev_df["delta_true"].values
            dp = dev_df["delta_pred"].values

            r2_val = compute_r2(dt, dp)
            mae_val = compute_mae(dt, dp)

            # Scatter
            ax.scatter(dt, dp, s=8, alpha=0.3, color=COLORS[model], edgecolors="none")

            # y=x line
            all_vals = np.concatenate([dt, dp])
            lo, hi = all_vals.min(), all_vals.max()
            pad = (hi - lo) * 0.05
            lim = [lo - pad, hi + pad]
            ax.plot(lim, lim, "k--", linewidth=1, alpha=0.7)
            ax.set_xlim(lim)
            ax.set_ylim(lim)

            # Annotations
            ax.text(
                0.05, 0.92, f"R² = {r2_val:.3f}\nMAE = {mae_val:.3f}",
                transform=ax.transAxes, fontsize=LABEL_SIZE,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

            ax.set_xlabel(f"True Δ{target}")
            ax.set_ylabel(f"Predicted Δ{target}")
            ax.set_title(f"{target} — {model_label} ({MODEL_DISPLAY[model]})")
            ax.set_aspect("equal", adjustable="box")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _save_fig(fig, out_dir, "figure4_deviation_scatter")


def figure5_architecture_residuals(
    preds: Dict,
    dataset_df: Optional[pd.DataFrame],
    arch_summary: pd.DataFrame,
    out_dir: Path,
):
    """Figure 5: Architecture-specific residual boxplots."""
    print("\n[Figure 5] Architecture-specific residuals")

    if not preds or dataset_df is None:
        print("  [skip] No predictions or dataset metadata available.")
        return

    # Find architecture column
    possible_arch_cols = ["poly_type", "architecture", "arch"]
    arch_col = None
    for c in possible_arch_cols:
        if c in dataset_df.columns:
            arch_col = c
            break
    if arch_col is None:
        print("  [skip] No architecture column found in dataset.")
        return

    # Pick best 2D0 and best 2D1 (average across targets)
    models_2d0 = [m for m in MODEL_KEYS if "2d0" in m]
    models_2d1 = [m for m in MODEL_KEYS if "2d1" in m]

    def _best(model_list):
        if arch_summary.empty:
            return model_list[0] if model_list else None
        best, best_val = None, -np.inf
        for m in model_list:
            row = arch_summary[arch_summary["model"] == m]
            if row.empty:
                continue
            vals = []
            for t in ["EA", "IP"]:
                col = f"{t}_arch_R2"
                if col in row.columns:
                    v = row[col].values[0]
                    if not np.isnan(v):
                        vals.append(v)
            avg = np.mean(vals) if vals else -np.inf
            if avg > best_val:
                best_val = avg
                best = m
        return best

    best_2d0 = _best(models_2d0)
    best_2d1 = _best(models_2d1)

    compare_models = ["stage2d_frac"]
    if best_2d0:
        compare_models.append(best_2d0)
    if best_2d1:
        compare_models.append(best_2d1)
    compare_models = [m for m in compare_models if m in preds]

    if not compare_models:
        print("  [skip] No models with predictions available.")
        return

    arch_labels_sorted = sorted(ARCH_DISPLAY.keys())
    n_arch = len(arch_labels_sorted)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle("Architecture-specific prediction residuals", fontsize=TITLE_SIZE, fontweight="bold")

    for row_idx, target in enumerate(["EA", "IP"]):
        ax = axes[row_idx]

        box_data = []
        box_positions = []
        box_colors = []
        tick_labels = []
        group_centers = []

        n_models = len(compare_models)
        width = 0.7 / n_models

        for arch_idx, arch_key in enumerate(arch_labels_sorted):
            arch_display = ARCH_DISPLAY[arch_key]
            center = arch_idx * 2

            for m_idx, model in enumerate(compare_models):
                if target not in preds[model]:
                    continue

                df_pred = preds[model][target].copy()
                if "test_id" not in df_pred.columns:
                    continue

                try:
                    indices = df_pred["test_id"].astype(int).values
                except (ValueError, TypeError):
                    continue

                valid = (indices >= 0) & (indices < len(dataset_df))
                meta_arch = dataset_df.iloc[indices[valid]][arch_col].values
                yt = df_pred.iloc[np.where(valid)[0]]["y_true"].astype(float).values
                yp = df_pred.iloc[np.where(valid)[0]]["y_pred"].astype(float).values

                # Filter to this architecture
                # arch_col might be string or numeric
                if pd.api.types.is_numeric_dtype(dataset_df[arch_col]):
                    arch_mask = meta_arch == arch_key
                else:
                    # Map string to int
                    arch_int_map = {v: k for k, v in ARCH_DISPLAY.items()}
                    mapped = np.array([ARCH_LABEL_MAP.get(str(a).lower(), -1) for a in meta_arch])
                    arch_mask = mapped == arch_key

                residuals = yt[arch_mask] - yp[arch_mask]
                if len(residuals) == 0:
                    continue

                pos = center + (m_idx - n_models / 2 + 0.5) * width
                box_data.append(residuals)
                box_positions.append(pos)
                box_colors.append(COLORS[model])

            group_centers.append(center)
            tick_labels.append(arch_display)

        if box_data:
            bp = ax.boxplot(
                box_data, positions=box_positions, widths=width * 0.85,
                patch_artist=True, showfliers=False, medianprops=dict(color="black", linewidth=1.5)
            )
            for patch, color in zip(bp["boxes"], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_xticks(group_centers)
        ax.set_xticklabels(tick_labels)
        ax.set_ylabel(f"Residual ({target})")
        ax.set_title(f"{target}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Legend
        if row_idx == 0:
            legend_handles = [
                plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[m], alpha=0.7,
                              edgecolor="black", linewidth=0.5)
                for m in compare_models
            ]
            legend_labels = [MODEL_DISPLAY[m] for m in compare_models]
            ax.legend(legend_handles, legend_labels, loc="upper right", fontsize=TICK_SIZE)

    fig.tight_layout()
    _save_fig(fig, out_dir, "figure5_architecture_residuals")


def figure6_alpha_analysis(pred_dir: Path, out_dir: Path):
    """Figure 6 (optional): Alpha analysis if logging data exists."""
    print("\n[Figure 6] Alpha analysis (optional)")

    # Look for alpha log files
    alpha_dir = pred_dir.parent if pred_dir.exists() else PROJECT_ROOT / "predictions" / "HPG2Stage"
    alpha_files = list(alpha_dir.glob("*alpha*")) if alpha_dir.exists() else []

    # Also check results dir for any alpha logs
    results_alpha_dir = PROJECT_ROOT / "results" / "HPG2Stage"
    if results_alpha_dir.exists():
        alpha_files.extend(results_alpha_dir.glob("*alpha*"))

    if not alpha_files:
        print("  [skip] No alpha logging data found. Skipping gracefully.")
        return

    # If alpha files exist, attempt to load and plot
    # This is a best-effort implementation
    print(f"  Found {len(alpha_files)} alpha file(s). Attempting to parse...")

    # Placeholder: future alpha logging integration
    print("  [info] Alpha analysis requires structured alpha logging from training.")
    print("         Add --log_alpha flag to training for future runs.")


# ═══════════════════════════════════════════════════════════════════════════════
#  Summary Tables & Report
# ═══════════════════════════════════════════════════════════════════════════════

def export_summary_tables(
    summary: pd.DataFrame,
    arch_summary: pd.DataFrame,
    out_dir: Path,
):
    """Export model_summary.csv and architecture_deviation_summary.csv."""
    print("\n[Tables] Exporting summary CSVs")

    # Model summary
    summary_out = summary.copy()
    summary_out["model"] = summary_out["model"].map(lambda m: MODEL_DISPLAY.get(m, m))
    summary_out.to_csv(out_dir / "model_summary.csv", index=False)
    print(f"  Saved: model_summary.csv")

    # Architecture deviation summary
    if not arch_summary.empty:
        arch_out = arch_summary.copy()
        arch_out["model"] = arch_out["model"].map(lambda m: MODEL_DISPLAY.get(m, m))
        arch_out.to_csv(out_dir / "architecture_deviation_summary.csv", index=False)
        print(f"  Saved: architecture_deviation_summary.csv")


def print_final_report(summary: pd.DataFrame, arch_summary: pd.DataFrame):
    """Print concise textual summary for experiment notes."""
    print("\n" + "=" * 70)
    print("  STAGE 2D ANALYSIS REPORT")
    print("=" * 70)

    # Best overall models
    for target in ["EA", "IP"]:
        col = f"{target}_R2"
        valid = summary.dropna(subset=[col])
        if valid.empty:
            print(f"\n  Best overall {target} model: N/A")
            continue
        best_row = valid.loc[valid[col].idxmax()]
        print(f"\n  Best overall {target} model: {MODEL_DISPLAY[best_row['model']]} "
              f"(R² = {best_row[col]:.4f})")

    # Best architecture-deviation models
    if not arch_summary.empty:
        for target in ["EA", "IP"]:
            col = f"{target}_arch_R2"
            if col not in arch_summary.columns:
                continue
            valid = arch_summary.dropna(subset=[col])
            if valid.empty:
                print(f"\n  Best arch-deviation {target} model: N/A")
                continue
            best_row = valid.loc[valid[col].idxmax()]
            print(f"\n  Best arch-deviation {target} model: {MODEL_DISPLAY[best_row['model']]} "
                  f"(R² = {best_row[col]:.4f})")

    # Improvement over Frac
    frac_row = summary[summary["model"] == "stage2d_frac"]
    if not frac_row.empty:
        print("\n  Improvement over Frac baseline:")
        for target in ["EA", "IP"]:
            col = f"{target}_R2"
            frac_val = frac_row[col].values[0]
            non_frac = summary[summary["model"] != "stage2d_frac"].dropna(subset=[col])
            if non_frac.empty:
                continue
            best = non_frac.loc[non_frac[col].idxmax()]
            delta = best[col] - frac_val
            print(f"    {target}: ΔR² = {delta:+.4f} "
                  f"(best: {MODEL_DISPLAY[best['model']]})")

    # 2D-1 vs 2D-0 comparison
    print("\n  2D-1 vs 2D-0 comparison:")
    for target in ["EA", "IP"]:
        col = f"{target}_R2"
        models_2d0 = summary[summary["model"].str.contains("2d0")].dropna(subset=[col])
        models_2d1 = summary[summary["model"].str.contains("2d1")].dropna(subset=[col])
        if models_2d0.empty or models_2d1.empty:
            print(f"    {target}: Insufficient data")
            continue
        best_2d0 = models_2d0[col].max()
        best_2d1 = models_2d1[col].max()
        winner = "2D-1" if best_2d1 > best_2d0 else "2D-0"
        print(f"    {target}: Best 2D-0 R² = {best_2d0:.4f}, "
              f"Best 2D-1 R² = {best_2d1:.4f} → {winner} wins")

    print("\n" + "=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="HPG2Stage Stage 2D analysis — architecture-aware modeling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR_DEFAULT,
                        help="Directory with stage2d result CSVs")
    parser.add_argument("--pred_dir", type=Path, default=PRED_DIR_DEFAULT,
                        help="Directory with stage2d prediction .npz files")
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR_DEFAULT,
                        help="Output directory for figures and tables")
    parser.add_argument("--data_dir", type=Path, default=None,
                        help="Data directory (for dataset metadata)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results dir : {args.results_dir}")
    print(f"Predictions : {args.pred_dir}")
    print(f"Output dir  : {args.out_dir}")

    # ── Load results ──
    print("\n[Loading] Result CSVs...")
    try:
        df_results = load_results(args.results_dir)
        print(f"  Loaded {len(df_results)} rows across "
              f"{df_results['model'].nunique()} models")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    # ── Build summary from fold-level results ──
    summary = build_summary_from_results(df_results)
    print("\n[Summary] Model performance (mean across folds):")
    print(summary.to_string(index=False))

    # ── Load predictions (for architecture analysis) ──
    print("\n[Loading] Prediction .npz files...")
    preds = load_predictions(args.pred_dir)
    if preds:
        print(f"  Loaded predictions for {len(preds)} models")
        # If predictions are available, recompute metrics from pooled predictions
        summary_pred = build_summary_from_predictions(preds)
        # Use prediction-based summary if it has valid data
        has_pred_data = summary_pred.dropna(subset=["EA_R2", "IP_R2"], how="all")
        if not has_pred_data.empty:
            print("  Using prediction-based metrics (pooled across splits)")
            summary = summary_pred
    else:
        print("  No prediction files found. Using fold-averaged results.")

    # ── Load dataset metadata ──
    print("\n[Loading] Dataset metadata...")
    dataset_df = load_dataset_metadata(args.data_dir)
    if dataset_df is not None:
        print(f"  Loaded ea_ip dataset with {len(dataset_df)} rows")
    else:
        print("  [warn] Could not load ea_ip.csv — architecture analysis will be limited")

    # ── Compute architecture deviations ──
    arch_summary, deviation_data = compute_architecture_deviations(preds, dataset_df)
    if not arch_summary.empty:
        print("\n[Architecture Deviations] Summary:")
        print(arch_summary.to_string(index=False))

    # ── Generate figures ──
    figure1_model_comparison(summary, args.out_dir)
    figure2_delta_r2(summary, args.out_dir)
    figure3_architecture_deviation_r2(arch_summary, args.out_dir)
    figure4_deviation_scatter(arch_summary, deviation_data, args.out_dir)
    figure5_architecture_residuals(preds, dataset_df, arch_summary, args.out_dir)
    figure6_alpha_analysis(args.pred_dir, args.out_dir)

    # ── Export tables ──
    export_summary_tables(summary, arch_summary, args.out_dir)

    # ── Final report ──
    print_final_report(summary, arch_summary)

    print(f"\nAll outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
