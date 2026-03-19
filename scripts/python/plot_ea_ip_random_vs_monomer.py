#!/usr/bin/env python3
"""
Plot EA/IP dataset: Random split vs Monomer (a_held_out) split comparison.

For each metric (mae, rmse, r2): one figure with 2 panels.
  Left panel:  Random split
  Right panel: Monomer split (held-out monomer)

Models are grouped into 3 categories:
  Identity  → IdentityBaseline
  Graph     → DMPNN, GAT, GIN
  Tabular   → Linear, RF, XGB

For each model, two bars are shown (one per target: EA vs SHE, IP vs SHE).
Best configuration (lowest RMSE) is selected per model/split/target.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
PROJECT_DIR  = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR  = PROJECT_DIR / "results"
CONSOL_CSV   = PROJECT_DIR / "plots" / "combined" / "datasets_ea_ip" / "ea_ip_consolidated_results.csv"
OUTPUT_DIR   = PROJECT_DIR / "plots" / "ea_ip_random_vs_monomer"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── colour palette ─────────────────────────────────────────────────────────
TARGET_COLORS = {
    "EA vs SHE (eV)": "#2171b5",   # blue
    "IP vs SHE (eV)": "#cb181d",   # red
}
TARGET_HATCHES = {
    "EA vs SHE (eV)": "",
    "IP vs SHE (eV)": "///",
}

CATEGORY_COLORS = {
    "Identity": "#737373",
    "Graph":    "#2171b5",
    "Tabular":  "#238b45",
}

# ── model ordering ─────────────────────────────────────────────────────────
# (category, model_name, copolymer_mode, poly_type)
# poly_type=True + mode=None → auto-pick best poly_type variant (raw result files)
MODEL_ORDER = [
    ("Identity", "IdentityBaseline", "mix",     False),
    ("Identity", "IdentityBaseline", "interact", False),
    ("Identity", "IdentityBaseline", None,       True),   # best of mix/interact +poly_type
    ("Graph",    "DMPNN",           "mix",       False),
    ("Graph",    "DMPNN",           "interact",  False),
    ("Graph",    "DMPNN",           None,        True),   # best of mix_meta/interact_meta +poly_type
    ("Graph",    "GAT",             "mix",       False),
    ("Graph",    "GAT",             "interact",  False),
    ("Graph",    "GAT",             None,        True),
    ("Graph",    "GIN",             "mix",       False),
    ("Graph",    "GIN",             "interact",  False),
    ("Graph",    "GIN",             None,        True),
    ("Graph",    "wDMPNN",          None,        False),  # wDMPNN doesn't use copoly modes
    ("Graph",    "HPG",             None,        False),  # HPG original (no desc)
    ("Graph",    "HPG",             "desc",      False),  # HPG with desc
    ("Graph",    "HPG",             "desc",      True),   # HPG with desc+poly_type
    ("Tabular",  "Linear",          None,        False),
    ("Tabular",  "Linear",          None,        True),
    ("Tabular",  "RF",              None,        False),
    ("Tabular",  "RF",              None,        True),
    ("Tabular",  "XGB",             None,        False),
    ("Tabular",  "XGB",             None,        True),
]

def model_label(model_name: str, mode: str | None, poly_type: bool = False) -> str:
    base = {"IdentityBaseline": "Identity"}.get(model_name, model_name)
    suffix = "\n+PT" if poly_type else ""
    if mode is None:
        return f"{base}{suffix}"
    # HPG uses 'desc' to indicate descriptor inclusion, show as +desc
    if model_name == "HPG" and mode == "desc":
        return f"{base}\n+desc{suffix}"
    # Other models use mode for copolymer modes (mix, interact)
    return f"{base}\n({mode}){suffix}"

METRICS = ["mae", "rmse", "r2"]
METRIC_LABELS = {"mae": "MAE", "rmse": "RMSE", "r2": "R²"}
LOWER_IS_BETTER = {"mae", "rmse"}


# ── helpers ─────────────────────────────────────────────────────────────────

IDENTITY_RESULTS_DIR = PROJECT_DIR / "results" / "IdentityBaseline"
TABULAR_RESULTS_DIR  = PROJECT_DIR / "results" / "tabular"


def load_raw_identity_poly_type(mode: str | None, split: str, target: str,
                                metric: str) -> tuple[float, float] | tuple[None, None]:
    """Load poly_type IdentityBaseline results. If mode is None, auto-pick best (lowest RMSE)."""
    suffix = "__a_held_out" if split == "monomer" else ""
    modes_to_try = [mode] if mode is not None else ["mix", "interact"]
    best_mean, best_std, best_rmse = None, None, float("inf")
    for m in modes_to_try:
        fname = f"ea_ip__copoly_{m}__poly_type{suffix}_results.csv"
        fpath = IDENTITY_RESULTS_DIR / fname
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath)
        df_t = df[df["target"].astype(str).str.strip() == target]
        if df_t.empty or metric not in df_t.columns:
            continue
        rmse_col = "test/rmse" if "test/rmse" in df_t.columns else metric
        rmse = float(df_t[rmse_col].mean()) if rmse_col in df_t.columns else float(df_t[metric].mean())
        if rmse < best_rmse:
            best_rmse = rmse
            best_mean = float(df_t[metric].mean())
            best_std  = float(df_t[metric].std())
    if best_mean is None:
        return None, None
    return best_mean, best_std


def load_raw_wdmpnn(split: str, target: str, metric: str) -> tuple[float, float] | tuple[None, None]:
    """Load wDMPNN results from raw result files. wDMPNN doesn't use copolymer modes."""
    suffix = "__a_held_out" if split == "monomer" else ""
    fname = f"ea_ip{suffix}__target_{target}_results.csv"
    fpath = RESULTS_DIR / "wDMPNN" / fname
    if not fpath.exists():
        return None, None
    df = pd.read_csv(fpath)
    if df.empty:
        return None, None
    metric_col = f"test/{metric}"
    if metric_col not in df.columns:
        return None, None
    return float(df[metric_col].mean()), float(df[metric_col].std())


def load_raw_hpg(mode: str | None, poly_type: bool, split: str, target: str, 
                 metric: str) -> tuple[float, float] | tuple[None, None]:
    """Load HPG results from raw result files. 
    
    mode: None (original), 'desc' (with descriptors)
    poly_type: True if using poly_type features
    """
    split_suffix = "__a_held_out" if split == "monomer" else ""
    desc_suffix = "__desc" if mode == "desc" else ""
    pt_suffix = "__poly_type" if poly_type else ""
    fname = f"ea_ip{desc_suffix}{pt_suffix}{split_suffix}__target_{target}_results.csv"
    fpath = RESULTS_DIR / "HPG" / fname
    if not fpath.exists():
        return None, None
    df = pd.read_csv(fpath)
    if df.empty:
        return None, None
    metric_col = f"test/{metric}"
    if metric_col not in df.columns:
        return None, None
    return float(df[metric_col].mean()), float(df[metric_col].std())


def load_raw_tabular_poly_type(model_name: str, split: str, target: str,
                               metric: str) -> tuple[float, float] | tuple[None, None]:
    """Load poly_type tabular results directly from raw result files (when available)."""
    suffix = "__a_held_out" if split == "monomer" else ""
    fname  = f"ea_ip_poly_type{suffix}.csv"
    fpath  = TABULAR_RESULTS_DIR / fname
    if not fpath.exists():
        return None, None
    df = pd.read_csv(fpath)
    if "model" in df.columns:
        df = df[df["model"].astype(str) == model_name]
    if "target" in df.columns:
        df = df[df["target"].astype(str).str.strip() == target]
    if df.empty or metric not in df.columns:
        return None, None
    return float(df[metric].mean()), float(df[metric].std())


def load_raw_graph_poly_type(model_name: str, mode: str | None, split: str, target: str,
                             metric: str) -> tuple[float, float] | tuple[None, None]:
    """Load poly_type graph model results. If mode is None, auto-pick best (lowest RMSE).
    
    Graph models with poly_type use {mode}_meta naming (e.g., mix_meta, interact_meta).
    """
    suffix = "__a_held_out" if split == "monomer" else ""
    modes_to_try = [mode] if mode is not None else ["mix", "interact"]
    best_mean, best_std, best_rmse = None, None, float("inf")
    for m in modes_to_try:
        fname = f"ea_ip__copoly_{m}_meta__poly_type{suffix}__target_{target}_results.csv"
        fpath = RESULTS_DIR / model_name / fname
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath)
        if df.empty:
            continue
        rmse_col = "test/rmse"
        metric_col = f"test/{metric}"
        if metric_col not in df.columns:
            continue
        rmse = float(df[rmse_col].mean()) if rmse_col in df.columns else float(df[metric_col].mean())
        if rmse < best_rmse:
            best_rmse = rmse
            best_mean = float(df[metric_col].mean())
            best_std  = float(df[metric_col].std())
    if best_mean is None:
        return None, None
    return best_mean, best_std


def load_consolidated() -> pd.DataFrame:
    if not CONSOL_CSV.exists():
        raise FileNotFoundError(
            f"Consolidated CSV not found:\n  {CONSOL_CSV}\n"
            "Run: python3 scripts/python/visualize_combined_results.py --dataset ea_ip"
        )
    df = pd.read_csv(CONSOL_CSV)
    print(f"Loaded {len(df)} rows from consolidated CSV")
    print(f"  columns : {list(df.columns)}")
    print(f"  methods : {sorted(df['method'].unique())}")
    print(f"  features: {sorted(df['features'].unique())[:10]}...")
    return df


def is_held_out(features_str: str) -> bool:
    return "[a_held_out]" in str(features_str)


def best_row(subset: pd.DataFrame, metric: str = "rmse") -> pd.Series | None:
    """Return the single best row (lowest RMSE mean, or highest R² mean)."""
    if subset.empty:
        return None
    mean_col = f"{metric}_mean"
    if mean_col not in subset.columns:
        mean_col = f"rmse_mean" if "rmse_mean" in subset.columns else None
    if mean_col is None:
        return subset.iloc[0]
    if metric in LOWER_IS_BETTER or mean_col == "rmse_mean":
        return subset.loc[subset[mean_col].idxmin()]
    else:
        return subset.loc[subset[mean_col].idxmax()]


def get_model_data(df: pd.DataFrame, category: str, model_name: str,
                   mode: str | None, target: str, split: str,
                   metric: str, poly_type: bool = False) -> tuple[float, float] | tuple[None, None]:
    """Get (mean, std) for a model/target/split/metric.

    For graph/identity models, ``mode`` selects the copolymer mode
    (e.g. 'mix' or 'interact'). For tabular models, mode is ignored.
    poly_type=True loads directly from raw result files.
    """
    # wDMPNN and HPG always load from raw files (don't use copoly modes)
    if model_name == "wDMPNN":
        return load_raw_wdmpnn(split, target, metric)
    if model_name == "HPG":
        return load_raw_hpg(mode, poly_type, split, target, metric)
    
    # poly_type variants come from raw files, not the consolidated CSV
    if poly_type:
        if category == "Identity":
            return load_raw_identity_poly_type(mode, split, target, metric)
        elif category == "Graph":
            return load_raw_graph_poly_type(model_name, mode, split, target, metric)
        elif category == "Tabular":
            return load_raw_tabular_poly_type(model_name, split, target, metric)
        return None, None

    held = (split == "monomer")

    # Filter by split type
    mask_split = df["features"].apply(is_held_out) == held
    subset = df[mask_split].copy()

    # Filter by target
    subset = subset[subset["target"].astype(str).str.strip() == target]

    if category == "Identity":
        subset = subset[subset["method"].astype(str).str.contains("IdentityBaseline")]
    elif category == "Graph":
        subset = subset[subset["method"].astype(str) == f"Graph_{model_name}"]
    elif category == "Tabular":
        subset = subset[
            (subset["method"].astype(str) == "Tabular") &
            (subset["model"].astype(str) == model_name)
        ]

    # Filter by copolymer mode (graph / identity only)
    if mode is not None:
        subset = subset[subset["features"].astype(str).str.contains(f"({mode})", regex=False)]

    # For tabular: pick the best feature set (lowest RMSE)
    row = best_row(subset, "rmse")
    if row is None:
        return None, None

    mean_col = f"{metric}_mean"
    std_col  = f"{metric}_std"
    if mean_col not in row.index:
        return None, None
    return float(row[mean_col]), float(row.get(std_col, 0) or 0)


# ── plot ─────────────────────────────────────────────────────────────────────

def make_figure(df: pd.DataFrame, metric: str):
    targets = ["EA vs SHE (eV)", "IP vs SHE (eV)"]
    splits  = ["random", "monomer"]
    split_titles = {"random": "Random Split", "monomer": "Monomer Split (held-out)"}

    n_models = len(MODEL_ORDER)
    n_targets = len(targets)
    bar_w = 0.32
    group_gap = 0.2  # extra gap between category groups

    # Compute x positions with category separators
    category_breaks = []   # indices where category changes
    prev_cat = MODEL_ORDER[0][0]
    x_positions = []
    x = 0.0
    for i, (cat, _m, _md, _pt) in enumerate(MODEL_ORDER):
        if cat != prev_cat:
            x += group_gap
            category_breaks.append(i)
            prev_cat = cat
        x_positions.append(x)
        x += n_targets * bar_w + 0.1   # spacing between model groups

    fig, axes = plt.subplots(1, 2, figsize=(22, 6), sharey=True)
    fig.subplots_adjust(wspace=0.05)

    for ax_idx, split in enumerate(splits):
        ax = axes[ax_idx]
        legend_done = (ax_idx > 0)

        for i, (cat, model, mode, pt) in enumerate(MODEL_ORDER):
            x_base = x_positions[i]
            for j, target in enumerate(targets):
                mean, std = get_model_data(df, cat, model, mode, target, split, metric, pt)
                if mean is None:
                    continue
                x_bar = x_base + j * bar_w
                color = TARGET_COLORS[target]
                hatch = TARGET_HATCHES[target]
                label = target if (ax_idx == 0 and i == 0) else None
                ax.bar(
                    x_bar, mean, bar_w,
                    yerr=std if std > 0 else None,
                    color=color, hatch=hatch, alpha=0.85,
                    edgecolor="black", linewidth=0.6,
                    capsize=3, label=label,
                    error_kw={"elinewidth": 0.8, "capthick": 0.8},
                )

        # X ticks: centred on each model group
        tick_pos = [x_positions[i] + (n_targets * bar_w - bar_w) / 2
                    for i in range(n_models)]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(
            [model_label(m, md, pt) for _, m, md, pt in MODEL_ORDER],
            fontsize=7, rotation=45, ha="right"
        )

        # Category shading: axvspan background + label inside plot at top
        y_top = ax.get_ylim()[1]
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        for cat_name, color in CATEGORY_COLORS.items():
            idxs = [i for i, (c, *_) in enumerate(MODEL_ORDER) if c == cat_name]
            if not idxs:
                continue
            x_lo = x_positions[idxs[0]] - 0.1
            x_hi = x_positions[idxs[-1]] + n_targets * bar_w + 0.1
            ax.axvspan(x_lo, x_hi, color=color, alpha=0.07, zorder=0)
            mid = (x_lo + x_hi) / 2
            ax.text(mid, y_top - y_range * 0.03, cat_name,
                    ha="center", va="top", fontsize=9,
                    color=color, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              alpha=0.75, edgecolor="none"))

        # Vertical separators between categories
        for break_idx in category_breaks:
            sep_x = (x_positions[break_idx - 1] + n_targets * bar_w + 0.05 +
                     x_positions[break_idx]) / 2
            ax.axvline(sep_x, color="gray", linestyle=":", linewidth=1, alpha=0.5)

        ax.set_title(split_titles[split], fontsize=12, fontweight="bold")
        if ax_idx == 0:
            ax.set_ylabel(METRIC_LABELS[metric], fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_xlim(x_positions[0] - 0.3,
                    x_positions[-1] + n_targets * bar_w + 0.3)

        # Print missing data
        for cat, model, mode, pt in MODEL_ORDER:
            for target in targets:
                m, _ = get_model_data(df, cat, model, mode, target, split, metric, pt)
                if m is None:
                    label = model_label(model, mode, pt)
                    print(f"  [MISSING] {split:8s} | {label:25s} | {target}")

    # Legend: targets
    handles = [
        mpatches.Patch(facecolor=TARGET_COLORS[t], hatch=TARGET_HATCHES[t],
                       edgecolor="black", linewidth=0.6, label=t)
        for t in targets
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.01),
               ncol=2, fontsize=10, frameon=True)
    fig.suptitle(f"EA/IP — {METRIC_LABELS[metric]}: Random vs Monomer Split",
                 fontsize=13, fontweight="bold", y=1.06)
    fig.tight_layout()

    out = OUTPUT_DIR / f"ea_ip_{metric}_random_vs_monomer.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def collect_all_results(df: pd.DataFrame) -> pd.DataFrame:
    """Collect all model results shown in the figure into a consolidated DataFrame."""
    targets = ["EA vs SHE (eV)", "IP vs SHE (eV)"]
    splits = ["random", "monomer"]
    
    rows = []
    for cat, model, mode, pt in MODEL_ORDER:
        model_label_str = model_label(model, mode, pt).replace("\n", " ")
        for split in splits:
            for target in targets:
                row_data = {
                    "category": cat,
                    "model": model,
                    "copolymer_mode": mode if mode is not None else "N/A",
                    "poly_type": pt,
                    "model_label": model_label_str,
                    "split": split,
                    "target": target,
                }
                
                # Get all metrics
                for metric in METRICS:
                    mean, std = get_model_data(df, cat, model, mode, target, split, metric, pt)
                    row_data[f"{metric}_mean"] = mean
                    row_data[f"{metric}_std"] = std
                
                rows.append(row_data)
    
    results_df = pd.DataFrame(rows)
    return results_df


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    df = load_consolidated()

    for metric in METRICS:
        mean_col = f"{metric}_mean"
        if mean_col not in df.columns:
            print(f"Skipping {metric} — column '{mean_col}' not in CSV")
            continue
        print(f"\nPlotting {metric.upper()} ...")
        make_figure(df, metric)

    print(f"\nAll plots → {OUTPUT_DIR}")
    
    # Save consolidated results
    print(f"\nCollecting consolidated results...")
    consolidated = collect_all_results(df)
    csv_out = OUTPUT_DIR / "ea_ip_random_vs_monomer_consolidated.csv"
    consolidated.to_csv(csv_out, index=False)
    print(f"Saved consolidated results → {csv_out}")
    print(f"  {len(consolidated)} rows × {len(consolidated.columns)} columns")


if __name__ == "__main__":
    main()
