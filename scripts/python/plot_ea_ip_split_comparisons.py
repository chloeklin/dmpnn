#!/usr/bin/env python3
"""
Generate 4 comparison figures for the ea_ip dataset:
  1. Tabular (random) vs Tabular (a_held_out)
  2. Graph (random) vs Graph (a_held_out)
  3. Tabular (random) vs Graph (random)
  4. Tabular (a_held_out) vs Graph (a_held_out)

For each comparison, selects the best model per target (lowest RMSE mean)
and shows grouped bars with error bars across both targets.

Usage:
    python3 scripts/python/plot_ea_ip_split_comparisons.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# ── Paths ──────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
RESULTS_DIR = PROJECT_DIR / "plots" / "combined" / "datasets_ea_ip"
OUTPUT_DIR = PROJECT_DIR / "plots" / "ea_ip_comparisons"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = RESULTS_DIR / "ea_ip_consolidated_results.csv"
if not CSV_PATH.exists():
    print(f"Error: {CSV_PATH} not found. Run visualize_combined_results.py --dataset ea_ip first.")
    sys.exit(1)

# ── Load data ──────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

# ── Categorise rows ────────────────────────────────────────────────
is_held = df["features"].str.contains(r"\[a_held_out\]", regex=True)
is_graph = df["method"].str.startswith("Graph_")
is_tabular = df["method"] == "Tabular"

tab_rand = df[is_tabular & ~is_held].copy()
tab_held = df[is_tabular & is_held].copy()
graph_rand = df[is_graph & ~is_held].copy()
graph_held = df[is_graph & is_held].copy()


def best_per_target(subset: pd.DataFrame, label: str) -> pd.DataFrame:
    """Return the single best (lowest rmse_mean) row per target."""
    if subset.empty:
        return pd.DataFrame()
    best_idx = subset.groupby("target")["rmse_mean"].idxmin()
    best = subset.loc[best_idx].copy()
    best["group_label"] = label
    return best


def make_comparison_plot(
    group_a: pd.DataFrame,
    group_b: pd.DataFrame,
    label_a: str,
    label_b: str,
    title: str,
    filename: str,
    metrics=("rmse", "mae", "r2"),
):
    """Create a multi-metric comparison figure for two groups."""
    targets = sorted(
        set(group_a["target"].unique()) | set(group_b["target"].unique())
    )
    n_targets = len(targets)

    for metric in metrics:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"

        if mean_col not in group_a.columns and mean_col not in group_b.columns:
            continue

        fig, ax = plt.subplots(figsize=(max(6, n_targets * 2.5), 5))

        x = np.arange(n_targets)
        width = 0.30

        # Nature / Science journal-style colours
        colour_a = "#0C5DA5"  # deep blue
        colour_b = "#FF2C00"  # vermilion
        err_kw = dict(elinewidth=1.5, capsize=4, capthick=1.5, ecolor="#333333")

        vals_a, errs_a, details_a = [], [], []
        vals_b, errs_b, details_b = [], [], []

        for t in targets:
            row_a = group_a[group_a["target"] == t]
            row_b = group_b[group_b["target"] == t]

            if not row_a.empty:
                vals_a.append(row_a[mean_col].iloc[0])
                errs_a.append(row_a[std_col].iloc[0] if std_col in row_a.columns else 0)
                details_a.append(f"{row_a['model'].iloc[0]}\n({row_a['features'].iloc[0]})")
            else:
                vals_a.append(np.nan)
                errs_a.append(np.nan)
                details_a.append("")

            if not row_b.empty:
                vals_b.append(row_b[mean_col].iloc[0])
                errs_b.append(row_b[std_col].iloc[0] if std_col in row_b.columns else 0)
                details_b.append(f"{row_b['model'].iloc[0]}\n({row_b['features'].iloc[0]})")
            else:
                vals_b.append(np.nan)
                errs_b.append(np.nan)
                details_b.append("")

        ax.bar(
            x - width / 2, vals_a, width, yerr=errs_a,
            label=label_a, color=colour_a, edgecolor="white", linewidth=0.5,
            error_kw=err_kw,
        )
        ax.bar(
            x + width / 2, vals_b, width, yerr=errs_b,
            label=label_b, color=colour_b, edgecolor="white", linewidth=0.5,
            error_kw=err_kw,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(targets, fontsize=11)
        ylabel = "RMSE" if metric == "rmse" else ("MAE" if metric == "mae" else "R²")
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"EA_IP – {title} ({metric.upper()})", fontsize=13, pad=10)
        ax.legend(
            fontsize=10, frameon=True, edgecolor="#cccccc",
            loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2,
        )
        ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.22)
        out = OUTPUT_DIR / f"ea_ip_{filename}_{metric}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")


# ── Select best per target for each category ───────────────────────
best_tab_rand = best_per_target(tab_rand, "Tabular")
best_tab_held = best_per_target(tab_held, "Tabular [a_held_out]")
best_graph_rand = best_per_target(graph_rand, "Graph")
best_graph_held = best_per_target(graph_held, "Graph [a_held_out]")

print("\n=== Best models selected ===")
for label, grp in [
    ("Tabular (random)", best_tab_rand),
    ("Tabular (held-out)", best_tab_held),
    ("Graph (random)", best_graph_rand),
    ("Graph (held-out)", best_graph_held),
]:
    if grp.empty:
        print(f"  {label}: NO DATA")
    else:
        for _, r in grp.iterrows():
            print(f"  {label} | {r['target']:25s} | {r['model']:15s} | {r['features']:35s} | RMSE={r['rmse_mean']:.4f}")
print()

# ── Generate the 4 comparisons ─────────────────────────────────────

# 1. Tabular (random) vs Tabular (a_held_out)
make_comparison_plot(
    best_tab_rand, best_tab_held,
    "Tabular (random)", "Tabular (held-out)",
    "Tabular: Random vs Held-Out Split",
    "tabular_vs_tabular_heldout",
)

# 2. Graph (random) vs Graph (a_held_out)
make_comparison_plot(
    best_graph_rand, best_graph_held,
    "Graph (random)", "Graph (held-out)",
    "Graph: Random vs Held-Out Split",
    "graph_vs_graph_heldout",
)

# 3. Tabular (random) vs Graph (random)
make_comparison_plot(
    best_tab_rand, best_graph_rand,
    "Tabular (random)", "Graph (random)",
    "Tabular vs Graph (Random Split)",
    "tabular_vs_graph_random",
)

# 4. Tabular (a_held_out) vs Graph (a_held_out)
make_comparison_plot(
    best_tab_held, best_graph_held,
    "Tabular (held-out)", "Graph (held-out)",
    "Tabular vs Graph (Held-Out Split)",
    "tabular_vs_graph_heldout",
)

print(f"\nAll figures saved to: {OUTPUT_DIR}")
