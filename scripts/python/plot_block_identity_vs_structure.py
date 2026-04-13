#!/usr/bin/env python3
"""
Block copolymer: Monomer Identity vs Chemical Structure comparison plot.

Compares:
  - Identity Baseline (mix & interact modes) — knows only *which* monomers
  - Best Tabular model (XGB on descriptors)  — knows descriptor chemistry
  - DMPNN / GIN / GAT + Mixture             — full graph-chemical representation

Task is multiclass classification (phase_label).
Primary metric: F1-macro  (secondary: Accuracy)
Mean ± std across 5-fold CV.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
results_dir = Path(__file__).parent.parent.parent / "results"
output_dir  = Path(__file__).parent.parent.parent / "plots" / "block_identity_vs_structure"
output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper: rename graph metric columns to canonical names
# ---------------------------------------------------------------------------
def _normalise_graph_cols(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for c in df.columns:
        lc = c.lower()
        if "accuracy" in lc and c != "acc":
            rename[c] = "acc"
        elif "f1" in lc and c != "f1_macro":
            rename[c] = "f1_macro"
        elif "roc" in lc and c != "roc_auc":
            rename[c] = "roc_auc"
        elif "logloss" in lc and c != "logloss":
            rename[c] = "logloss"
    if rename:
        df = df.rename(columns=rename)
    # drop the test/ prefix style columns that remain
    df.columns = [c.replace("test/", "") for c in df.columns]
    return df

# ---------------------------------------------------------------------------
# Load Identity Baseline (mix and interact modes)
# ---------------------------------------------------------------------------
records = []

for mode, label in [("mix", "Identity\n(Mix)")]:
    fpath = results_dir / "IdentityBaseline" / f"block__copoly_{mode}_results.csv"
    if not fpath.exists():
        print(f"[WARN] Missing: {fpath}")
        continue
    df = pd.read_csv(fpath)
    df = df[df["target"] == "phase_label"] if "target" in df.columns else df
    for metric in ["acc", "f1_macro"]:
        if metric in df.columns:
            vals = df[metric].dropna().values
            records.append({
                "label": label,
                "metric": metric,
                "mean": vals.mean(),
                "std": vals.std(),
                "group": "identity",
            })

# ---------------------------------------------------------------------------
# Load Tabular — pick best sub-model per metric (highest mean f1_macro)
# ---------------------------------------------------------------------------
tab_file = results_dir / "tabular" / "block_descriptors.csv"
if tab_file.exists():
    df_tab = pd.read_csv(tab_file)
    df_tab = df_tab[df_tab["target"] == "phase_label"] if "target" in df_tab.columns else df_tab
    # Find best sub-model by mean f1_macro
    best_model = (df_tab.groupby("model")["f1_macro"].mean()
                  .idxmax() if "f1_macro" in df_tab.columns else None)
    for metric in ["acc", "f1_macro"]:
        if metric not in df_tab.columns:
            continue
        # Use best-model filter
        df_best = df_tab[df_tab["model"] == best_model] if best_model else df_tab
        vals = df_best[metric].dropna().values
        records.append({
            "label": f"Tabular\n({best_model})",
            "metric": metric,
            "mean": vals.mean(),
            "std": vals.std(),
            "group": "tabular",
        })

# ---------------------------------------------------------------------------
# Load Graph + Mixture models
# ---------------------------------------------------------------------------
graph_models = [("DMPNN", "DMPNN\n+Mixture"),
                ("GIN",   "GIN\n+Mixture"),
                ("GAT",   "GAT\n+Mixture")]

for model_name, label in graph_models:
    fpath = results_dir / model_name / "block__copoly_mix_results.csv"
    if not fpath.exists():
        print(f"[WARN] Missing: {fpath}")
        continue
    df = pd.read_csv(fpath)
    df = _normalise_graph_cols(df)
    df = df[df["target"] == "phase_label"] if "target" in df.columns else df
    for metric in ["acc", "f1_macro"]:
        if metric in df.columns:
            vals = df[metric].dropna().values
            records.append({
                "label": label,
                "metric": metric,
                "mean": vals.mean(),
                "std": vals.std(),
                "group": "graph",
            })

# ---------------------------------------------------------------------------
# Build summary DataFrame
# ---------------------------------------------------------------------------
summary = pd.DataFrame(records)
print("\nSummary:")
print(summary.to_string(index=False))

# ---------------------------------------------------------------------------
# Define display order and colours
# ---------------------------------------------------------------------------
LABEL_ORDER = [
    "Identity\n(Mix)",
    f"Tabular\n({best_model})",
    "DMPNN\n+Mixture",
    "GIN\n+Mixture",
    "GAT\n+Mixture",
]
# Keep only labels that have data
LABEL_ORDER = [l for l in LABEL_ORDER if l in summary["label"].values]

GROUP_COLORS = {
    "identity": "#E67E22",   # orange — highlight
    "tabular":  "#95A5A6",   # neutral grey
    "graph":    "#2980B9",   # blue family
}
# Give graph models slightly different shades
GRAPH_SHADES = ["#1A5276", "#2980B9", "#85C1E9"]
graph_labels_seen = []

bar_colors = []
edge_colors = []
for lbl in LABEL_ORDER:
    row = summary[summary["label"] == lbl].iloc[0]
    grp = row["group"]
    if grp == "graph":
        idx = len(graph_labels_seen)
        bar_colors.append(GRAPH_SHADES[idx % len(GRAPH_SHADES)])
        graph_labels_seen.append(lbl)
    else:
        bar_colors.append(GROUP_COLORS[grp])
    edge_colors.append("black")

# ---------------------------------------------------------------------------
# Plot — two panels: F1-macro (primary) and Accuracy (secondary)
# ---------------------------------------------------------------------------
METRICS = [
    ("f1_macro", "F1-Macro"),
    ("acc",      "Accuracy"),
]

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
fig.suptitle(
    "Block Copolymer Phase Behaviour: Monomer Identity vs Chemical Structure",
    fontsize=13, fontweight="bold", y=1.02,
)

x = np.arange(len(LABEL_ORDER))
bar_width = 0.55

for ax, (metric, metric_label) in zip(axes, METRICS):
    means, stds = [], []
    for lbl in LABEL_ORDER:
        row = summary[(summary["label"] == lbl) & (summary["metric"] == metric)]
        if row.empty:
            means.append(np.nan)
            stds.append(0.0)
        else:
            means.append(row.iloc[0]["mean"])
            stds.append(row.iloc[0]["std"])

    bars = ax.bar(
        x, means, bar_width,
        yerr=stds,
        color=bar_colors,
        edgecolor=edge_colors,
        linewidth=0.8,
        capsize=5,
        error_kw={"elinewidth": 1.2, "ecolor": "black"},
        alpha=0.88,
        zorder=3,
    )

    # Annotate bar tops with mean value
    for rect, mean_val in zip(bars, means):
        if not np.isnan(mean_val):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + max(stds) * 0.05 + 0.005,
                f"{mean_val:.3f}",
                ha="center", va="bottom", fontsize=8, color="#2C3E50",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_ORDER, fontsize=9)
    ax.set_ylabel(metric_label, fontsize=11, fontweight="bold")
    ax.set_title(metric_label, fontsize=11, fontweight="bold")
    ax.set_ylim(0, min(1.0, max(m for m in means if not np.isnan(m)) * 1.30))
    ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    # Vertical separators
    ax.axvline(x=0.5, color="grey", linestyle=":", linewidth=1.0, alpha=0.6)  # after identity
    ax.axvline(x=1.5, color="grey", linestyle=":", linewidth=1.0, alpha=0.6)  # after tabular

    # Section labels
    id_span   = 0
    tab_span  = 1
    graph_span = (2 + 3 + 4) / 3 if len(LABEL_ORDER) > 4 else (2 + 3) / 2
    ylim_top = ax.get_ylim()[1]
    ax.text(id_span,  ylim_top * 0.98, "Identity", ha="center", va="top",
            fontsize=8, color=GROUP_COLORS["identity"], fontstyle="italic")
    ax.text(tab_span, ylim_top * 0.98, "Tabular",  ha="center", va="top",
            fontsize=8, color="#717D7E",               fontstyle="italic")
    ax.text(graph_span, ylim_top * 0.98, "Graph",  ha="center", va="top",
            fontsize=8, color=GRAPH_SHADES[1],         fontstyle="italic")

# Legend
legend_handles = [
    mpatches.Patch(facecolor=GROUP_COLORS["identity"], edgecolor="black",
                   linewidth=0.8, label="Identity Baseline"),
    mpatches.Patch(facecolor=GROUP_COLORS["tabular"],  edgecolor="black",
                   linewidth=0.8, label=f"Tabular ({best_model})"),
]
for i, (mn, lbl) in enumerate(graph_models):
    legend_handles.append(
        mpatches.Patch(facecolor=GRAPH_SHADES[i % len(GRAPH_SHADES)],
                       edgecolor="black", linewidth=0.8, label=lbl.replace("\n", " "))
    )
fig.legend(handles=legend_handles, loc="upper center",
           bbox_to_anchor=(0.5, -0.04), ncol=len(legend_handles),
           frameon=True, fontsize=9)

plt.tight_layout()
out_path = output_dir / "block_identity_vs_structure.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out_path}")

# ---------------------------------------------------------------------------
# Save consolidated CSV
# ---------------------------------------------------------------------------
csv_path = output_dir / "block_identity_vs_structure_summary.csv"
summary.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")
