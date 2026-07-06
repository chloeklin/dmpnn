#!/usr/bin/env python3
"""
Fusion Ablation — Per-Fold Publication Figure
===============================================
Reads the existing per-fold metrics CSV and produces a single 2×2 publication-quality
figure showing consistency across the 9 LOMO folds.

Outputs:
  output/fusion_ablation/fusion_ablation_per_fold.png
  output/fusion_ablation/fusion_ablation_per_fold.pdf
"""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / 'output' / 'fusion_ablation' / 'fusion_per_fold_metrics.csv'
OUT_DIR = ROOT / 'output' / 'fusion_ablation'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style config ────────────────────────────────────────────────────────────
MODEL_ORDER = ['Additive (2D1)', 'FiLM', 'NLMix', 'FiLM+NLMix']

COLORS = {
    'Additive (2D1)': '#2ca02c',   # green
    'FiLM':           '#d62728',   # red
    'NLMix':          '#ff7f0e',   # orange
    'FiLM+NLMix':     '#9467bd',   # purple
}

MARKERS = {
    'Additive (2D1)': 'o',
    'FiLM':           's',
    'NLMix':          '^',
    'FiLM+NLMix':     'D',
}

# Emphasise Additive and FiLM; keep NLMix and FiLM+NLMix lighter/thinner
LINEWIDTH = {
    'Additive (2D1)': 2.2,
    'FiLM':           2.2,
    'NLMix':          1.3,
    'FiLM+NLMix':     1.3,
}

MARKERSIZE = {
    'Additive (2D1)': 6.5,
    'FiLM':           6.5,
    'NLMix':          4.5,
    'FiLM+NLMix':     4.5,
}

ALPHA = {
    'Additive (2D1)': 1.0,
    'FiLM':           1.0,
    'NLMix':          0.75,
    'FiLM+NLMix':     0.75,
}

FOLD_HIGHLIGHT = 6  # difficult held-out monomer fold
N_FOLDS = 9

# Publication-quality font sizes
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '-',
    'grid.linewidth': 0.5,
    'axes.linewidth': 0.8,
})


def load_metric_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure models are ordered
    df['Model'] = pd.Categorical(df['Model'], categories=MODEL_ORDER, ordered=True)
    return df.sort_values(['Model', 'Target', 'Fold'])


def plot_panel(ax, df, target, metric, title, ylabel):
    """Plot one panel of the 2×2 figure."""
    folds = np.arange(N_FOLDS)
    df_sub = df[(df['Target'] == target)]

    for model in MODEL_ORDER:
        df_model = df_sub[df_sub['Model'] == model].sort_values('Fold')
        vals = df_model[metric].values
        ax.plot(
            folds, vals,
            color=COLORS[model],
            marker=MARKERS[model],
            markersize=MARKERSIZE[model],
            linewidth=LINEWIDTH[model],
            alpha=ALPHA[model],
            label=model,
            zorder=3,
        )

    # Highlight fold 6 with a light grey vertical band
    ax.axvspan(FOLD_HIGHLIGHT - 0.4, FOLD_HIGHLIGHT + 0.4,
               color='grey', alpha=0.12, zorder=1)

    ax.set_xticks(folds)
    ax.set_xticklabels([str(f) for f in folds])
    ax.set_xlim(-0.5, N_FOLDS - 0.5)
    ax.set_xlabel('LOMO Fold')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(bottom=None, top=None)


def main():
    df = load_metric_table(CSV_PATH)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True)
    axes = axes.flatten()

    panels = [
        ('EA', 'R2', '(a) Overall EA Prediction'),
        ('IP', 'R2', '(b) Overall IP Prediction'),
        ('EA', 'ArchDev_R2', '(c) EA Architecture-Deviation Prediction'),
        ('IP', 'ArchDev_R2', '(d) IP Architecture-Deviation Prediction'),
    ]

    for ax, (target, metric, title) in zip(axes, panels):
        plot_panel(ax, df, target, metric, title, r'$R^2$')

    # Shared legend below the figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        ncol=len(MODEL_ORDER),
        frameon=True,
        fancybox=False,
        edgecolor='black',
        framealpha=0.95,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    fig.savefig(OUT_DIR / 'fusion_ablation_per_fold.png')
    fig.savefig(OUT_DIR / 'fusion_ablation_per_fold.pdf')
    print(f"Saved: {OUT_DIR / 'fusion_ablation_per_fold.png'}")
    print(f"Saved: {OUT_DIR / 'fusion_ablation_per_fold.pdf'}")


if __name__ == '__main__':
    main()
