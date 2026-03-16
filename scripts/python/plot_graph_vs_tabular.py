#!/usr/bin/env python
"""
Plot graph vs tabular model comparisons for multiple datasets.

Plot 1: For tc, polyinfo, htpmd, insulator — grouped bars with
        Tabular (green tones) vs Graph (blue tones) on x-axis.
Plot 2: For htpmd — graph-only models in specified order.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── colour palettes ──────────────────────────────────────────────
# Tabular: green family
TABULAR_COLORS = {
    'Linear':  '#1b7837',   # dark green
    'LogReg':  '#1b7837',   # dark green (classification equivalent)
    'RF':      '#74c476',   # medium green
    'XGB':     '#bae4b3',   # light green
}

# Graph: blue family  (dark → light keeps models distinguishable)
GRAPH_COLORS = {
    'DMPNN':          '#08306b',  # darkest blue
    'DMPNN_DiffPool': '#08519c',
    'AttentiveFP':    '#2171b5',
    'PPG':            '#4292c6',
    'wDMPNN':         '#6baed6',
    'GIN':            '#9ecae1',
    'GAT':            '#c6dbef',  # lightest blue
}

# ── helpers ──────────────────────────────────────────────────────
def normalise_columns(df):
    """Rename metric columns to a consistent scheme."""
    rename_map = {
        # Regression
        'test/mae':  'mae',
        'test/rmse': 'rmse',
        'test/r2':   'r2',
        'test/mse':  'mse',
        # Classification (chemprop graph models)
        'test/multiclass-accuracy': 'acc',
        'test/multiclass-f1':      'f1_macro',
        'test/multiclass-roc':     'roc_auc',
        # Classification (binary variants)
        'test/binary-accuracy': 'acc',
        'test/binary-f1':      'f1_macro',
        'test/binary-auroc':   'roc_auc',
    }
    df = df.rename(columns=rename_map)
    # Compute rmse from mse when rmse is missing
    if 'rmse' not in df.columns and 'mse' in df.columns:
        df['rmse'] = np.sqrt(df['mse'].astype(float))
    return df


def load_graph_results(results_dir, dataset, model_name):
    """Load graph-only results for one model, trying several naming patterns."""
    candidates = [
        results_dir / model_name / f'{dataset}_results.csv',
        results_dir / model_name / f'{dataset}__{model_name.lower()}_baselines.csv',
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            # baselines file stores many encoders; keep only this one
            if 'encoder' in df.columns:
                df = df[df['encoder'] == model_name].copy()
            df = normalise_columns(df)
            df['display_model'] = model_name
            df['category'] = 'Graph'
            print(f"  Loaded graph {model_name:18s} from {path.name}  ({len(df)} rows)")
            return df
    print(f"  MISSING graph {model_name:18s} for {dataset}")
    return None


def load_tabular_results(results_dir, dataset):
    """Load tabular results. Returns list of per-model DataFrames."""
    # Try _ab.csv first, then plain .csv
    for suffix in ['_ab.csv', '.csv']:
        path = results_dir / 'tabular' / f'{dataset}{suffix}'
        if path.exists():
            df = pd.read_csv(path)
            df = normalise_columns(df)
            frames = []
            for model_name in df['model'].unique():
                sub = df[df['model'] == model_name].copy()
                sub['display_model'] = model_name
                sub['category'] = 'Tabular'
                frames.append(sub)
                print(f"  Loaded tabular {model_name:18s} from {path.name}  ({len(sub)} rows)")
            return frames
    print(f"  MISSING tabular results for {dataset}")
    return []


# ── regression metrics to plot ───────────────────────────────────
REG_METRICS  = ['mae', 'r2', 'rmse']
CLF_METRICS  = ['acc', 'f1_macro', 'roc_auc']


def pick_metrics(df):
    """Return the list of metrics available in df."""
    for suite in [REG_METRICS, CLF_METRICS]:
        present = [m for m in suite if m in df.columns]
        if present:
            return present
    return []


# ── Plot 1 ───────────────────────────────────────────────────────
def plot_dataset_comparison(results_dir, dataset, output_dir):
    """Grouped‐bar plot: Tabular (green) vs Graph (blue) for one dataset."""

    tabular_model_order = ['Linear', 'LogReg', 'RF', 'XGB']
    graph_model_order   = ['DMPNN', 'DMPNN_DiffPool', 'AttentiveFP', 'PPG', 'wDMPNN']

    frames = load_tabular_results(results_dir, dataset)
    for model in graph_model_order:
        gdf = load_graph_results(results_dir, dataset, model)
        if gdf is not None:
            frames.append(gdf)

    if not frames:
        print(f"  ⇒ Nothing to plot for {dataset}\n")
        return

    combined = pd.concat(frames, ignore_index=True)
    targets  = sorted(combined['target'].unique())
    metrics  = pick_metrics(combined)
    if not metrics:
        print(f"  ⇒ No plottable metrics for {dataset}\n")
        return

    for metric in metrics:
        n_targets = len(targets)
        fig, axes = plt.subplots(1, n_targets,
                                 figsize=(max(7, 4*n_targets), 6),
                                 squeeze=False)
        axes = axes[0]

        first_ax = True          # only first subplot adds legend labels
        legend_handles, legend_labels = [], []

        for ax, target in zip(axes, targets):
            tdata = combined[combined['target'] == target]
            summary = (tdata.groupby(['category', 'display_model'])[metric]
                       .agg(['mean', 'std']).reset_index())

            # Order tabular models
            tab_rows = summary[summary['category'] == 'Tabular'].copy()
            tab_rows['_ord'] = tab_rows['display_model'].map(
                {m: i for i, m in enumerate(tabular_model_order)})
            tab_rows = tab_rows.sort_values('_ord').reset_index(drop=True)

            # Order graph models
            gph_rows = summary[summary['category'] == 'Graph'].copy()
            gph_rows['_ord'] = gph_rows['display_model'].map(
                {m: i for i, m in enumerate(graph_model_order)})
            gph_rows = gph_rows.sort_values('_ord').reset_index(drop=True)

            n_tab = len(tab_rows)
            n_gph = len(gph_rows)
            bar_w = 0.13

            # centre each group
            tab_pos = np.arange(n_tab)*bar_w - (n_tab-1)*bar_w/2
            gph_pos = 1.0 + np.arange(n_gph)*bar_w - (n_gph-1)*bar_w/2

            for i, (_, row) in enumerate(tab_rows.iterrows()):
                c = TABULAR_COLORS.get(row['display_model'], '#2d5f2e')
                bar = ax.bar(tab_pos[i], row['mean'], yerr=row['std'],
                             width=bar_w, color=c, capsize=3,
                             edgecolor='black', linewidth=0.5)
                if first_ax:
                    legend_handles.append(bar[0])
                    legend_labels.append(row['display_model'])

            for i, (_, row) in enumerate(gph_rows.iterrows()):
                c = GRAPH_COLORS.get(row['display_model'], '#08519c')
                bar = ax.bar(gph_pos[i], row['mean'], yerr=row['std'],
                             width=bar_w, color=c, capsize=3,
                             edgecolor='black', linewidth=0.5)
                if first_ax:
                    legend_handles.append(bar[0])
                    legend_labels.append(row['display_model'])

            first_ax = False
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Tabular', 'Graph'], fontsize=12, fontweight='bold')
            ax.set_ylabel(metric.upper(), fontsize=11)
            ax.set_title(target, fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            pad = max(n_tab, n_gph) * bar_w / 2 + 0.3
            ax.set_xlim(-pad, 1 + pad)

        fig.legend(legend_handles, legend_labels,
                   loc='lower center', bbox_to_anchor=(0.5, -0.12),
                   ncol=min(len(legend_labels), 8), fontsize=9,
                   frameon=True)
        fig.suptitle(f'{dataset.upper()} — {metric.upper()}',
                     y=0.98, fontsize=14, fontweight='bold')
        fig.tight_layout()

        out = output_dir / f'{dataset}_{metric}_graph_vs_tabular.png'
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f"  Saved: {out}")
        plt.close(fig)


# ── Plot 2 ───────────────────────────────────────────────────────
def plot_htpmd_graph_only(results_dir, output_dir):
    """Bar plot of htpmd graph models — all that have results, auto-discovered."""

    # Preferred display order (models found but not here are appended alphabetically)
    preferred_order = ['GIN', 'GAT', 'DMPNN', 'AttentiveFP',
                       'DMPNN_DiffPool', 'wDMPNN', 'PPG']
    non_graph = {'tabular'}  # subdirs to skip

    # Auto-discover models that have htpmd_results.csv
    discovered = sorted(
        d.name for d in results_dir.iterdir()
        if d.is_dir() and d.name.lower() not in non_graph
        and (d / 'htpmd_results.csv').exists()
    )
    # Order: preferred first (if present), then any extras alphabetically
    model_order = [m for m in preferred_order if m in discovered] + \
                  [m for m in discovered if m not in preferred_order]

    frames = []
    for model in model_order:
        gdf = load_graph_results(results_dir, 'htpmd', model)
        if gdf is not None:
            frames.append(gdf)

    if not frames:
        print("  ⇒ Nothing to plot for htpmd graph-only")
        return

    combined = pd.concat(frames, ignore_index=True)
    targets  = sorted(combined['target'].unique())
    metrics  = pick_metrics(combined)

    for metric in metrics:
        n_targets = len(targets)
        fig, axes = plt.subplots(1, n_targets,
                                 figsize=(max(7, 4*n_targets), 6),
                                 squeeze=False)
        axes = axes[0]

        legend_handles, legend_labels = [], []
        first_ax = True

        for ax, target in zip(axes, targets):
            tdata = combined[combined['target'] == target]
            summary = (tdata.groupby('display_model')[metric]
                       .agg(['mean', 'std']).reset_index())
            summary['_ord'] = summary['display_model'].map(
                {m: i for i, m in enumerate(model_order)})
            summary = summary.sort_values('_ord').reset_index(drop=True)

            x = np.arange(len(summary))
            bar_w = 0.7

            for i, (_, row) in enumerate(summary.iterrows()):
                c = GRAPH_COLORS.get(row['display_model'], '#08519c')
                bar = ax.bar(x[i], row['mean'], yerr=row['std'],
                             width=bar_w, color=c, capsize=3,
                             edgecolor='black', linewidth=0.5)
                if first_ax:
                    legend_handles.append(bar[0])
                    legend_labels.append(row['display_model'])

            first_ax = False
            ax.set_xticks(x)
            ax.set_xticklabels(summary['display_model'],
                               rotation=45, ha='right', fontsize=9)
            ax.set_ylabel(metric.upper(), fontsize=11)
            ax.set_title(target, fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')

        fig.legend(legend_handles, legend_labels,
                   loc='lower center', bbox_to_anchor=(0.5, -0.12),
                   ncol=min(len(legend_labels), 7), fontsize=9,
                   frameon=True)
        fig.suptitle(f'HTPMD — {metric.upper()} (Graph Models)',
                     y=0.98, fontsize=14, fontweight='bold')
        fig.tight_layout()

        out = output_dir / f'htpmd_{metric}_graph_only.png'
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f"  Saved: {out}")
        plt.close(fig)


# ── main ─────────────────────────────────────────────────────────
def main():
    results_dir = Path('/Users/u6788552/Desktop/experiments/dmpnn/results')
    output_dir  = Path('/Users/u6788552/Desktop/experiments/dmpnn/plots/graph_vs_tabular')
    output_dir.mkdir(parents=True, exist_ok=True)

    for ds in ['tc', 'polyinfo', 'htpmd', 'insulator']:
        print(f"\n{'='*50}\nDataset: {ds}\n{'='*50}")
        plot_dataset_comparison(results_dir, ds, output_dir)

    print(f"\n{'='*50}\nHTMPD graph-only\n{'='*50}")
    plot_htpmd_graph_only(results_dir, output_dir)

    print(f"\nAll plots → {output_dir}")


if __name__ == '__main__':
    main()
