"""Publication-quality plots comparing copolymer representation methods.

Compares mixture (baseline), frac_attn, mix_pair, mix_pair_attn, and
self_attn across DMPNN, GIN, GAT models on EA/IP targets under
a_held_out split.

Generates:
  1. Absolute RMSE by representation method
  2. ΔRMSE vs mixture   (negative = better)
  3. ΔR² vs mixture     (positive = better)
  4. Scatter: RMSE_method vs RMSE_mixture  (diagnostic)
  5. Scatter: self_attn vs mix_pair_attn   (head-to-head)
  6. Heatmap: mean ΔRMSE vs mixture
  7. Win-rate: fraction of folds improved over mixture
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_DIR = Path(__file__).resolve().parent / "repr_report"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["DMPNN", "GIN", "GAT"]
TARGETS = {
    "EA vs SHE (eV)": "EA",
    "IP vs SHE (eV)": "IP",
}

# Method key → (copolymer_mode substring in filename, display label)
METHOD_MAP = {
    "mixture":       ("mix_meta",           "mixture"),
    "frac_attn":     ("frac_attn_meta",     "frac_attn"),
    "mix_pair":      ("mix_pair_meta",      "mix_pair"),
    "mix_pair_attn": ("mix_pair_attn_meta", "mix_pair_attn"),
    "self_attn":     ("self_attn_meta",     "self_attn"),
}

BASELINE = "mixture"
METHOD_ORDER = ["mixture", "frac_attn", "mix_pair", "mix_pair_attn", "self_attn"]
DELTA_ORDER = ["frac_attn", "mix_pair", "mix_pair_attn", "self_attn"]
SPLIT = "a_held_out"

MODEL_PALETTE = {"DMPNN": "#1b9e77", "GIN": "#d95f02", "GAT": "#7570b3"}

# Seaborn / matplotlib defaults
sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 12,
})


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────

def _build_filename(model: str, method_key: str, target: str) -> Path:
    """Build result CSV path for a given model × method × target."""
    mode_substr = METHOD_MAP[method_key][0]

    # mix_pair_attn_meta contains "mix_pair_meta" as substring, so we
    # need exact matching to avoid ambiguity.  The filename pattern is:
    #   ea_ip__copoly_{mode}__poly_type__a_held_out__target_{target}_results.csv
    fname = (
        f"ea_ip__copoly_{mode_substr}__poly_type__{SPLIT}"
        f"__target_{target}_results.csv"
    )
    return RESULTS_DIR / model / fname


def load_all_results() -> pd.DataFrame:
    """Load and concatenate all result CSVs into a tidy DataFrame."""
    rows = []
    missing = []
    for model in MODELS:
        for method_key, (_, label) in METHOD_MAP.items():
            for target_full, target_short in TARGETS.items():
                fpath = _build_filename(model, method_key, target_full)
                if not fpath.exists():
                    missing.append(str(fpath))
                    continue
                df = pd.read_csv(fpath)
                df = df.rename(columns={
                    "test/rmse": "rmse",
                    "test/r2": "r2",
                    "test/mae": "mae",
                    "split": "fold",
                })
                df["model"] = model
                df["target"] = target_short
                df["method"] = label
                rows.append(df[["model", "target", "method", "fold", "rmse", "r2", "mae"]])

    if missing:
        print(f"WARNING: {len(missing)} result files not found:")
        for m in missing:
            print(f"  {m}")

    data = pd.concat(rows, ignore_index=True)
    data["method"] = pd.Categorical(data["method"], categories=METHOD_ORDER, ordered=True)
    print(f"Loaded {len(data)} rows  "
          f"({data['model'].nunique()} models × {data['target'].nunique()} targets × "
          f"{data['method'].nunique()} methods × {data['fold'].nunique()} folds)")
    return data


def compute_deltas(data: pd.DataFrame, baseline: str = BASELINE) -> pd.DataFrame:
    """Compute per-fold delta metrics vs baseline method.

    Returns DataFrame with columns:
      model, target, method, fold, delta_rmse, delta_r2
    """
    base = data[data["method"] == baseline][["model", "target", "fold", "rmse", "r2"]]
    base = base.rename(columns={"rmse": "rmse_base", "r2": "r2_base"})

    others = data[data["method"] != baseline].copy()
    merged = others.merge(base, on=["model", "target", "fold"], how="inner")
    merged["delta_rmse"] = merged["rmse"] - merged["rmse_base"]
    merged["delta_r2"] = merged["r2"] - merged["r2_base"]
    merged["method"] = pd.Categorical(merged["method"], categories=DELTA_ORDER, ordered=True)
    return merged


# ──────────────────────────────────────────────────────────────────────
# Task 1: Absolute RMSE
# ──────────────────────────────────────────────────────────────────────

def plot_absolute_rmse(data: pd.DataFrame) -> None:
    """Boxplot + strip of absolute RMSE by method, hue=model, col=target."""
    g = sns.catplot(
        data=data, x="method", y="rmse", hue="model", col="target",
        kind="box", palette=MODEL_PALETTE, height=4.5, aspect=1.3,
        order=METHOD_ORDER, hue_order=MODELS,
        boxprops=dict(alpha=0.4), fliersize=0, width=0.65,
    )
    for ax in g.axes.flat:
        sns.stripplot(
            data=data[data["target"] == ax.get_title().split(" = ")[1]],
            x="method", y="rmse", hue="model", order=METHOD_ORDER,
            hue_order=MODELS, palette=MODEL_PALETTE, dodge=True,
            size=4, alpha=0.7, ax=ax, legend=False,
        )
    g.set_titles("{col_var} = {col_name}")
    g.set_axis_labels("Representation", "RMSE (eV)")
    g.figure.suptitle("RMSE by representation method", y=1.02, fontsize=14)
    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=25)
    g.tight_layout()
    out = OUTPUT_DIR / "repr_absolute_rmse.png"
    g.savefig(out)
    plt.close(g.figure)
    print(f"Saved {out}")


# ──────────────────────────────────────────────────────────────────────
# Task 2: ΔRMSE vs mixture
# ──────────────────────────────────────────────────────────────────────

def plot_delta_rmse(deltas: pd.DataFrame) -> None:
    """Boxplot + strip of ΔRMSE vs mixture, with conceptual section labels."""
    g = sns.catplot(
        data=deltas, x="method", y="delta_rmse", hue="model", col="target",
        kind="box", palette=MODEL_PALETTE, height=4.5, aspect=1.2,
        order=DELTA_ORDER, hue_order=MODELS,
        boxprops=dict(alpha=0.4), fliersize=0, width=0.6,
    )
    for ax in g.axes.flat:
        target_name = ax.get_title().split(" = ")[1]
        sns.stripplot(
            data=deltas[deltas["target"] == target_name],
            x="method", y="delta_rmse", hue="model", order=DELTA_ORDER,
            hue_order=MODELS, palette=MODEL_PALETTE, dodge=True,
            size=4, alpha=0.7, ax=ax, legend=False,
        )
        ax.axhline(0, ls="--", lw=1, color="grey", zorder=0)

        # ── Section labels above x-axis ────────────────────────────────
        # DELTA_ORDER = ["frac_attn", "mix_pair", "mix_pair_attn", "self_attn"]
        # Tick positions are 0, 1, 2, 3 (catplot integer x-axis)
        # Sections: frac_attn(0), mix_pair(1), mix_pair_attn(2), self_attn(3)
        SECTION_LABELS = [
            (0, 0,   "Better\nMixing",        "#AED6F1"),   # single tick
            (1, 1,   "Naive\nInteraction",     "#A9DFBF"),
            (2, 2,   "Learned\nInteraction",   "#FAD7A0"),
            (3, 3,   "Full\nAttention",        "#F1948A"),
        ]
        ylim = ax.get_ylim()
        yrange = ylim[1] - ylim[0]
        label_y = ylim[1] + yrange * 0.02   # just above top of plot

        for x_lo, x_hi, sect_label, sect_color in SECTION_LABELS:
            cx = (x_lo + x_hi) / 2
            # light background span
            ax.axvspan(x_lo - 0.45, x_hi + 0.45,
                       ymin=0, ymax=1,
                       color=sect_color, alpha=0.12, zorder=0, lw=0)
            # label text above the plot area (clip_on=False so it's visible)
            ax.text(cx, label_y, sect_label,
                    ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold", color="#2C3E50",
                    clip_on=False)

        # Expand ylim slightly so section labels don't overlap data
        ax.set_ylim(ylim[0], ylim[1] + yrange * 0.18)

    g.set_titles("{col_var} = {col_name}")
    g.set_axis_labels("Representation", "ΔRMSE vs mixture (eV)")
    g.figure.suptitle("ΔRMSE vs mixture (negative = better)", y=1.02, fontsize=14)
    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=25)
    g.tight_layout()
    out = OUTPUT_DIR / "repr_delta_rmse.png"
    g.savefig(out)
    plt.close(g.figure)
    print(f"Saved {out}")


# ──────────────────────────────────────────────────────────────────────
# Task 3: ΔR² vs mixture
# ──────────────────────────────────────────────────────────────────────

def plot_delta_r2(deltas: pd.DataFrame) -> None:
    """Boxplot + strip of ΔR² vs mixture."""
    g = sns.catplot(
        data=deltas, x="method", y="delta_r2", hue="model", col="target",
        kind="box", palette=MODEL_PALETTE, height=4.5, aspect=1.2,
        order=DELTA_ORDER, hue_order=MODELS,
        boxprops=dict(alpha=0.4), fliersize=0, width=0.6,
    )
    for ax in g.axes.flat:
        target_name = ax.get_title().split(" = ")[1]
        sns.stripplot(
            data=deltas[deltas["target"] == target_name],
            x="method", y="delta_r2", hue="model", order=DELTA_ORDER,
            hue_order=MODELS, palette=MODEL_PALETTE, dodge=True,
            size=4, alpha=0.7, ax=ax, legend=False,
        )
        ax.axhline(0, ls="--", lw=1, color="grey", zorder=0)
    g.set_titles("{col_var} = {col_name}")
    g.set_axis_labels("Representation", "ΔR² vs mixture")
    g.figure.suptitle("ΔR² vs mixture (positive = better)", y=1.02, fontsize=14)
    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=25)
    g.tight_layout()
    out = OUTPUT_DIR / "repr_delta_r2.png"
    g.savefig(out)
    plt.close(g.figure)
    print(f"Saved {out}")


# ──────────────────────────────────────────────────────────────────────
# Task 4: Scatter RMSE_method vs RMSE_mixture
# ──────────────────────────────────────────────────────────────────────

def plot_scatter_rmse(data: pd.DataFrame) -> None:
    """Scatter of RMSE_method vs RMSE_mixture, rows=method, cols=target."""
    base = data[data["method"] == BASELINE][["model", "target", "fold", "rmse"]]
    base = base.rename(columns={"rmse": "rmse_mixture"})
    others = data[data["method"] != BASELINE].copy()
    merged = others.merge(base, on=["model", "target", "fold"], how="inner")
    merged["method"] = pd.Categorical(merged["method"], categories=DELTA_ORDER, ordered=True)

    g = sns.FacetGrid(
        merged, row="method", col="target", hue="model",
        palette=MODEL_PALETTE, hue_order=MODELS,
        height=3.2, aspect=1.1, margin_titles=True,
    )
    g.map_dataframe(sns.scatterplot, x="rmse_mixture", y="rmse", s=50, alpha=0.8)

    # Add diagonal y=x reference
    for ax in g.axes.flat:
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, ls="--", lw=1, color="grey", zorder=0)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")

    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    g.set_axis_labels("RMSE (mixture)", "RMSE (method)")
    g.add_legend(title="Model")
    g.figure.suptitle("RMSE: method vs mixture", y=1.02, fontsize=14)
    g.tight_layout()
    out = OUTPUT_DIR / "repr_scatter_rmse.png"
    g.savefig(out)
    plt.close(g.figure)
    print(f"Saved {out}")


# ──────────────────────────────────────────────────────────────────────
# Task 5: Scatter self_attn vs mix_pair_attn
# ──────────────────────────────────────────────────────────────────────

def plot_self_attn_vs_mix_pair_attn(data: pd.DataFrame) -> None:
    """Head-to-head scatter: self_attn vs mix_pair_attn RMSE."""
    sa = data[data["method"] == "self_attn"][["model", "target", "fold", "rmse"]]
    sa = sa.rename(columns={"rmse": "rmse_self_attn"})
    mpa = data[data["method"] == "mix_pair_attn"][["model", "target", "fold", "rmse"]]
    mpa = mpa.rename(columns={"rmse": "rmse_mix_pair_attn"})
    merged = sa.merge(mpa, on=["model", "target", "fold"], how="inner")

    g = sns.FacetGrid(
        merged, col="target", hue="model",
        palette=MODEL_PALETTE, hue_order=MODELS,
        height=4, aspect=1.1,
    )
    g.map_dataframe(
        sns.scatterplot,
        x="rmse_mix_pair_attn", y="rmse_self_attn", s=60, alpha=0.8,
    )

    for ax in g.axes.flat:
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, ls="--", lw=1, color="grey", zorder=0)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")

    g.set_titles("{col_name}")
    g.set_axis_labels("RMSE (mix_pair_attn)", "RMSE (self_attn)")
    g.add_legend(title="Model")
    g.figure.suptitle("self_attn vs mix_pair_attn", y=1.02, fontsize=14)
    g.tight_layout()
    out = OUTPUT_DIR / "self_attn_vs_mix_pair_attn.png"
    g.savefig(out)
    plt.close(g.figure)
    print(f"Saved {out}")


# ──────────────────────────────────────────────────────────────────────
# Task 6: Heatmap of mean ΔRMSE vs mixture
# ──────────────────────────────────────────────────────────────────────

def plot_heatmap_delta_rmse(deltas: pd.DataFrame) -> None:
    """One annotated heatmap per target: rows=model, cols=method, values=mean ΔRMSE."""
    for target_short in TARGETS.values():
        sub = deltas[deltas["target"] == target_short]
        pivot = sub.pivot_table(
            index="model", columns="method", values="delta_rmse",
            aggfunc="mean", observed=False,
        )
        # Ensure ordering
        pivot = pivot.reindex(index=MODELS, columns=DELTA_ORDER)

        fig, ax = plt.subplots(figsize=(5.5, 3))
        vmax = max(abs(pivot.values.min()), abs(pivot.values.max())) * 1.1
        sns.heatmap(
            pivot, annot=True, fmt=".4f", center=0,
            cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            linewidths=0.5, ax=ax,
            cbar_kws={"label": "mean ΔRMSE (eV)"},
        )
        ax.set_title(f"{target_short}: mean ΔRMSE vs mixture", fontsize=13)
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        out = OUTPUT_DIR / f"repr_delta_rmse_heatmap_{target_short}.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")


# ──────────────────────────────────────────────────────────────────────
# Task 7: Win-rate plot
# ──────────────────────────────────────────────────────────────────────

def plot_win_rate(deltas: pd.DataFrame) -> None:
    """Bar chart: % of folds where method beats mixture (lower RMSE)."""
    wins = (
        deltas.groupby(["model", "target", "method"], observed=False)
        .apply(lambda g: (g["delta_rmse"] < 0).mean() * 100, include_groups=False)
        .reset_index(name="win_pct")
    )
    wins["method"] = pd.Categorical(wins["method"], categories=DELTA_ORDER, ordered=True)

    g = sns.catplot(
        data=wins, x="method", y="win_pct", hue="model", col="target",
        kind="bar", palette=MODEL_PALETTE, height=4.5, aspect=1.1,
        order=DELTA_ORDER, hue_order=MODELS, alpha=0.85,
    )
    for ax in g.axes.flat:
        ax.axhline(50, ls="--", lw=1, color="grey", zorder=0)
        ax.set_ylim(0, 105)
    g.set_titles("{col_var} = {col_name}")
    g.set_axis_labels("Representation", "Win rate vs mixture (%)")
    g.figure.suptitle("Win rate: % folds with lower RMSE than mixture", y=1.02, fontsize=14)
    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=25)
    g.tight_layout()
    out = OUTPUT_DIR / "repr_win_rate.png"
    g.savefig(out)
    plt.close(g.figure)
    print(f"Saved {out}")


# ──────────────────────────────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────────────────────────────

def print_summary(data: pd.DataFrame, deltas: pd.DataFrame) -> None:
    """Print summary statistics to console."""
    print("\n" + "=" * 70)
    print("SUMMARY: Mean RMSE ± std across folds")
    print("=" * 70)
    summary = (
        data.groupby(["target", "model", "method"], observed=False)["rmse"]
        .agg(["mean", "std"])
        .round(5)
    )
    print(summary.to_string())

    print("\n" + "=" * 70)
    print("SUMMARY: Mean ΔRMSE vs mixture (negative = better)")
    print("=" * 70)
    delta_summary = (
        deltas.groupby(["target", "model", "method"], observed=False)["delta_rmse"]
        .agg(["mean", "std"])
        .round(5)
    )
    print(delta_summary.to_string())

    print("\n" + "=" * 70)
    print("SUMMARY: Mean ΔR² vs mixture (positive = better)")
    print("=" * 70)
    r2_summary = (
        deltas.groupby(["target", "model", "method"], observed=False)["delta_r2"]
        .agg(["mean", "std"])
        .round(5)
    )
    print(r2_summary.to_string())


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    data = load_all_results()
    deltas = compute_deltas(data)

    plot_absolute_rmse(data)
    plot_delta_rmse(deltas)
    plot_delta_r2(deltas)
    plot_scatter_rmse(data)
    plot_self_attn_vs_mix_pair_attn(data)
    plot_heatmap_delta_rmse(deltas)
    plot_win_rate(deltas)

    print_summary(data, deltas)
    print(f"\nAll plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
