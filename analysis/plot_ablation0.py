"""Ablation 0 – baseline comparison before introducing HPG components.

Compares HPG_baseline (HPG + incl_desc + poly_type) against the best
DMPNN/GIN/GAT copolymer representations: mixture, mix_pair_attn, self_attn.

HPG is its own model architecture, so HPG_baseline results are
model-agnostic.  For delta computations the HPG fold RMSE/R² is used
as the shared reference for *every* GNN model on the same (target, fold).

Generates:
  1. Absolute RMSE boxplot
  2. ΔRMSE vs HPG_baseline   (negative = better)
  3. ΔR² vs HPG_baseline     (positive = better)
  4. Scatter: RMSE_method vs RMSE_HPG_baseline  (diagnostic)
  5. Win-rate: % folds with lower RMSE than HPG_baseline
  6. Heatmap: mean ΔRMSE vs HPG_baseline
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
OUTPUT_DIR = Path(__file__).resolve().parent / "ablation0_report"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GNN_MODELS = ["DMPNN", "GIN", "GAT"]
TARGETS = {
    "EA vs SHE (eV)": "EA",
    "IP vs SHE (eV)": "IP",
}
SPLIT = "a_held_out"

BASELINE = "HPG_baseline"
METHOD_ORDER = ["HPG_baseline", "mixture", "mix_pair_attn", "self_attn"]
DELTA_ORDER = ["mixture", "mix_pair_attn", "self_attn"]

# GNN method key → (filename copolymer_mode substring, display label)
GNN_METHOD_MAP = {
    "mixture":       "mix_meta",
    "mix_pair_attn": "mix_pair_attn_meta",
    "self_attn":     "self_attn_meta",
}

# Colours: HPG gets its own colour; GNN models keep the existing scheme.
MODEL_PALETTE = {
    "HPG":   "#e7298a",
    "DMPNN": "#1b9e77",
    "GIN":   "#d95f02",
    "GAT":   "#7570b3",
}

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

def _gnn_filename(model: str, mode_substr: str, target_full: str) -> Path:
    """Build per-target result CSV path for a GNN model × copolymer mode."""
    fname = (
        f"ea_ip__copoly_{mode_substr}__poly_type__{SPLIT}"
        f"__target_{target_full}_results.csv"
    )
    return RESULTS_DIR / model / fname


def load_all_results() -> pd.DataFrame:
    """Load HPG_baseline + GNN method results into a tidy DataFrame."""
    rows: list[pd.DataFrame] = []
    missing: list[str] = []

    # --- HPG_baseline (single multi-target file) ---
    hpg_path = RESULTS_DIR / "HPG" / "ea_ip__desc__poly_type__a_held_out_results.csv"
    if hpg_path.exists():
        hpg = pd.read_csv(hpg_path).rename(columns={
            "test/rmse": "rmse", "test/r2": "r2", "test/mae": "mae", "split": "fold",
        })
        hpg["target"] = hpg["target"].map(TARGETS)
        hpg["model"] = "HPG"
        hpg["method"] = "HPG_baseline"
        rows.append(hpg[["model", "target", "method", "fold", "rmse", "r2", "mae"]])
    else:
        missing.append(str(hpg_path))

    # --- GNN models × copolymer methods ---
    for model in GNN_MODELS:
        for method_label, mode_substr in GNN_METHOD_MAP.items():
            for target_full, target_short in TARGETS.items():
                fpath = _gnn_filename(model, mode_substr, target_full)
                if not fpath.exists():
                    missing.append(str(fpath))
                    continue
                df = pd.read_csv(fpath).rename(columns={
                    "test/rmse": "rmse", "test/r2": "r2", "test/mae": "mae",
                    "split": "fold",
                })
                df["model"] = model
                df["target"] = target_short
                df["method"] = method_label
                rows.append(df[["model", "target", "method", "fold", "rmse", "r2", "mae"]])

    if missing:
        print(f"WARNING: {len(missing)} result file(s) not found:")
        for m in missing:
            print(f"  {m}")

    data = pd.concat(rows, ignore_index=True)
    data["method"] = pd.Categorical(data["method"], categories=METHOD_ORDER, ordered=True)
    print(f"Loaded {len(data)} rows  "
          f"(models: {sorted(data['model'].unique())}, "
          f"targets: {sorted(data['target'].unique())}, "
          f"methods: {sorted(data['method'].unique())}, "
          f"folds: {sorted(data['fold'].unique())})")
    return data


def compute_deltas(data: pd.DataFrame) -> pd.DataFrame:
    """Compute per-fold delta metrics vs HPG_baseline.

    HPG_baseline is model-agnostic so we merge on (target, fold) only.

    Returns DataFrame with columns:
      model, target, method, fold, rmse, r2, delta_rmse, delta_r2
    """
    base = data[data["method"] == BASELINE][["target", "fold", "rmse", "r2"]]
    base = base.rename(columns={"rmse": "rmse_base", "r2": "r2_base"})

    others = data[data["method"] != BASELINE].copy()
    merged = others.merge(base, on=["target", "fold"], how="inner")
    merged["delta_rmse"] = merged["rmse"] - merged["rmse_base"]
    merged["delta_r2"] = merged["r2"] - merged["r2_base"]
    merged["method"] = pd.Categorical(merged["method"], categories=DELTA_ORDER, ordered=True)
    return merged


# ──────────────────────────────────────────────────────────────────────
# Task 1: Absolute RMSE
# ──────────────────────────────────────────────────────────────────────

def plot_absolute_rmse(data: pd.DataFrame) -> None:
    """Boxplot + strip of absolute RMSE by method, hue=model, col=target."""
    all_models = ["HPG"] + GNN_MODELS
    g = sns.catplot(
        data=data, x="method", y="rmse", hue="model", col="target",
        kind="box", palette=MODEL_PALETTE, height=4.5, aspect=1.3,
        order=METHOD_ORDER, hue_order=all_models,
        boxprops=dict(alpha=0.4), fliersize=0, width=0.65,
    )
    for ax in g.axes.flat:
        target_name = ax.get_title().split(" = ")[1]
        sns.stripplot(
            data=data[data["target"] == target_name],
            x="method", y="rmse", hue="model", order=METHOD_ORDER,
            hue_order=all_models, palette=MODEL_PALETTE, dodge=True,
            size=4, alpha=0.7, ax=ax, legend=False,
        )
    g.set_titles("{col_var} = {col_name}")
    g.set_axis_labels("Representation", "RMSE (eV)")
    g.figure.suptitle("RMSE by baseline representation", y=1.02, fontsize=14)
    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=25)
    g.tight_layout()
    out = OUTPUT_DIR / "ablation0_absolute_rmse.png"
    g.savefig(out)
    plt.close(g.figure)
    print(f"Saved {out}")


# ──────────────────────────────────────────────────────────────────────
# Task 2: ΔRMSE vs HPG_baseline
# ──────────────────────────────────────────────────────────────────────

def plot_delta_rmse(deltas: pd.DataFrame) -> None:
    """Boxplot + strip of ΔRMSE vs HPG_baseline."""
    g = sns.catplot(
        data=deltas, x="method", y="delta_rmse", hue="model", col="target",
        kind="box", palette=MODEL_PALETTE, height=4.5, aspect=1.2,
        order=DELTA_ORDER, hue_order=GNN_MODELS,
        boxprops=dict(alpha=0.4), fliersize=0, width=0.6,
    )
    for ax in g.axes.flat:
        target_name = ax.get_title().split(" = ")[1]
        sns.stripplot(
            data=deltas[deltas["target"] == target_name],
            x="method", y="delta_rmse", hue="model", order=DELTA_ORDER,
            hue_order=GNN_MODELS, palette=MODEL_PALETTE, dodge=True,
            size=4, alpha=0.7, ax=ax, legend=False,
        )
        ax.axhline(0, ls="--", lw=1, color="grey", zorder=0)
    g.set_titles("{col_var} = {col_name}")
    g.set_axis_labels("Representation", "ΔRMSE vs HPG_baseline (eV)")
    g.figure.suptitle("ΔRMSE vs HPG_baseline (negative = better)", y=1.02, fontsize=14)
    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=25)
    g.tight_layout()
    out = OUTPUT_DIR / "ablation0_delta_rmse.png"
    g.savefig(out)
    plt.close(g.figure)
    print(f"Saved {out}")


# ──────────────────────────────────────────────────────────────────────
# Task 3: ΔR² vs HPG_baseline
# ──────────────────────────────────────────────────────────────────────

def plot_delta_r2(deltas: pd.DataFrame) -> None:
    """Boxplot + strip of ΔR² vs HPG_baseline."""
    g = sns.catplot(
        data=deltas, x="method", y="delta_r2", hue="model", col="target",
        kind="box", palette=MODEL_PALETTE, height=4.5, aspect=1.2,
        order=DELTA_ORDER, hue_order=GNN_MODELS,
        boxprops=dict(alpha=0.4), fliersize=0, width=0.6,
    )
    for ax in g.axes.flat:
        target_name = ax.get_title().split(" = ")[1]
        sns.stripplot(
            data=deltas[deltas["target"] == target_name],
            x="method", y="delta_r2", hue="model", order=DELTA_ORDER,
            hue_order=GNN_MODELS, palette=MODEL_PALETTE, dodge=True,
            size=4, alpha=0.7, ax=ax, legend=False,
        )
        ax.axhline(0, ls="--", lw=1, color="grey", zorder=0)
    g.set_titles("{col_var} = {col_name}")
    g.set_axis_labels("Representation", "ΔR² vs HPG_baseline")
    g.figure.suptitle("ΔR² vs HPG_baseline (positive = better)", y=1.02, fontsize=14)
    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=25)
    g.tight_layout()
    out = OUTPUT_DIR / "ablation0_delta_r2.png"
    g.savefig(out)
    plt.close(g.figure)
    print(f"Saved {out}")


# ──────────────────────────────────────────────────────────────────────
# Task 4: Scatter RMSE_method vs RMSE_HPG_baseline
# ──────────────────────────────────────────────────────────────────────

def plot_scatter_rmse(deltas: pd.DataFrame) -> None:
    """Scatter of RMSE_method vs RMSE_HPG_baseline, rows=method, cols=target."""
    plot_data = deltas.copy()
    plot_data["method"] = pd.Categorical(
        plot_data["method"], categories=DELTA_ORDER, ordered=True,
    )

    g = sns.FacetGrid(
        plot_data, row="method", col="target", hue="model",
        palette=MODEL_PALETTE, hue_order=GNN_MODELS,
        height=3.2, aspect=1.1, margin_titles=True,
    )
    g.map_dataframe(sns.scatterplot, x="rmse_base", y="rmse", s=50, alpha=0.8)

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
    g.set_axis_labels("RMSE (HPG_baseline)", "RMSE (method)")
    g.add_legend(title="Model")
    g.figure.suptitle("RMSE: method vs HPG_baseline", y=1.02, fontsize=14)
    g.tight_layout()
    out = OUTPUT_DIR / "ablation0_scatter_rmse.png"
    g.savefig(out)
    plt.close(g.figure)
    print(f"Saved {out}")


# ──────────────────────────────────────────────────────────────────────
# Task 5: Win-rate plot
# ──────────────────────────────────────────────────────────────────────

def plot_win_rate(deltas: pd.DataFrame) -> None:
    """Bar chart: % of folds where method beats HPG_baseline (lower RMSE)."""
    wins = (
        deltas.groupby(["model", "target", "method"], observed=False)
        .apply(lambda g: (g["delta_rmse"] < 0).mean() * 100, include_groups=False)
        .reset_index(name="win_pct")
    )
    wins["method"] = pd.Categorical(wins["method"], categories=DELTA_ORDER, ordered=True)

    g = sns.catplot(
        data=wins, x="method", y="win_pct", hue="model", col="target",
        kind="bar", palette=MODEL_PALETTE, height=4.5, aspect=1.1,
        order=DELTA_ORDER, hue_order=GNN_MODELS, alpha=0.85,
    )
    for ax in g.axes.flat:
        ax.axhline(50, ls="--", lw=1, color="grey", zorder=0)
        ax.set_ylim(0, 105)
    g.set_titles("{col_var} = {col_name}")
    g.set_axis_labels("Representation", "Win rate vs HPG_baseline (%)")
    g.figure.suptitle(
        "Win rate: % folds with lower RMSE than HPG_baseline",
        y=1.02, fontsize=14,
    )
    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=25)
    g.tight_layout()
    out = OUTPUT_DIR / "ablation0_win_rate.png"
    g.savefig(out)
    plt.close(g.figure)
    print(f"Saved {out}")


# ──────────────────────────────────────────────────────────────────────
# Task 6: Heatmap of mean ΔRMSE vs HPG_baseline
# ──────────────────────────────────────────────────────────────────────

def plot_heatmap_delta_rmse(deltas: pd.DataFrame) -> None:
    """One annotated heatmap per target: rows=model, cols=method, values=mean ΔRMSE."""
    for target_short in TARGETS.values():
        sub = deltas[deltas["target"] == target_short]
        pivot = sub.pivot_table(
            index="model", columns="method", values="delta_rmse",
            aggfunc="mean", observed=False,
        )
        pivot = pivot.reindex(index=GNN_MODELS, columns=DELTA_ORDER)

        fig, ax = plt.subplots(figsize=(5, 3))
        vmax = max(abs(pivot.values.min()), abs(pivot.values.max())) * 1.1
        sns.heatmap(
            pivot, annot=True, fmt=".4f", center=0,
            cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            linewidths=0.5, ax=ax,
            cbar_kws={"label": "mean ΔRMSE (eV)"},
        )
        ax.set_title(f"{target_short}: mean ΔRMSE vs HPG_baseline", fontsize=13)
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        out = OUTPUT_DIR / f"ablation0_delta_rmse_heatmap_{target_short}.png"
        fig.savefig(out)
        plt.close(fig)
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
        .dropna()
    )
    print(summary.to_string())

    print("\n" + "=" * 70)
    print("SUMMARY: Mean ΔRMSE vs HPG_baseline (negative = better)")
    print("=" * 70)
    delta_summary = (
        deltas.groupby(["target", "model", "method"], observed=False)["delta_rmse"]
        .agg(["mean", "std"])
        .round(5)
    )
    print(delta_summary.to_string())

    print("\n" + "=" * 70)
    print("SUMMARY: Mean ΔR² vs HPG_baseline (positive = better)")
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
    plot_scatter_rmse(deltas)
    plot_win_rate(deltas)
    plot_heatmap_delta_rmse(deltas)

    print_summary(data, deltas)
    print(f"\nAll plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
