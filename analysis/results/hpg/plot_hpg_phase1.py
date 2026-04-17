"""Publication-quality plots for HPG Phase 1 variant comparison.

Compares three HPG representation strategies on ea_ip (a_held_out split):

  * HPG_baseline    — original sum pooling, no X_d
  * HPG_frac        — fraction-weighted pooling over fragment nodes
  * HPG_frac_polytype — fraction-weighted pooling + polytype one-hot as X_d

Generates seven figures:
  1. phase1_absolute_rmse.png      — absolute RMSE by method
  2. phase1_delta_rmse.png         — ΔRMSE vs HPG_baseline
  3. phase1_delta_r2.png           — ΔR² vs HPG_baseline
  4. phase1_incremental.png        — step-by-step Δ (+frac → +polytype)
  5. phase1_scatter_rmse.png       — scatter: method RMSE vs baseline RMSE
  6. phase1_win_rate.png           — fraction of folds that beat baseline
  7. phase1_variance.png           — RMSE std across folds (stability)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_DIR = Path(__file__).resolve().parent / "hpg_phase1_report"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# model name → subdirectory under RESULTS_DIR
# For Phase 1 there is only one architecture (HPG).
# When Phase 2 compares HPG against DMPNN/GIN/GAT baselines, extend this.
MODELS = ["HPG"]

TARGETS = {
    "EA vs SHE (eV)": "EA",
    "IP vs SHE (eV)": "IP",
}

# Internal key → display label used in all plots
METHOD_MAP: dict[str, str] = {
    "baseline":      "HPG_baseline",
    "frac":          "HPG_frac",
    "frac_polytype": "HPG_frac_polytype",
}

# Canonical display order kept consistent across all plots
METHOD_ORDER = ["HPG_baseline", "HPG_frac", "HPG_frac_polytype"]

BASELINE = "HPG_baseline"
SPLIT    = "a_held_out"
DATASET  = "ea_ip"

# ---------------------------------------------------------------------------
# Filename candidate lists  (tried in order; first existing file wins)
#
# Each entry is a (stem, is_per_target) tuple:
#   is_per_target=True  → stem already embeds the target string;
#                         one file per target, no filtering needed.
#   is_per_target=False → single file holds ALL targets in a 'target' column;
#                         filtered by target after loading.
#
# Ordering: preferred (new naming) → current/legacy (old naming).
# When future experiments produce hpg_frac / hpg_frac_polytype files the
# script will automatically pick those up without any further changes.
# ---------------------------------------------------------------------------
METHOD_CANDIDATES: dict[str, list[tuple[str, bool]]] = {
    "baseline": [
        # New per-target format (current — hpg_variant=baseline, default)
        (f"{DATASET}__{SPLIT}__target_{{target}}", True),
        # Old multi-target format (legacy fallback)
        (f"{DATASET}__HPG_baseline__{SPLIT}", False),
    ],
    "frac": [
        # Per-target format produced by hpg_variant=frac
        (f"{DATASET}__hpg_frac__{SPLIT}__target_{{target}}", True),
    ],
    "frac_polytype": [
        # Per-target format produced by hpg_variant=frac_polytype + incl_poly_type
        (f"{DATASET}__hpg_frac_polytype__poly_type__{SPLIT}__target_{{target}}", True),
    ],
}

# Palette: one colour per model architecture (extensible for Phase 2)
MODEL_PALETTE: dict[str, str] = {
    "HPG":   "#2c7bb6",
    "DMPNN": "#1b9e77",
    "GIN":   "#d95f02",
    "GAT":   "#7570b3",
}


# ──────────────────────────────────────────────────────────────────────
# Filename resolution  (priority-based candidate lookup)
# ──────────────────────────────────────────────────────────────────────

def _resolve_file(
    model: str,
    method_key: str,
    target_full: str,
    results_dir: Path = RESULTS_DIR,
) -> tuple[Path, bool] | tuple[None, None]:
    """Return the first existing (path, is_per_target) candidate for this
    model / method / target combination, or (None, None) if nothing found.

    Candidate stems in METHOD_CANDIDATES are tried in declaration order so
    that preferred (new-naming) files are chosen over legacy ones.
    """
    candidates = METHOD_CANDIDATES.get(method_key, [])
    for stem_template, is_per_target in candidates:
        stem = stem_template.format(target=target_full)
        path = results_dir / model / (stem + "_results.csv")
        if path.exists():
            return path, is_per_target
    return None, None


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────

def load_phase1_results(
    results_dir: Path = RESULTS_DIR,
    models: list[str] = MODELS,
    method_map: dict[str, str] = METHOD_MAP,
    targets: dict[str, str] = TARGETS,
) -> pd.DataFrame:
    """Load all Phase 1 HPG variant CSVs into a tidy DataFrame.

    Handles two file formats transparently:
    - **Per-target** (new):   one CSV per target; target encoded in filename.
    - **Multi-target** (old): one CSV for all targets; filtered by 'target' col.

    Returns
    -------
    DataFrame with columns: model, target, method, fold, rmse, r2, mae
    """
    rows: list[dict] = []
    for model in models:
        for method_key, method_label in method_map.items():
            for target_full, target_short in targets.items():
                fpath, is_per_target = _resolve_file(
                    model, method_key, target_full, results_dir
                )
                if fpath is None:
                    # Show what was tried so it's easy to diagnose
                    tried = [
                        stem_t.format(target=target_full) + "_results.csv"
                        for stem_t, _ in METHOD_CANDIDATES.get(method_key, [])
                    ]
                    print(f"  SKIP ({method_label} / {target_short}):  "
                          f"none of {len(tried)} candidate(s) found")
                    for t in tried:
                        print(f"    - {t}")
                    continue

                df_raw = pd.read_csv(fpath)

                # Multi-target files: filter to the requested target
                if not is_per_target:
                    df_raw = df_raw[df_raw["target"] == target_full]

                if df_raw.empty:
                    print(f"  SKIP ({method_label} / {target_short}): "
                          f"no rows after filtering {fpath.name}")
                    continue

                print(f"  OK  [{method_label:20s} / {target_short:2s}]  "
                      f"{fpath.name}")
                for _, r in df_raw.iterrows():
                    rows.append(
                        {
                            "model":  model,
                            "target": target_short,
                            "method": method_label,
                            "fold":   int(r["split"]),
                            "rmse":   float(r["test/rmse"]),
                            "r2":     float(r["test/r2"]),
                            "mae":    float(r["test/mae"]),
                        }
                    )

    df = pd.DataFrame(rows)
    if df.empty:
        print("WARNING: No data loaded — check RESULTS_DIR and filenames.")
        return df

    present = df["method"].unique().tolist()
    print(
        f"\nLoaded {len(df)} rows  "
        f"({df['model'].nunique()} model(s), "
        f"{df['target'].nunique()} target(s), "
        f"{df['method'].nunique()} method(s))"
    )
    missing = [m for m in METHOD_ORDER if m not in present]
    if missing:
        print(f"  Methods still missing: {missing}")
    return df


# ──────────────────────────────────────────────────────────────────────
# Delta & incremental computations
# ──────────────────────────────────────────────────────────────────────

def compute_deltas(
    df: pd.DataFrame,
    baseline: str = BASELINE,
    metric: str = "rmse",
) -> pd.DataFrame:
    """Per-fold Δmetric relative to *baseline* method.

    Returns a DataFrame (baseline rows excluded) with an extra column
    ``delta_{metric}``.  Alignment is on (model, target, fold).
    """
    if baseline not in df["method"].values:
        raise ValueError(
            f"Baseline '{baseline}' not found in data. "
            f"Available: {df['method'].unique().tolist()}"
        )
    base = (
        df[df["method"] == baseline]
        .set_index(["model", "target", "fold"])[[metric]]
        .rename(columns={metric: f"{metric}_baseline"})
    )
    merged = df.merge(base, on=["model", "target", "fold"], how="inner")
    merged[f"delta_{metric}"] = merged[metric] - merged[f"{metric}_baseline"]
    return merged[merged["method"] != baseline].copy()


def compute_incremental_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Compute step-by-step incremental ΔRMSE.

    Steps:
      +frac      : rmse(HPG_frac)          − rmse(HPG_baseline)
      +polytype  : rmse(HPG_frac_polytype) − rmse(HPG_frac)

    Returns a tidy DataFrame with columns:
        model, target, fold, step, delta_rmse
    """
    required = ["HPG_baseline", "HPG_frac", "HPG_frac_polytype"]
    missing = [m for m in required if m not in df["method"].values]
    if missing:
        print(f"  WARNING: incremental plot skipped — missing methods: {missing}")
        return pd.DataFrame()

    pivot = df.pivot_table(
        index=["model", "target", "fold"],
        columns="method",
        values="rmse",
    ).reset_index()

    rows: list[dict] = []
    for _, r in pivot.iterrows():
        base = (
            dict(model=r["model"], target=r["target"], fold=int(r["fold"]))
        )
        if "HPG_frac" in pivot.columns and "HPG_baseline" in pivot.columns:
            rows.append({**base, "step": "+frac",
                         "delta_rmse": r["HPG_frac"] - r["HPG_baseline"]})
        if "HPG_frac_polytype" in pivot.columns and "HPG_frac" in pivot.columns:
            rows.append({**base, "step": "+polytype",
                         "delta_rmse": r["HPG_frac_polytype"] - r["HPG_frac"]})

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# Style & layout helpers
# ──────────────────────────────────────────────────────────────────────

def _setup_style() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.3)
    plt.rcParams.update(
        {
            "figure.dpi":    150,
            "savefig.dpi":   300,
            "savefig.bbox":  "tight",
            "font.family":   "sans-serif",
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.4,
            "grid.alpha":    0.5,
        }
    )


def _sync_ylims(axes: list[plt.Axes], margin_frac: float = 0.05) -> None:
    """Force all axes to share the same y-limits."""
    ymin = min(ax.get_ylim()[0] for ax in axes)
    ymax = max(ax.get_ylim()[1] for ax in axes)
    margin = (ymax - ymin) * margin_frac
    for ax in axes:
        ax.set_ylim(ymin - margin, ymax + margin)


def _place_unified_legend(
    axes: list[plt.Axes],
    models: list[str],
    palette: dict[str, str],
    title: str = "Model",
    loc: str = "best",
) -> None:
    """Remove per-axis legends; place one consolidated legend on the last axis."""
    for ax in axes:
        leg = ax.get_legend()
        if leg:
            leg.remove()
    present = [m for m in models if m in palette]
    patches = [
        plt.matplotlib.patches.Patch(color=palette[m], label=m)
        for m in present
    ]
    axes[-1].legend(
        handles=patches, title=title, loc=loc, framealpha=0.9
    )


def _box_strip(
    ax: plt.Axes,
    data: pd.DataFrame,
    x: str,
    y: str,
    order: list[str],
    models: list[str],
    palette: dict[str, str],
    hue: str = "model",
) -> None:
    """Shared box + strip layer drawn onto *ax*."""
    hue_order = [m for m in models if m in data[hue].unique()]
    common = dict(data=data, x=x, y=y, hue=hue,
                  order=order, hue_order=hue_order,
                  palette=palette, ax=ax)
    sns.boxplot(**common, width=0.6, linewidth=0.8, fliersize=0)
    sns.stripplot(**common, dodge=True, size=5, alpha=0.7,
                  edgecolor="black", linewidth=0.4, legend=False)


# ──────────────────────────────────────────────────────────────────────
# Task 1 — Absolute RMSE
# ──────────────────────────────────────────────────────────────────────

def plot_absolute_rmse(
    df: pd.DataFrame,
    models: list[str] = MODELS,
    palette: dict[str, str] = MODEL_PALETTE,
    method_order: Optional[list[str]] = None,
    targets: list[str] = ["EA", "IP"],
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Task 1: absolute RMSE boxplot, one column per target."""
    _setup_style()
    order = method_order or [m for m in METHOD_ORDER if m in df["method"].unique()]

    fig, axes = plt.subplots(
        1, len(targets), figsize=(6 * len(targets), 5), sharey=True
    )
    if len(targets) == 1:
        axes = [axes]

    for ax, tgt in zip(axes, targets):
        sub = df[df["target"] == tgt]
        if sub.empty:
            ax.set_title(tgt); continue
        _box_strip(ax, sub, "method", "rmse", order, models, palette)
        ax.set_title(tgt, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("RMSE (eV)" if ax is axes[0] else "")
        ax.tick_params(axis="x", rotation=15)

    _sync_ylims(axes)
    _place_unified_legend(axes, models, palette)
    fig.suptitle("Phase 1: RMSE by HPG variant", fontsize=14,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    out = output_dir / "phase1_absolute_rmse.png"
    fig.savefig(out); plt.close(fig)
    print(f"Saved: {out}")


# ──────────────────────────────────────────────────────────────────────
# Task 2 — ΔRMSE vs HPG_baseline
# ──────────────────────────────────────────────────────────────────────

def plot_delta_rmse(
    df_delta: pd.DataFrame,
    models: list[str] = MODELS,
    palette: dict[str, str] = MODEL_PALETTE,
    method_order: Optional[list[str]] = None,
    targets: list[str] = ["EA", "IP"],
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Task 2: ΔRMSE vs HPG_baseline, one column per target."""
    _setup_style()
    order = method_order or [
        m for m in METHOD_ORDER if m in df_delta["method"].unique()
    ]

    fig, axes = plt.subplots(
        1, len(targets), figsize=(5 * len(targets), 5), sharey=True
    )
    if len(targets) == 1:
        axes = [axes]

    for ax, tgt in zip(axes, targets):
        sub = df_delta[df_delta["target"] == tgt]
        if sub.empty:
            ax.set_title(tgt); continue
        _box_strip(ax, sub, "method", "delta_rmse", order, models, palette)
        ax.axhline(0, color="grey", linewidth=1.0, linestyle="--", zorder=0)
        ax.set_title(tgt, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("ΔRMSE (eV)" if ax is axes[0] else "")
        ax.tick_params(axis="x", rotation=15)

    _sync_ylims(axes)
    _place_unified_legend(axes, models, palette)
    fig.suptitle(
        "ΔRMSE vs HPG_baseline  (negative = better)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    out = output_dir / "phase1_delta_rmse.png"
    fig.savefig(out); plt.close(fig)
    print(f"Saved: {out}")


# ──────────────────────────────────────────────────────────────────────
# Task 3 — ΔR² vs HPG_baseline
# ──────────────────────────────────────────────────────────────────────

def plot_delta_r2(
    df_delta: pd.DataFrame,
    models: list[str] = MODELS,
    palette: dict[str, str] = MODEL_PALETTE,
    method_order: Optional[list[str]] = None,
    targets: list[str] = ["EA", "IP"],
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Task 3: ΔR² vs HPG_baseline, one column per target."""
    _setup_style()
    order = method_order or [
        m for m in METHOD_ORDER if m in df_delta["method"].unique()
    ]

    fig, axes = plt.subplots(
        1, len(targets), figsize=(5 * len(targets), 5), sharey=True
    )
    if len(targets) == 1:
        axes = [axes]

    for ax, tgt in zip(axes, targets):
        sub = df_delta[df_delta["target"] == tgt]
        if sub.empty:
            ax.set_title(tgt); continue
        _box_strip(ax, sub, "method", "delta_r2", order, models, palette)
        ax.axhline(0, color="grey", linewidth=1.0, linestyle="--", zorder=0)
        ax.set_title(tgt, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("ΔR²" if ax is axes[0] else "")
        ax.tick_params(axis="x", rotation=15)

    _sync_ylims(axes)
    _place_unified_legend(axes, models, palette)
    fig.suptitle(
        "ΔR² vs HPG_baseline  (positive = better)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    out = output_dir / "phase1_delta_r2.png"
    fig.savefig(out); plt.close(fig)
    print(f"Saved: {out}")


# ──────────────────────────────────────────────────────────────────────
# Task 4 — Incremental ΔRMSE (+frac → +polytype)
# ──────────────────────────────────────────────────────────────────────

def plot_incremental(
    df_incr: pd.DataFrame,
    models: list[str] = MODELS,
    palette: dict[str, str] = MODEL_PALETTE,
    targets: list[str] = ["EA", "IP"],
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Task 4: step-by-step incremental ΔRMSE (+frac, +polytype)."""
    if df_incr.empty:
        print("  Skipping incremental plot (insufficient data).")
        return
    _setup_style()
    step_order = ["+frac", "+polytype"]
    step_order = [s for s in step_order if s in df_incr["step"].unique()]

    fig, axes = plt.subplots(
        1, len(targets), figsize=(5 * len(targets), 5), sharey=True
    )
    if len(targets) == 1:
        axes = [axes]

    for ax, tgt in zip(axes, targets):
        sub = df_incr[df_incr["target"] == tgt]
        if sub.empty:
            ax.set_title(tgt); continue
        _box_strip(ax, sub, "step", "delta_rmse", step_order, models, palette)
        ax.axhline(0, color="grey", linewidth=1.0, linestyle="--", zorder=0)
        ax.set_title(tgt, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Incremental ΔRMSE (eV)" if ax is axes[0] else "")

    _sync_ylims(axes)
    _place_unified_legend(axes, models, palette)
    fig.suptitle(
        "Incremental ΔRMSE  (composition and polytype effects)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    out = output_dir / "phase1_incremental.png"
    fig.savefig(out); plt.close(fig)
    print(f"Saved: {out}")


# ──────────────────────────────────────────────────────────────────────
# Task 5 — Scatter: method RMSE vs baseline RMSE
# ──────────────────────────────────────────────────────────────────────

def plot_scatter_rmse(
    df: pd.DataFrame,
    models: list[str] = MODELS,
    palette: dict[str, str] = MODEL_PALETTE,
    methods_to_plot: Optional[list[str]] = None,
    targets: list[str] = ["EA", "IP"],
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Task 5: scatter of method RMSE vs HPG_baseline RMSE.

    One figure per non-baseline method, columns = targets.
    Diagonal y = x reference line marks no-change.
    """
    _setup_style()
    if BASELINE not in df["method"].values:
        print("  Skipping scatter plot (baseline missing).")
        return

    compare_methods = methods_to_plot or [
        m for m in METHOD_ORDER if m != BASELINE and m in df["method"].values
    ]
    if not compare_methods:
        print("  Skipping scatter plot (no non-baseline methods).")
        return

    base = (
        df[df["method"] == BASELINE]
        .set_index(["model", "target", "fold"])[["rmse"]]
        .rename(columns={"rmse": "rmse_baseline"})
    )

    for method in compare_methods:
        sub = df[df["method"] == method].merge(
            base, on=["model", "target", "fold"], how="inner"
        )
        if sub.empty:
            print(f"  Skipping scatter for {method} (no data).")
            continue

        fig, axes = plt.subplots(
            1, len(targets), figsize=(5 * len(targets), 5)
        )
        if len(targets) == 1:
            axes = [axes]

        for ax, tgt in zip(axes, targets):
            s = sub[sub["target"] == tgt]
            if s.empty:
                ax.set_title(tgt); continue

            for mdl in [m for m in models if m in s["model"].unique()]:
                ms = s[s["model"] == mdl]
                ax.scatter(
                    ms["rmse_baseline"], ms["rmse"],
                    color=palette.get(mdl, "#333333"),
                    label=mdl, s=60, alpha=0.8,
                    edgecolors="black", linewidths=0.5, zorder=3,
                )

            # Diagonal
            lims = [
                min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1]),
            ]
            ax.plot(lims, lims, "k--", linewidth=1.0, alpha=0.6, zorder=0)
            ax.set_xlim(lims); ax.set_ylim(lims)
            ax.set_aspect("equal", adjustable="box")

            ax.set_title(tgt, fontweight="bold")
            ax.set_xlabel("RMSE — HPG_baseline (eV)")
            ax.set_ylabel(f"RMSE — {method} (eV)" if ax is axes[0] else "")

        _place_unified_legend(axes, models, palette, loc="upper left")
        safe = method.replace("/", "_")
        fig.suptitle(
            f"Scatter: {method} vs HPG_baseline",
            fontsize=14, fontweight="bold", y=1.02,
        )
        fig.tight_layout()
        out = output_dir / f"phase1_scatter_{safe}.png"
        fig.savefig(out); plt.close(fig)
        print(f"Saved: {out}")


# ──────────────────────────────────────────────────────────────────────
# Task 6 — Win-rate plot
# ──────────────────────────────────────────────────────────────────────

def plot_win_rate(
    df: pd.DataFrame,
    models: list[str] = MODELS,
    palette: dict[str, str] = MODEL_PALETTE,
    methods_to_plot: Optional[list[str]] = None,
    targets: list[str] = ["EA", "IP"],
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Task 6: fraction of folds where method beats HPG_baseline on RMSE."""
    _setup_style()
    if BASELINE not in df["method"].values:
        print("  Skipping win-rate plot (baseline missing).")
        return

    compare_methods = methods_to_plot or [
        m for m in METHOD_ORDER if m != BASELINE and m in df["method"].values
    ]
    if not compare_methods:
        print("  Skipping win-rate plot (no non-baseline methods).")
        return

    base = (
        df[df["method"] == BASELINE]
        .set_index(["model", "target", "fold"])[["rmse"]]
        .rename(columns={"rmse": "rmse_baseline"})
    )
    rows: list[dict] = []
    for method in compare_methods:
        sub = df[df["method"] == method].merge(
            base, on=["model", "target", "fold"], how="inner"
        )
        if sub.empty:
            continue
        sub["wins"] = (sub["rmse"] < sub["rmse_baseline"]).astype(int)
        agg = sub.groupby(["model", "target"])["wins"].agg(
            win_pct=lambda x: 100 * x.sum() / len(x),
            n_folds="count",
        ).reset_index()
        agg["method"] = method
        rows.append(agg)

    if not rows:
        print("  Skipping win-rate plot (no merged data).")
        return
    wr = pd.concat(rows, ignore_index=True)

    order = [m for m in compare_methods if m in wr["method"].unique()]
    fig, axes = plt.subplots(
        1, len(targets), figsize=(5 * len(targets), 5), sharey=True
    )
    if len(targets) == 1:
        axes = [axes]

    for ax, tgt in zip(axes, targets):
        sub = wr[wr["target"] == tgt]
        if sub.empty:
            ax.set_title(tgt); continue

        hue_order = [m for m in models if m in sub["model"].unique()]
        sns.barplot(
            data=sub, x="method", y="win_pct", hue="model",
            order=order, hue_order=hue_order,
            palette=palette, ax=ax, width=0.6, errorbar=None,
        )
        ax.axhline(50, color="grey", linewidth=1.0, linestyle="--",
                   label="50 %", zorder=0)
        ax.set_ylim(0, 105)
        ax.set_title(tgt, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("% folds beating baseline" if ax is axes[0] else "")
        ax.tick_params(axis="x", rotation=15)

    _place_unified_legend(axes, models, palette)
    fig.suptitle(
        "Win rate vs HPG_baseline  (% folds with lower RMSE)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    out = output_dir / "phase1_win_rate.png"
    fig.savefig(out); plt.close(fig)
    print(f"Saved: {out}")


# ──────────────────────────────────────────────────────────────────────
# Task 7 — Variance / stability plot
# ──────────────────────────────────────────────────────────────────────

def plot_variance(
    df: pd.DataFrame,
    models: list[str] = MODELS,
    palette: dict[str, str] = MODEL_PALETTE,
    method_order: Optional[list[str]] = None,
    targets: list[str] = ["EA", "IP"],
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Task 7: RMSE standard deviation across folds (stability)."""
    _setup_style()
    order = method_order or [m for m in METHOD_ORDER if m in df["method"].unique()]

    std_df = (
        df.groupby(["model", "target", "method"])["rmse"]
        .std()
        .reset_index()
        .rename(columns={"rmse": "std_rmse"})
    )

    fig, axes = plt.subplots(
        1, len(targets), figsize=(5 * len(targets), 5), sharey=True
    )
    if len(targets) == 1:
        axes = [axes]

    for ax, tgt in zip(axes, targets):
        sub = std_df[std_df["target"] == tgt]
        if sub.empty:
            ax.set_title(tgt); continue

        hue_order = [m for m in models if m in sub["model"].unique()]
        sns.barplot(
            data=sub, x="method", y="std_rmse", hue="model",
            order=order, hue_order=hue_order,
            palette=palette, ax=ax, width=0.6, errorbar=None,
        )
        ax.set_title(tgt, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Std RMSE (eV)" if ax is axes[0] else "")
        ax.tick_params(axis="x", rotation=15)

    _place_unified_legend(axes, models, palette)
    fig.suptitle(
        "RMSE variability across folds  (lower = more stable)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    out = output_dir / "phase1_variance.png"
    fig.savefig(out); plt.close(fig)
    print(f"Saved: {out}")


# ──────────────────────────────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    """Print mean ± std RMSE and R² per (target, method, model)."""
    if df.empty:
        return
    summary = (
        df.groupby(["target", "method", "model"])
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            n=("fold", "count"),
        )
        .round(4)
    )
    # Reorder rows by canonical method order
    method_idx = {m: i for i, m in enumerate(METHOD_ORDER)}
    summary = summary.sort_values(
        by="method",
        key=lambda s: s.map(lambda x: method_idx.get(x, 99)),
    )
    print("\n" + "=" * 70)
    print("  Phase 1 HPG Variant Summary")
    print("=" * 70)
    print(summary.to_string())
    print()


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    df = load_phase1_results()
    if df.empty:
        return

    print_summary(df)

    targets = sorted(df["target"].unique())
    models  = sorted(df["model"].unique())
    order   = [m for m in METHOD_ORDER if m in df["method"].unique()]

    # Task 1: Absolute RMSE
    plot_absolute_rmse(df, models=models, method_order=order, targets=targets)

    # Task 2: ΔRMSE vs HPG_baseline
    if BASELINE in df["method"].values:
        delta_rmse = compute_deltas(df, baseline=BASELINE, metric="rmse")
        plot_delta_rmse(delta_rmse, models=models, targets=targets)

        # Task 3: ΔR²
        delta_r2 = compute_deltas(df, baseline=BASELINE, metric="r2")
        plot_delta_r2(delta_r2, models=models, targets=targets)

        # Task 6: Win rate
        plot_win_rate(df, models=models, targets=targets)

    # Task 4: Incremental
    df_incr = compute_incremental_deltas(df)
    plot_incremental(df_incr, models=models, targets=targets)

    # Task 5: Scatter
    plot_scatter_rmse(df, models=models, targets=targets)

    # Task 7: Variance
    plot_variance(df, models=models, method_order=order, targets=targets)

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
