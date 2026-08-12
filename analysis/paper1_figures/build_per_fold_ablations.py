"""
build_per_fold_ablations.py

Per-fold line plots for two ablation arms:
    * Factor 2 — positional embeddings off vs on (A-split, monomer_heldout)
    * Factor 5 — K=1 vs K=16 random-octamer reads (B-split, monomer_b_heldout_clustered)

Both figures use two 1×5 rows (one row per target, one column per metric).
The factor-5 figure overlays three row subsets — random only, block+alternating,
and all rows — because the K=1 ablation only directly changes the featurisation
of random polymers.  Factor 2 is shown on the full (all rows) subset because the
position-embedding ablation acts on every row.

Usage (from project root):
    .venv/bin/python analysis/paper1_figures/build_per_fold_ablations.py
"""
from __future__ import annotations

import importlib.util
import sys
import warnings
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluation.metrics import compute_copolymer_metrics

# Load the shared figure style used by the other paper1_figures scripts.
_spec = importlib.util.spec_from_file_location(
    "figstyle", ROOT / "29-07-2026 supervisor_update" / "figstyle.py"
)
figstyle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(figstyle)

OUTDIR = ROOT / "analysis" / "paper1_figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = ROOT / "data" / "ea_ip.csv"

SEEDS = [42, 43, 44]
TARGETS = {"EA": "EA_vs_SHE_eV", "IP": "IP_vs_SHE_eV"}
TARGET_TITLES = {"EA": "EA vs SHE (eV)", "IP": "IP vs SHE (eV)"}
METRICS = ["overall_r2", "rmse", "mae", "group_mean_r2", "delta_r2"]
METRIC_LABELS = {
    "overall_r2": r"overall $R^2$",
    "rmse": "RMSE (eV)",
    "mae": "MAE (eV)",
    "group_mean_r2": r"group-mean $R^2$",
    "delta_r2": r"$\Delta R^2$",
}

# Row subsets.  `None` means use all rows; a tuple is the set of allowed poly_type values.
SUBSET_DEFS: OrderedDict[str, tuple[str, ...] | None] = OrderedDict([
    ("all", None),
    ("random", ("random",)),
    ("block_alternating", ("block", "alternating")),
])

SUBSET_LABELS = {
    "all": "all rows",
    "random": "random only",
    "block_alternating": "block + alternating",
}

# Base plotting style per subset.  The primary subset of each arm is promoted to
# solid lines with markers below.
SUBSET_BASE_STYLE = {
    "all": {"linestyle": ":", "linewidth": 1.2, "alpha": 0.75, "has_markers": False, "zorder": 3},
    "random": {"linestyle": "-", "linewidth": 1.4, "alpha": 0.85, "has_markers": False, "zorder": 5},
    "block_alternating": {"linestyle": "--", "linewidth": 1.3, "alpha": 0.85, "has_markers": False, "zorder": 4},
}

# Legend order: primary first, then control, then context.
LEGEND_SUBSET_ORDER = ["random", "block_alternating", "all"]

DISPLAY_TITLES = {
    "factor2_posemb_per_fold": "Factor 2 — positional embeddings (off vs on)",
    "factor5_k1_per_fold": "Factor 5 — K=1 vs K=16 octamer reads",
}

# File-naming conventions for the two ablation arms.
ARMS: OrderedDict[str, dict] = OrderedDict([
    ("factor2_posemb_per_fold", {
        "title": DISPLAY_TITLES["factor2_posemb_per_fold"],
        "split_name": "monomer_heldout",
        "split_label": "A-split (monomer_heldout)",
        "baseline_root": ROOT / "predictions" / "regen_v1" / "ea_ip_lomo",
        "ablated_root": ROOT / "predictions" / "octamer_posemb" / "ea_ip_lomo",
        "model": "hpg_hier_octamer",
        "ablated_suffix": "__noposemb",
        "s_folds": None,
        "d_folds": None,
        "primary_subset": "all",
        "subsets": OrderedDict([("all", None)]),
    }),
    ("factor5_k1_per_fold", {
        "title": DISPLAY_TITLES["factor5_k1_per_fold"],
        "split_name": "monomer_b_heldout_clustered",
        "split_label": "B-split (monomer_b_heldout_clustered)",
        "baseline_root": ROOT / "predictions" / "regen_v1" / "ea_ip_lomo_b_clustered",
        "ablated_root": ROOT / "predictions" / "octamer_k1" / "ea_ip_lomo_b_clustered",
        "model": "hpg_hier_octamer",
        "ablated_suffix": "__k1",
        "s_folds": [0, 1, 2, 3],
        "d_folds": [4, 5, 6, 7, 8],
        "primary_subset": "random",
        "subsets": OrderedDict([
            ("random", ("random",)),
            ("block_alternating", ("block", "alternating")),
            ("all", None),
        ]),
    }),
])

# §5 expected ranges for ablated - baseline, to 3 decimal places (all-rows subset).
EXPECTED_RANGES = {
    "factor2_posemb_per_fold": {
        "EA": {
            "overall_r2": (-0.008, 0.035),
            "rmse": (-0.060, 0.012),
            "mae": (-0.051, 0.011),
            "group_mean_r2": (-0.008, 0.037),
            "delta_r2": (-0.053, 0.105),
        },
        "IP": {
            "overall_r2": (-0.032, 0.088),
            "rmse": (-0.032, 0.013),
            "mae": (-0.027, 0.013),
            "group_mean_r2": (-0.032, 0.092),
            "delta_r2": (-0.108, 0.051),
        },
    },
    "factor5_k1_per_fold": {
        "EA": {
            "S": {
                "overall_r2": (-0.010, 0.001),
                "rmse": (-0.002, 0.011),
                "mae": (-0.001, 0.008),
                "group_mean_r2": (-0.010, 0.001),
                "delta_r2": (-0.002, 0.029),
            },
            "D": {
                "overall_r2": (-0.006, 0.018),
                "rmse": (-0.015, 0.007),
                "mae": (-0.010, 0.002),
                "group_mean_r2": (-0.006, 0.019),
                "delta_r2": (-0.024, 0.032),
            },
        },
        "IP": {
            "S": {
                "overall_r2": (-0.003, 0.003),
                "rmse": (-0.004, 0.003),
                "mae": (-0.000, 0.002),
                "group_mean_r2": (-0.004, 0.003),
                "delta_r2": (-0.019, 0.005),
            },
            "D": {
                "overall_r2": (-0.006, 0.010),
                "rmse": (-0.006, 0.008),
                "mae": (-0.004, 0.005),
                "group_mean_r2": (-0.006, 0.010),
                "delta_r2": (-0.062, 0.025),
            },
        },
    },
}


def _prediction_path(root: Path, model: str, target_token: str, split: str,
                     fold: int, seed: int, suffix: str) -> Path:
    return root / f"ea_ip__{target_token}__{model}__{split}__fold{fold}__s{seed}{suffix}.npz"


def _load_one(path: Path) -> dict | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as archive:
        return {
            "y_true": archive["y_true"].astype(np.float64).ravel(),
            "y_pred": archive["y_pred"].astype(np.float64).ravel(),
            "indices": archive["test_indices"].astype(int).ravel(),
        }


def _avg_seeds(root: Path, model: str, target_token: str, split: str,
               fold: int, seeds: list[int], suffix: str) -> dict | None:
    """Average predictions across available seeds; return n_seeds and averaged arrays."""
    loaded = [_load_one(_prediction_path(root, model, target_token, split, fold, s, suffix))
              for s in seeds]
    present = [p for p in loaded if p is not None]
    if not present:
        return None

    first = present[0]
    y_true = first["y_true"]
    indices = first["indices"]
    for p in present[1:]:
        if not np.array_equal(p["y_true"], y_true) or not np.array_equal(p["indices"], indices):
            raise AssertionError(
                f"y_true or test_indices differ across seeds for {root.name} "
                f"{target_token} {split} fold {fold}"
            )
    y_pred_avg = np.mean([p["y_pred"] for p in present], axis=0)
    return {
        "y_true": y_true,
        "y_pred": y_pred_avg,
        "indices": indices,
        "n_seeds": len(present),
    }


def _compute_subset_metrics(df: pd.DataFrame, payload: dict, allowed: tuple[str, ...] | None,
                            metrics: list[str]) -> dict:
    """Compute metrics for a row subset; fall back to pointwise metrics if no matched groups."""
    if allowed is None:
        mask = np.ones(len(payload["indices"]), dtype=bool)
    else:
        mask = df.iloc[payload["indices"]]["poly_type"].astype(str).isin(allowed).to_numpy()

    if not mask.any():
        return {m: np.nan for m in metrics}

    yt = payload["y_true"][mask]
    yp = payload["y_pred"][mask]
    idx = payload["indices"][mask]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            full, _ = compute_copolymer_metrics(df, yt, yp, idx)
            return {m: full[m] for m in metrics}
        except ValueError:
            # No matched multi-architecture groups: pointwise metrics are still defined.
            yt_arr = np.asarray(yt, dtype=np.float64)
            yp_arr = np.asarray(yp, dtype=np.float64)
            return {
                "overall_r2": float(r2_score(yt_arr, yp_arr)),
                "rmse": float(np.sqrt(np.mean((yp_arr - yt_arr) ** 2))),
                "mae": float(mean_absolute_error(yt_arr, yp_arr)),
                "group_mean_r2": np.nan,
                "delta_r2": np.nan,
            }


def _collect_arm(df: pd.DataFrame, arm: dict) -> pd.DataFrame:
    rows = []
    for fold in range(9):
        for tshort, ttoken in TARGETS.items():
            for setting, suffix, root in [
                ("baseline", "", arm["baseline_root"]),
                ("ablated", arm["ablated_suffix"], arm["ablated_root"]),
            ]:
                payload = _avg_seeds(root, arm["model"], ttoken, arm["split_name"],
                                     fold, SEEDS, suffix)
                n_seeds = payload["n_seeds"] if payload else 0

                if payload is None:
                    for subset in arm["subsets"]:
                        for m in METRICS:
                            rows.append({
                                "fold": fold,
                                "target": tshort,
                                "subset": subset,
                                "setting": setting,
                                "metric": m,
                                "value": np.nan,
                                "n_seeds": n_seeds,
                            })
                    continue

                for subset, allowed in arm["subsets"].items():
                    vals = _compute_subset_metrics(df, payload, allowed, METRICS)
                    for m in METRICS:
                        rows.append({
                            "fold": fold,
                            "target": tshort,
                            "subset": subset,
                            "setting": setting,
                            "metric": m,
                            "value": vals[m],
                            "n_seeds": n_seeds,
                        })
    return pd.DataFrame(rows)


def _assert_ranges(records: pd.DataFrame, arm_key: str) -> list[str]:
    """Check §5 ablated - baseline ranges on the all-rows subset and return notes."""
    all_rows = records[records["subset"] == "all"].copy()
    diff = all_rows.pivot_table(
        index=["fold", "target", "metric"],
        columns="setting",
        values="value",
        aggfunc="first",
    ).reset_index()
    diff["diff"] = diff["ablated"] - diff["baseline"]

    if arm_key == "factor2_posemb_per_fold":
        diff["fold_group"] = "all"
    else:
        diff["fold_group"] = diff["fold"].apply(lambda f: "S" if f <= 3 else "D")

    notes = []
    all_ok = True
    expected = EXPECTED_RANGES[arm_key]

    for target in ["EA", "IP"]:
        groups = ["all"] if arm_key == "factor2_posemb_per_fold" else ["S", "D"]
        for group in groups:
            sub = diff[(diff["target"] == target) & (diff["fold_group"] == group)]
            for metric in METRICS:
                vals = sub[sub["metric"] == metric]["diff"].dropna().values
                if len(vals) == 0:
                    note = f"| {target} | {group} | {metric} | no data | no data | FAIL |"
                    notes.append(note)
                    all_ok = False
                    continue
                lo = round(float(vals.min()), 3)
                hi = round(float(vals.max()), 3)

                if arm_key == "factor2_posemb_per_fold":
                    exp_lo, exp_hi = expected[target][metric]
                else:
                    exp_lo, exp_hi = expected[target][group][metric]

                ok = (lo == exp_lo) and (hi == exp_hi)
                if not ok:
                    all_ok = False
                notes.append(
                    f"| {target} | {group} | {metric} | "
                    f"{lo} to {hi} | {exp_lo} to {exp_hi} | "
                    f"{'PASS' if ok else 'FAIL'} |"
                )

    if not all_ok:
        raise AssertionError(
            f"{arm_key}: §5 ablated - baseline ranges disagree with expected. "
            "Do not adjust — stop and reconcile.\n" + "\n".join(notes)
        )
    return notes


def _auto_ylim(ax: plt.Axes, values: list[float], metric: str) -> tuple[float, float]:
    """Set y-limits that include the data; return (y_lo, y_hi)."""
    vals = np.array([v for v in values if np.isfinite(v)], dtype=np.float64)
    if len(vals) == 0:
        return ax.get_ylim()

    lo, hi = float(vals.min()), float(vals.max())
    if metric in ("overall_r2", "group_mean_r2", "delta_r2"):
        y_hi = min(1.0, hi) + 0.02
        if lo < 0:
            y_lo = lo - 0.03 * (y_hi - lo)
        else:
            y_lo = max(0.0, lo - 0.03 * (y_hi - lo))
    else:
        pad = 0.03 * (hi - lo) if hi != lo else 0.02 * hi
        y_hi = hi + pad
        y_lo = max(0.0, lo - pad)

    ax.set_ylim(y_lo, y_hi)
    return y_lo, y_hi


def _add_offscale_markers(ax: plt.Axes, folds: list[int], values: list[float],
                          color: str, y_lo: float, y_hi: float) -> None:
    """Mark points that fall outside the chosen y-limits, with their real value."""
    eps = 1e-9
    bbox = dict(facecolor="white", alpha=0.85, edgecolor="none")
    for f, v in zip(folds, values):
        if not np.isfinite(v):
            continue
        if v < y_lo - eps:
            ax.scatter([f], [y_lo], marker="v", color=color, s=40,
                       zorder=10, clip_on=False)
            ax.text(f, y_lo, f"{v:.2f}", ha="center", va="top",
                    fontsize=5.5, color=color, bbox=bbox, zorder=20,
                    clip_on=False)
        elif v > y_hi + eps:
            ax.scatter([f], [y_hi], marker="^", color=color, s=40,
                       zorder=10, clip_on=False)
            ax.text(f, y_hi, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=5.5, color=color, bbox=bbox, zorder=20,
                    clip_on=False)


def _subset_style(subset: str, is_primary: bool) -> dict:
    """Return line style for a subset, promoting the primary subset."""
    style = SUBSET_BASE_STYLE[subset].copy()
    if is_primary:
        style["linestyle"] = "-"
        style["linewidth"] = 2.0
        style["alpha"] = 1.0
        style["has_markers"] = True
        style["zorder"] = 5
    return style


def _plot_series(ax: plt.Axes, folds: np.ndarray, values: np.ndarray,
                 n_seeds: np.ndarray, color: str, marker: str,
                 style: dict) -> None:
    """Plot one baseline/ablated series for a subset."""
    lw = style["linewidth"]
    ls = style["linestyle"]
    alpha = style["alpha"]
    z = style["zorder"]

    # Plot the connecting line.
    ax.plot(folds, values, color=color, lw=lw, ls=ls, alpha=alpha, zorder=z)

    # Markers only for the primary subset, and only for < 3 seeds on any subset.
    if style["has_markers"]:
        for f, v, n in zip(folds, values, n_seeds):
            if np.isfinite(v):
                if n < 3:
                    ax.plot([f], [v], color=color, marker=marker, linestyle="none",
                            markerfacecolor="none", markeredgecolor=color,
                            markeredgewidth=1.2, zorder=z + 1)
                else:
                    ax.plot([f], [v], color=color, marker=marker, linestyle="none",
                            markerfacecolor=color, markeredgecolor=color,
                            markeredgewidth=0.8, zorder=z + 1)


def _plot_arm(records: pd.DataFrame, arm: dict, arm_key: str) -> None:
    figstyle.apply_style("paper")
    plt.rcParams["savefig.dpi"] = 200

    base_color = figstyle.MODEL_COLORS.get("hpg_hier_octamer", figstyle.COLORS[0])
    abl_color = figstyle.OI_VERMILLION

    fig, axes = plt.subplots(2, 5, figsize=(14.5, 6.0), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array(axes)
    axes = axes.reshape(2, 5)

    # Plot order: draw context subsets first, then primary on top.
    plot_order = [s for s in ["all", "block_alternating", "random"] if s in arm["subsets"]]

    for row, target in enumerate(["EA", "IP"]):
        for col, metric in enumerate(METRICS):
            ax = axes[row, col]
            sub = records[(records["target"] == target) & (records["metric"] == metric)]

            all_values = []
            for subset in plot_order:
                baseline = sub[(sub["setting"] == "baseline") & (sub["subset"] == subset)]
                ablated = sub[(sub["setting"] == "ablated") & (sub["subset"] == subset)]
                baseline = baseline.sort_values("fold").reset_index(drop=True)
                ablated = ablated.sort_values("fold").reset_index(drop=True)

                b_vals = baseline["value"].to_numpy(dtype=float)
                a_vals = ablated["value"].to_numpy(dtype=float)
                folds = baseline["fold"].to_numpy(dtype=int)

                all_values.extend([v for v in b_vals if np.isfinite(v)])
                all_values.extend([v for v in a_vals if np.isfinite(v)])

            y_lo, y_hi = _auto_ylim(ax, all_values, metric)

            # Draw each subset series.
            for subset in plot_order:
                baseline = sub[(sub["setting"] == "baseline") & (sub["subset"] == subset)]
                ablated = sub[(sub["setting"] == "ablated") & (sub["subset"] == subset)]
                baseline = baseline.sort_values("fold").reset_index(drop=True)
                ablated = ablated.sort_values("fold").reset_index(drop=True)

                b_vals = baseline["value"].to_numpy(dtype=float)
                a_vals = ablated["value"].to_numpy(dtype=float)
                folds = baseline["fold"].to_numpy(dtype=int)

                is_primary = (arm["primary_subset"] == subset)
                style = _subset_style(subset, is_primary)

                _plot_series(ax, folds, b_vals, baseline["n_seeds"].to_numpy(),
                             base_color, "o", style)
                _plot_series(ax, folds, a_vals, ablated["n_seeds"].to_numpy(),
                             abl_color, "^", style)

                _add_offscale_markers(ax, folds.tolist(), b_vals.tolist(),
                                      base_color, y_lo, y_hi)
                _add_offscale_markers(ax, folds.tolist(), a_vals.tolist(),
                                      abl_color, y_lo, y_hi)

            # If the random-only subset has no data for a group metric, add the required note.
            if "random" in arm["subsets"] and metric in ("group_mean_r2", "delta_r2"):
                random_b = sub[(sub["setting"] == "baseline") & (sub["subset"] == "random")]["value"]
                random_a = sub[(sub["setting"] == "ablated") & (sub["subset"] == "random")]["value"]
                if not np.isfinite(random_b).any() and not np.isfinite(random_a).any():
                    ax.text(0.05, 0.12,
                            "Random-only:\nnot computable",
                            transform=ax.transAxes, ha="left", va="bottom",
                            fontsize=6.5, zorder=20,
                            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))

            ax.set_xlim(-0.5, 8.5)
            ax.set_xticks(folds)
            ax.set_xlabel("Fold")
            ax.set_title(f"{target}: {METRIC_LABELS[metric]}", fontsize=8.5)

            # S/D boundary for the B split.
            if arm["s_folds"] is not None:
                ax.axvline(3.5, color="#999999", lw=0.9, ls="--", zorder=1)
                if row == 0:
                    bbox = dict(facecolor="white", alpha=0.85, edgecolor="none")
                    ax.text(1.5, y_hi - 0.05 * (y_hi - y_lo),
                            "S", ha="center", va="top", fontsize=8,
                            color="#555555", bbox=bbox, zorder=20)
                    ax.text(6.0, y_hi - 0.05 * (y_hi - y_lo),
                            "D", ha="center", va="top", fontsize=8,
                            color="#555555", bbox=bbox, zorder=20)

    # Build a single figure-level legend.
    legend_subsets = [s for s in LEGEND_SUBSET_ORDER if s in arm["subsets"]]
    handles = []
    for subset in legend_subsets:
        is_primary = (arm["primary_subset"] == subset)
        style = _subset_style(subset, is_primary)
        has_markers = style["has_markers"]
        ls = style["linestyle"]
        lw = style["linewidth"]
        alpha = style["alpha"]

        # Baseline handle.
        handles.append(
            plt.Line2D([0], [0], color=base_color, ls=ls, lw=lw, alpha=alpha,
                       marker="o" if has_markers else "none",
                       markerfacecolor=base_color if has_markers else "none",
                       markeredgecolor=base_color if has_markers else "none",
                       ms=6 if has_markers else 0,
                       label=f"baseline ({SUBSET_LABELS[subset]})")
        )
        # Ablated handle.
        handles.append(
            plt.Line2D([0], [0], color=abl_color, ls=ls, lw=lw, alpha=alpha,
                       marker="^" if has_markers else "none",
                       markerfacecolor=abl_color if has_markers else "none",
                       markeredgecolor=abl_color if has_markers else "none",
                       ms=6 if has_markers else 0,
                       label=f"ablated ({SUBSET_LABELS[subset]})")
        )

    ncol = 2
    leg = fig.legend(handles=handles, loc="upper center", ncol=ncol,
                     fontsize=8.5, frameon=True,
                     bbox_to_anchor=(0.5, 0.94))
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("none")
    leg.set_zorder(20)

    fig.suptitle(f"{arm['title']} — {arm['split_label']}",
                 fontsize=10, y=0.99)

    plt.tight_layout(rect=[0, 0.03, 1, 0.84])

    for fmt in ("png", "pdf"):
        out_path = OUTDIR / f"{arm_key}.{fmt}"
        fig.savefig(out_path)
        print(f"  wrote {out_path.relative_to(ROOT)}")
    plt.close(fig)


def _write_manifest(arm_key: str, arm: dict, records: pd.DataFrame,
                    assertion_notes: list[str], missing_cells: list[str]) -> None:
    n_cells = len(records)
    n_expected = 9 * 2 * len(arm["subsets"]) * 2 * 5

    incomplete = records[records["n_seeds"] < 3]
    if len(incomplete):
        incomplete_summary = "\n".join(
            f"- {row.target} fold {row.fold} {row.setting} {row.subset} {row.metric}: n_seeds={row.n_seeds}"
            for _, row in incomplete.iterrows()
        )
    else:
        incomplete_summary = "None — all plotted cells have 3 seeds."

    if missing_cells:
        missing_summary = "\n".join(f"- {c}" for c in missing_cells)
    else:
        missing_summary = "None"

    subset_rows = "\n".join(
        f"- `{subset}`: {SUBSET_LABELS[subset]}" + (
            f" (poly_type in {list(allowed)})" if allowed else " (all rows)"
        )
        for subset, allowed in arm["subsets"].items()
    )

    if arm["s_folds"] is not None:
        split_boundary = (
            f"S folds (same-scaffold interpolation): {arm['s_folds']}\n"
            f"D folds (cross-scaffold extrapolation): {arm['d_folds']}"
        )
    else:
        split_boundary = "No S/D boundary (A-split)."

    lines = [
        f"# {DISPLAY_TITLES[arm_key]}",
        "",
        "## Layout",
        "Two 1×5 rows per figure: one row per target (EA top, IP bottom),",
        "one column per metric in the order:",
        "overall_r2, rmse, mae, group_mean_r2, delta_r2.",
        f"Factor 5 overlays three row subsets in each panel; Factor 2 uses `all` rows only.",
        "",
        "## Prediction paths and seed handling",
        f"Baseline: `{arm['baseline_root'].relative_to(ROOT)}`",
        f"Ablated:  `{arm['ablated_root'].relative_to(ROOT)}`",
        f"Model: `{arm['model']}` | split: `{arm['split_name']}` | seeds averaged at prediction level (42/43/44)",
        f"Ablated filename suffix: `{arm['ablated_suffix']}`.",
        "",
        "## Row subsets used",
        subset_rows,
        "",
        "## Split",
        split_boundary,
        "",
        "## Cells",
        f"Expected: 9 folds × 2 targets × {len(arm['subsets'])} subsets × 2 settings × 5 metrics = {n_expected} rows.",
        f"Plotted rows: {n_cells}.",
        "",
        "## Incomplete cells (n_seeds < 3)",
        incomplete_summary,
        "",
        "## Missing prediction files",
        missing_summary,
        "",
        "## §5 ablated − baseline range assertions (all-rows subset)",
        "| target | group | metric | computed | expected | result |",
        "| --- | --- | --- | --- | --- | --- |",
    ] + assertion_notes

    manifest = "\n".join(lines)
    out_path = OUTDIR / f"{arm_key}_manifest.md"
    out_path.write_text(manifest)
    print(f"  wrote {out_path.relative_to(ROOT)}")


def build_one(arm_key: str, df: pd.DataFrame) -> None:
    print(f"\nBuilding {arm_key} ...")
    arm = ARMS[arm_key]
    records = _collect_arm(df, arm)

    # Detect missing cells (one missing file affects all its subsets).
    missing_cells = (
        records[records["n_seeds"] == 0][["target", "fold", "setting"]]
        .drop_duplicates()
        .apply(lambda r: f"{r.target} fold {r.fold} {r.setting}", axis=1)
        .tolist()
    )

    # Assert §5 ranges on the all-rows subset first; raise if they disagree.
    assertion_notes = _assert_ranges(records, arm_key)
    print(f"  §5 range assertions passed ({len(assertion_notes)} checks)")

    # Write the plotted-value CSV.
    csv_path = OUTDIR / f"{arm_key}.csv"
    records.to_csv(csv_path, index=False)
    print(f"  wrote {csv_path.relative_to(ROOT)}")

    # Render the figure.
    _plot_arm(records, arm, arm_key)

    # Write manifest.
    _write_manifest(arm_key, arm, records, assertion_notes, missing_cells)


def main() -> None:
    print(f"Loading {DATA_PATH.relative_to(ROOT)} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df)} rows, columns: {list(df.columns)}")

    for arm_key in ARMS:
        build_one(arm_key, df)

    print(f"\nAll figures, CSVs and manifests written to {OUTDIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
