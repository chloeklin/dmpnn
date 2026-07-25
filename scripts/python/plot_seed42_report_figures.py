from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.metrics import mean_absolute_error, r2_score

from analysis.diagnostics.data_loading import load_all_meta, load_dataset, load_predictions_single
from analysis.diagnostics.grouping import add_group_means, build_fold_df, filter_matched_groups, group_level_stats

SEED = 42
OUT = ROOT / "analysis" / "model_diagnostics" / "report_figures" / "all"
DIAG = ROOT / "analysis" / "model_diagnostics" / "seed_42"
MODELS = ["hpg_hier", "wdmpnn", "chemarch"]
MODEL_LABELS = {"hpg_hier": "HPG-hier", "wdmpnn": "wDMPNN", "chemarch": "ChemArch"}
COLORS = {"hpg_hier": "#E8A33D", "wdmpnn": "#12314E", "chemarch": "#1C7293"}
SPLITS = ["group_disjoint", "pair_disjoint", "monomer_heldout"]
SPLIT_LABELS = {"group_disjoint": "GD", "pair_disjoint": "PD", "monomer_heldout": "LOMO"}
TARGETS = ["EA", "IP"]
N_FOLDS = {"group_disjoint": 5, "pair_disjoint": 5, "monomer_heldout": 9}
FOLD_NAMES = ["spiro-bifluorene", "dibenzothiophene sulfone", "difluorobenzene diboronic acid", "DTT fused trithiophene", "pyrene diboronic acid", "bithiophene diboronic acid", "benzothiadiazole diboronic acid", "benzene-1,4-diboronic acid", "carbazole"]
FIGURES: list[tuple[str, str]] = []


def clean_output() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)


def save(fig: plt.Figure, name: str, description: str) -> None:
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    FIGURES.append((name, description))


def finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def clipped_limits(x: np.ndarray, y: np.ndarray, pad: float = 0.06) -> tuple[float, float]:
    v = np.concatenate([finite(np.asarray(x)), finite(np.asarray(y))])
    lo, hi = np.quantile(v, [0.002, 0.998])
    span = max(hi - lo, 0.1)
    return lo - pad * span, hi + pad * span


def grouped_metric_plot(frame: pd.DataFrame, metric: str, title: str, ylabel: str, filename: str, reference: float | None = None, clamp: tuple[float, float] | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    width = 0.22
    for ax, target in zip(axes, TARGETS):
        sub = frame[frame.target == target]
        for mi, model in enumerate(MODELS):
            medians, means = [], []
            for split in SPLITS:
                values = sub[(sub.model == model) & (sub.split == split)][metric].dropna().to_numpy()
                medians.append(np.median(values))
                means.append(np.mean(values))
            x = np.arange(len(SPLITS)) + (mi - 1) * width
            ax.bar(x, medians, width=width, color=COLORS[model], label=MODEL_LABELS[model], zorder=2)
            for si, split in enumerate(SPLITS):
                values = sub[(sub.model == model) & (sub.split == split)][metric].dropna().to_numpy()
                jitter = np.linspace(-0.045, 0.045, len(values)) if len(values) > 1 else np.array([0.0])
                shown = values.copy()
                if clamp is not None:
                    shown = np.clip(shown, clamp[0], clamp[1])
                ax.scatter(np.full(len(values), x[si]) + jitter, shown, s=18, color="white", edgecolor=COLORS[model], linewidth=0.8, zorder=3)
                if split == "monomer_heldout":
                    ax.text(x[si], medians[si] + 0.025 * (ax.get_ylim()[1] - ax.get_ylim()[0] if clamp is None else clamp[1] - clamp[0]), f"mean {means[si]:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)
                if clamp is not None and np.any((values < clamp[0]) | (values > clamp[1])):
                    bad = values[(values < clamp[0]) | (values > clamp[1])]
                    ax.annotate(f"off-scale {bad.min():.2f}", (x[si], np.clip(bad.min(), *clamp)), xytext=(2, 4), textcoords="offset points", fontsize=6, rotation=90)
        if reference is not None:
            ax.axhline(reference, color="#555555", ls="--", lw=1, zorder=1)
        ax.set_xticks(np.arange(len(SPLITS)))
        ax.set_xticklabels([SPLIT_LABELS[s] for s in SPLITS])
        ax.set_title(f"{target}: {title}\nseed 42, single seed")
        ax.grid(axis="y", alpha=0.25)
        if clamp is not None:
            ax.set_ylim(*clamp)
    axes[0].set_ylabel(ylabel)
    axes[1].legend(frameon=False, loc="best")
    fig.tight_layout()
    save(fig, filename, f"Median bars and individual fold dots for {title}; LOMO text labels give fold mean.")


def load_fold_data(df: pd.DataFrame, meta: dict[str, list]) -> dict[tuple[str, str, str, int], pd.DataFrame]:
    result: dict[tuple[str, str, str, int], pd.DataFrame] = {}
    for model in MODELS:
        for target in TARGETS:
            for split in SPLITS:
                for fold in range(N_FOLDS[split]):
                    pred = load_predictions_single(model, target, split, fold, meta[split], seed=SEED)
                    if pred is None:
                        continue
                    fdf = build_fold_df(df, pred["y_true"], pred["y_pred"], pred["global_idx"])
                    result[(model, target, split, fold)] = add_group_means(fdf)
    return result


def overall_metrics(folds: dict[tuple[str, str, str, int], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for (model, target, split, fold), fdf in folds.items():
        yt, yp = fdf.y_true.to_numpy(), fdf.y_pred.to_numpy()
        rows.append({"model": model, "target": target, "split": split, "fold": fold, "overall_r2": r2_score(yt, yp), "overall_mae": mean_absolute_error(yt, yp), "overall_rmse": np.sqrt(np.mean((yt - yp) ** 2))})
    return pd.DataFrame(rows)


def parity_grid(folds: dict[tuple[str, str, str, int], pd.DataFrame], level: str, target: str, filename: str) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for ri, model in enumerate(["wdmpnn", "hpg_hier", "chemarch"]):
        for ci, split in enumerate(SPLITS):
            ax = axes[ri, ci]
            pieces = [folds[(model, target, split, fold)] for fold in range(N_FOLDS[split]) if (model, target, split, fold) in folds]
            full = pd.concat(pieces, ignore_index=True)
            if level == "overall":
                x, y = full.y_true.to_numpy(), full.y_pred.to_numpy()
                xlabel, ylabel = "True value (eV)", "Predicted value (eV)"
            elif level == "group_mean":
                group = group_level_stats(filter_matched_groups(full))
                x, y = group.y_bar_true.to_numpy(), group.y_bar_pred.to_numpy()
                xlabel, ylabel = "True group mean (eV)", "Predicted group mean (eV)"
            else:
                matched = filter_matched_groups(full)
                x, y = matched.delta_true.to_numpy(), matched.delta_pred.to_numpy()
                xlabel, ylabel = "True architecture deviation Δy (eV)", "Predicted architecture deviation Δŷ (eV)"
            lo, hi = clipped_limits(x, y)
            ax.scatter(x, y, s=2, alpha=0.10, color=COLORS[model], rasterized=True)
            ax.plot([lo, hi], [lo, hi], "--", color="#333333", lw=0.8)
            slope = linregress(x, y).slope
            ax.text(0.04, 0.96, f"R²={r2_score(x, y):.3f}\nslope={slope:.3f}", transform=ax.transAxes, va="top", fontsize=8, bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"})
            ax.set(xlim=(lo, hi), ylim=(lo, hi), aspect="equal", title=f"{MODEL_LABELS[model]} | {SPLIT_LABELS[split]}")
            if ri == 2:
                ax.set_xlabel(xlabel, fontsize=8)
            if ci == 0:
                ax.set_ylabel(ylabel, fontsize=8)
    fig.suptitle(f"{target} {level.replace('_', '-')} parity | seed 42, single seed", y=0.995, fontweight="bold")
    fig.tight_layout()
    save(fig, filename, f"3×3 {level} parity grid for {target}; each panel pools matched test samples from all folds and annotates pooled R² and slope.")


def lomo_breakdown(gm: pd.DataFrame, target: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    sub = gm[(gm.target == target) & (gm.split == "monomer_heldout")]
    width = 0.22
    ymax = 1.15
    ymin = -0.5
    for mi, model in enumerate(MODELS):
        rows = sub[sub.model == model].sort_values("fold")
        vals = rows.gm_r2.to_numpy()
        shown = np.clip(vals, ymin, ymax)
        x = np.arange(9) + (mi - 1) * width
        ax.bar(x, shown, width, color=COLORS[model], label=MODEL_LABELS[model])
        for xi, v in zip(x, vals):
            if v < ymin or v > ymax:
                ax.annotate(f"{v:.2f}", (xi, ymin), xytext=(0, -14), textcoords="offset points", ha="center", fontsize=7, color=COLORS[model], rotation=90)
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set(ylim=(ymin, ymax), ylabel="Group-mean R²", xlabel="LOMO fold / held-out monomer", title=f"{target}: LOMO per-fold chemistry baseline | seed 42, single seed")
    ax.set_xticks(range(9))
    ax.set_xticklabels([f"{i}\n{name}" for i, name in enumerate(FOLD_NAMES)], fontsize=7, rotation=0)
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save(fig, f"08_lomo_group_mean_r2_{target}.png", f"LOMO per-fold group-mean R² for {target}; ChemArch EA fold 6 is clipped at -0.50 and annotated -12.87.")


def scorecard(gm: pd.DataFrame, cal: pd.DataFrame, target: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=False, sharey=False)
    for ax, split in zip(axes, SPLITS):
        for model in MODELS:
            chemistry = gm[(gm.model == model) & (gm.target == target) & (gm.split == split)].gm_r2.median()
            architecture = cal[(cal.model == model) & (cal.target == target) & (cal.split == split)].delta_r2.median()
            x = max(chemistry, -0.5)
            y = max(architecture, -0.5)
            ax.scatter(x, y, s=100, color=COLORS[model], edgecolor="white", linewidth=1.2, zorder=3)
            ax.annotate(MODEL_LABELS[model], (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8)
            if chemistry < -0.5 or architecture < -0.5:
                ax.annotate(f"clipped ({chemistry:.2f}, {architecture:.2f})", (x, y), xytext=(4, -16), textcoords="offset points", fontsize=6)
        ax.axhline(0, color="#777777", lw=0.7)
        ax.axvline(0, color="#777777", lw=0.7)
        ax.grid(alpha=0.25)
        ax.set_title(SPLIT_LABELS[split])
        ax.set_xlabel("Median group-mean R²")
    axes[0].set_ylabel("Median architecture ΔR²")
    fig.suptitle(f"{target}: chemistry vs architecture scorecard | seed 42, single seed", fontweight="bold")
    fig.tight_layout()
    save(fig, f"09_scorecard_{target}.png", f"Median fold scorecard for {target}: x=group-mean R² and y=architecture ΔR²; values below -0.50 are clipped and annotated.")


def calibration_lomo(folds: dict[tuple[str, str, str, int], pd.DataFrame]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    for ri, target in enumerate(TARGETS):
        for ci, model in enumerate(MODELS):
            ax = axes[ri, ci]
            full = pd.concat([folds[(model, target, "monomer_heldout", f)] for f in range(9)], ignore_index=True)
            matched = filter_matched_groups(full)
            x, y = matched.delta_true.to_numpy(), matched.delta_pred.to_numpy()
            lo, hi = clipped_limits(x, y)
            slope = linregress(x, y).slope
            ax.scatter(x, y, s=2, alpha=0.08, color=COLORS[model], rasterized=True)
            ax.plot([lo, hi], [lo, hi], "--", color="#333333", lw=0.8)
            ax.text(0.04, 0.96, f"R²={r2_score(x, y):.3f}\nslope={slope:.3f}", transform=ax.transAxes, va="top", fontsize=8, bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"})
            ax.set(xlim=(lo, hi), ylim=(lo, hi), title=f"{target} | {MODEL_LABELS[model]}")
            if ri == 1:
                ax.set_xlabel("True Δy (eV)")
            if ci == 0:
                ax.set_ylabel("Predicted Δŷ (eV)")
    fig.suptitle("LOMO architecture-deviation calibration | seed 42, single seed", fontweight="bold")
    fig.tight_layout()
    save(fig, "10_lomo_calibration_scatter.png", "LOMO architecture-deviation parity by target and model; panels pool all matched test samples across 9 folds.")


def effect_size_plot(folds: dict[tuple[str, str, str, int], pd.DataFrame], target: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, split in zip(axes, SPLITS):
        for model in MODELS:
            full = pd.concat([folds[(model, target, split, f)] for f in range(N_FOLDS[split])], ignore_index=True)
            matched = filter_matched_groups(full)
            effect = np.abs(matched.delta_true.to_numpy())
            error = np.abs(matched.delta_pred.to_numpy() - matched.delta_true.to_numpy())
            edges = np.quantile(effect, np.linspace(0, 1, 7))
            edges = np.unique(edges)
            if len(edges) < 3:
                continue
            inds = np.digitize(effect, edges[1:-1], right=True)
            centers = [np.median(effect[inds == i]) for i in range(len(edges) - 1)]
            medians = [np.median(error[inds == i]) for i in range(len(edges) - 1)]
            ax.plot(centers, medians, marker="o", ms=4, color=COLORS[model], label=MODEL_LABELS[model])
        ax.set(title=SPLIT_LABELS[split], xlabel="|True architecture effect| (eV)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Median |Δ prediction error| (eV)")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{target}: error vs architecture-effect size | seed 42, single seed", fontweight="bold")
    fig.tight_layout()
    save(fig, f"11_effect_size_error_{target}.png", f"Per-model median absolute architecture-deviation error in six equal-count |Δy| bins for {target}.")


def error_split_plot(ed: pd.DataFrame, target: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(SPLITS))
    width = 0.22
    for mi, model in enumerate(MODELS):
        sub = ed[(ed.model == model) & (ed.target == target)]
        between = [sub[sub.split == split].frac_SSE_between.median() for split in SPLITS]
        within = [sub[sub.split == split].frac_SSE_within.median() for split in SPLITS]
        xpos = x + (mi - 1) * width
        ax.bar(xpos, between, width, color=COLORS[model], label=f"{MODEL_LABELS[model]} between")
        ax.bar(xpos, within, width, bottom=between, color=COLORS[model], alpha=0.35, hatch="//", label=f"{MODEL_LABELS[model]} within")
    ax.set(xticks=x, xticklabels=[SPLIT_LABELS[s] for s in SPLITS], ylim=(0, 1), ylabel="Median fraction of squared error", title=f"{target}: between- vs within-group error | seed 42, single seed")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save(fig, f"12_error_decomposition_{target}.png", f"Stacked median fold fractions of squared error attributable to group-mean (between) and architecture (within) components for {target}.")


def table_for_readme(frame: pd.DataFrame, metric: str, label: str) -> str:
    rows = [f"### {label}", "", "| Target | Model | Split | Fold values | Median | Mean |", "|---|---|---|---|---:|---:|"]
    for target in TARGETS:
        for model in MODELS:
            for split in SPLITS:
                values = frame[(frame.target == target) & (frame.model == model) & (frame.split == split)][metric].dropna().to_numpy()
                rows.append(f"| {target} | {MODEL_LABELS[model]} | {SPLIT_LABELS[split]} | {', '.join(f'{v:.4f}' for v in values)} | {np.median(values):.4f} | {np.mean(values):.4f} |")
    return "\n".join(rows)


def write_readme(gm: pd.DataFrame, cal: pd.DataFrame, ordering: pd.DataFrame, overall: pd.DataFrame, ed: pd.DataFrame) -> None:
    anomalies = ["All 114 prediction cells were present and valid according to `seed_42/01_validation/evaluation_inventory.csv`.", "ChemArch LOMO EA fold 6 has group-mean R² = -12.8726 and is clipped/annotated in robust-scale figures.", "ChemArch LOMO IP fold 5 has group-mean R² = -0.3493.", "ChemArch LOMO EA fold 4 has architecture ΔR² = -0.0163; fold 6 has architecture ΔR² = -2.4912."]
    parts = ["# Seed-42 Diagnostics Figure Index", "", "All figures use existing seed-42 prediction NPZs and diagnostics CSVs only; no models were trained. Bars are fold medians and dots are individual folds. LOMO bar annotations show fold means. Colors: HPG-hier `#E8A33D`, wDMPNN `#12314E`, ChemArch `#1C7293`.", "", "## Figures", ""]
    parts.extend([f"- `{name}` — {description}" for name, description in FIGURES])
    parts.extend(["", "## Anomalies", ""] + [f"- {a}" for a in anomalies])
    parts.extend(["", table_for_readme(gm, "gm_r2", "Numbers: group-mean R²"), "", table_for_readme(cal, "delta_r2", "Numbers: architecture ΔR²"), "", table_for_readme(ordering, "pairwise_acc", "Numbers: pairwise ordering accuracy"), "", table_for_readme(overall, "overall_r2", "Numbers: overall R²"), "", table_for_readme(overall, "overall_mae", "Numbers: overall MAE (eV)"), "", table_for_readme(overall, "overall_rmse", "Numbers: overall RMSE (eV)"), "", table_for_readme(cal, "delta_slope", "Numbers: calibration slope"), "", table_for_readme(ed, "frac_SSE_between", "Numbers: between-group squared-error fraction")])
    (OUT / "README.md").write_text("\n".join(parts) + "\n")


def main() -> None:
    clean_output()
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    gm = pd.read_csv(DIAG / "03_group_mean_prediction" / "group_mean_metrics.csv")
    cal = pd.read_csv(DIAG / "04_architecture_calibration" / "calibration_metrics.csv")
    ordering = pd.read_csv(DIAG / "05_architecture_ordering" / "ordering_metrics.csv")
    ed = pd.read_csv(DIAG / "02_variance_geometry" / "model_error_decomposition.csv")
    df, meta = load_dataset(), load_all_meta()
    folds = load_fold_data(df, meta)
    overall = overall_metrics(folds)
    grouped_metric_plot(gm, "gm_r2", "Group-mean R² (chemistry baseline)", "Group-mean R²", "01_group_mean_r2.png", clamp=(-0.5, 1.05))
    grouped_metric_plot(cal, "delta_r2", "Architecture ΔR²", "Architecture ΔR²", "02_architecture_delta_r2.png", clamp=(-0.5, 1.05))
    grouped_metric_plot(ordering, "pairwise_acc", "Architecture ordering accuracy", "Pairwise ordering accuracy", "03_ordering_accuracy.png", clamp=(0.0, 1.05))
    grouped_metric_plot(overall, "overall_r2", "Overall R²", "Overall R²", "04_overall_r2.png", clamp=(-0.5, 1.05))
    grouped_metric_plot(overall, "overall_mae", "Overall MAE", "MAE (eV)", "05_overall_mae.png")
    grouped_metric_plot(overall, "overall_rmse", "Overall RMSE", "RMSE (eV)", "06_overall_rmse.png")
    grouped_metric_plot(cal, "delta_slope", "Calibration slope", "Calibration slope", "07_calibration_slope.png", reference=1.0, clamp=(0.0, 1.15))
    for target in TARGETS:
        parity_grid(folds, "overall", target, f"02a_overall_parity_{target}.png")
        parity_grid(folds, "group_mean", target, f"02b_group_mean_parity_{target}.png")
        parity_grid(folds, "architecture_deviation", target, f"02c_architecture_deviation_parity_{target}.png")
        lomo_breakdown(gm, target)
        scorecard(gm, cal, target)
        effect_size_plot(folds, target)
        error_split_plot(ed, target)
    calibration_lomo(folds)
    write_readme(gm, cal, ordering, overall, ed)
    print(f"Wrote {len(FIGURES)} PNG figures to {OUT}")
    print(f"Index: {OUT / 'README.md'}")


if __name__ == "__main__":
    main()
