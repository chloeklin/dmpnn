#!/usr/bin/env python3
"""Figures for the R1 (A-heldout) and R3 (B-heldout clustered) regeneration results.

Reads the CSVs written by scripts/python/analyze_regen_v1.py and
scripts/python/analyze_regen_v1_r3.py. Every figure is skipped with a message if
its input is missing, so this can be run repeatedly while results land.

  fig_r1_architecture       ΔR² and ordering by model, seed SD as error bars
  fig_r1_chemistry_floor    per-fold group-mean R² against its null floor
  fig_r1_paired             paired per-fold differences vs HPG-hier, by metric
  fig_r1_seed_sd            per-fold seed SD — which folds are unstable
  fig_r1_checkpoint_gap     final-minus-best MAE per model family
  fig_r3_architecture       as above, split into fold groups S and D
  fig_r3_chemistry_floor    per-fold group-mean R² against the B-blind floor
  fig_ab_comparison         A-heldout vs B-heldout side by side, per model
  fig_r3_novelty            performance vs held-out-monomer novelty (needs RDKit)
  fig_r{1,3}_error_absolute MAE and RMSE in eV per fold, against the null
  fig_r{1,3}_skill_vs_null  skill score 1 - MSE_model/MSE_null, per fold
  fig_r1_overall_performance      plain overall R2 / RMSE / MAE per model (A split)
  fig_r3_overall_performance      the same on the B split, split by fold group S / D
  *_mean                    mean-across-folds counterparts of the summary figures

Usage:
    python "29-07-2026 supervisor_update/figures_results.py"
    python "29-07-2026 supervisor_update/figures_results.py" --only fig_r1_architecture
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from figstyle import (BERRY, TEAL, ROSE, INK, MUTED, CREAM, GREY, RULE, PANEL,
                      MODEL_COLORS, MODEL_MARKERS, MODEL_HATCHES, MODEL_LABELS,
                      METRIC_LABELS, apply_style, save, note)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DIAG = ROOT / "analysis" / "model_diagnostics"
OUT = HERE / "figures"

R1_CELLS = DIAG / "_regen_v1_results_cells.csv"
R1_COMP = DIAG / "_regen_v1_results_comparisons.csv"
R1_RUNS = DIAG / "_regen_v1_results_individual_runs.csv"
R3_CELLS = DIAG / "_regen_v1_r3_results_cells.csv"
R3_COMP = DIAG / "_regen_v1_r3_results_comparisons.csv"
R3_FOLDS = DIAG / "_regen_v1_r3_results_fold_composition.csv"

MODEL_ORDER = ["hpg_hier", "wdmpnn", "hpg_hier_octamer",
               "hpg_hier_junction", "hpg_hier_junction1"]


# --------------------------------------------------------------------------- #
def load(path: Path, label: str) -> pd.DataFrame | None:
    if not path.is_file():
        print(f"  skip: {label} missing ({path.name})")
        return None
    frame = pd.read_csv(path)
    if frame.empty:
        print(f"  skip: {label} is empty ({path.name})")
        return None
    return frame


def models_in(frame: pd.DataFrame) -> list[str]:
    return [m for m in MODEL_ORDER if m in set(frame.model)]


def label_of(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def _agg(series, stat):
    return series.mean() if stat == "mean" else series.median()


def _grouped_metric_axes(ax, cells, metric, models, targets, title, ylabel, stat="median"):
    """Bars = model, groups = target, error bars = seed SD across 3 seeds.

    stat="median" is robust to the pathological folds; stat="mean" is what a reader
    expects by default and is sensitive to them. Both are reported — a large gap
    between them is itself informative about fold heterogeneity.
    """
    width = 0.8 / max(1, len(models))
    for j, model in enumerate(models):
        vals, errs = [], []
        for tgt in targets:
            row = cells[(cells.model == model) & (cells.target == tgt)]
            vals.append(_agg(row[metric], stat) if not row.empty else np.nan)
            sd_col = f"{metric}_seed_sd"
            errs.append(_agg(row[sd_col], stat) if sd_col in row.columns and not row.empty else 0.0)
        pos = np.arange(len(targets)) + (j - (len(models) - 1) / 2) * width
        ax.bar(pos, vals, width * 0.9, label=label_of(model),
               color=MODEL_COLORS.get(model, GREY),
               hatch=MODEL_HATCHES.get(model, ""), edgecolor="white",
               linewidth=0.6, zorder=3)
        ax.errorbar(pos, vals, yerr=errs, fmt="none", ecolor=INK,
                    elinewidth=1.0, capsize=3, zorder=4)
    ax.set_xticks(np.arange(len(targets)))
    ax.set_xticklabels(targets)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


# --------------------------------------------------------------------------- #
def fig_r1_architecture(stat="median"):
    """The headline: architecture recovery by model, with seed error bars."""
    cells = load(R1_CELLS, "R1 cells")
    if cells is None:
        return
    models, targets = models_in(cells), sorted(cells.target.unique())
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for ax, metric in zip(axes, ["delta_r2", "ordering"]):
        _grouped_metric_axes(ax, cells, metric, models, targets,
                             METRIC_LABELS.get(metric, metric),
                             f"{stat} across folds", stat=stat)
    axes[0].legend(ncol=len(models), loc="lower center",
                   bbox_to_anchor=(1.05, -0.30), fontsize=8.5)
    note(axes[0], f"A-heldout · 3-seed averaged predictions · bars are the {stat} across 9 folds · "
                  f"error bars = {stat} per-fold seed SD", dy=-0.40)
    save(fig, OUT, "fig_r1_architecture" + ("_mean" if stat == "mean" else ""))


def fig_r1_chemistry_floor():
    """Per-fold group-mean R² against its fold-specific null floor."""
    cells = load(R1_CELLS, "R1 cells")
    if cells is None:
        return
    if "null_group_mean_r2" not in cells.columns:
        print("  skip fig_r1_chemistry_floor: no null_group_mean_r2 column")
        return
    targets = sorted(cells.target.unique())
    models = models_in(cells)
    fig, axes = plt.subplots(1, len(targets), figsize=(6.0 * len(targets), 4.3),
                             squeeze=False)
    for ax, tgt in zip(axes[0], targets):
        sub = cells[cells.target == tgt]
        folds = sorted(sub.fold.unique())
        floor = [sub[sub.fold == f].null_group_mean_r2.iloc[0] for f in folds]
        ax.fill_between(folds, -1, floor, color=CREAM, zorder=1,
                        label="below the null floor")
        ax.plot(folds, floor, color=GREY, lw=1.6, ls="--", zorder=5,
                label="null floor (knows nothing of the held-out monomer)")
        for model in models:
            ms = sub[sub.model == model].sort_values("fold")
            ax.plot(ms.fold, ms.group_mean_r2, marker=MODEL_MARKERS.get(model, "o"),
                    color=MODEL_COLORS.get(model, GREY), label=label_of(model), zorder=6)
        ax.set_xticks(folds)
        ax.set_xlabel("fold  (held-out monomer)")
        ax.set_ylabel("group-mean R²")
        ax.set_ylim(min(-0.1, min(floor) - 0.1), 1.05)
        ax.set_title(f"{tgt} — chemistry placement vs its floor")
    axes[0][0].legend(loc="lower left", fontsize=8)
    note(axes[0][0], "A cell below the dashed line is beaten by a predictor that "
                     "ignores the held-out monomer entirely.", dy=-0.20)
    save(fig, OUT, "fig_r1_chemistry_floor")


def fig_r1_paired():
    """Paired per-fold differences vs HPG-hier — forest style, with the noise band."""
    comp = load(R1_COMP, "R1 comparisons")
    if comp is None:
        return
    metrics = [m for m in ["group_mean_r2", "delta_r2", "ordering", "mae"]
               if m in set(comp.metric)]
    targets = sorted(comp.target.unique())
    fig, axes = plt.subplots(1, len(targets), figsize=(6.4 * len(targets), 4.6),
                             squeeze=False, sharex=True)
    for ax, tgt in zip(axes[0], targets):
        rows, labels, colors = [], [], []
        for metric in metrics:
            sub = comp[(comp.target == tgt) & (comp.metric == metric)]
            for _, r in sub.iterrows():
                rows.append(r)
                labels.append(f"{label_of(r.model)}  ·  {metric}")
                colors.append(MODEL_COLORS.get(r.model, GREY))
        y = np.arange(len(rows))[::-1]
        ax.axvline(0, color=INK, lw=1.0, zorder=4)
        for yi, r, c in zip(y, rows, colors):
            sig = r.get("holm_p", 1.0) < 0.05
            ax.plot([r.median_paired_difference], [yi], marker="D" if sig else "o",
                    ms=7 if sig else 6, color=c, zorder=6)
            ax.annotate(f"{int(r.wins)}W/{int(r.losses)}L  p={r.exact_sign_p:.3f}",
                        xy=(r.median_paired_difference, yi), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=7.5,
                        color=MUTED)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("median paired per-fold difference vs HPG-hier")
        ax.set_title(f"{tgt}")
        ax.grid(axis="y", visible=False)
    note(axes[0][0], "Positive = candidate better (MAE sign already flipped). "
                     "Diamonds are Holm-corrected p < 0.05. Minimum attainable "
                     "two-sided p with 9 folds is 0.0039.", dy=-0.16)
    save(fig, OUT, "fig_r1_paired")


def fig_r1_seed_sd():
    """Which folds are unstable — seed SD per fold, per model."""
    cells = load(R1_CELLS, "R1 cells")
    if cells is None:
        return
    metric_sd = "group_mean_r2_seed_sd"
    if metric_sd not in cells.columns:
        print(f"  skip fig_r1_seed_sd: no {metric_sd} column")
        return
    targets = sorted(cells.target.unique())
    models = models_in(cells)
    fig, axes = plt.subplots(1, len(targets), figsize=(6.2 * len(targets), 4.0),
                             squeeze=False)
    for ax, tgt in zip(axes[0], targets):
        sub = cells[cells.target == tgt]
        folds = sorted(sub.fold.unique())
        width = 0.8 / max(1, len(models))
        for j, model in enumerate(models):
            ms = sub[sub.model == model].set_index("fold").reindex(folds)
            pos = np.arange(len(folds)) + (j - (len(models) - 1) / 2) * width
            ax.bar(pos, ms[metric_sd].values, width * 0.9,
                   color=MODEL_COLORS.get(model, GREY), label=label_of(model), zorder=3)
        ax.set_xticks(np.arange(len(folds)))
        ax.set_xticklabels(folds)
        ax.set_xlabel("fold")
        ax.set_ylabel("SD of group-mean R² across 3 seeds")
        ax.set_title(f"{tgt} — per-fold instability")
    axes[0][0].legend(fontsize=8, ncol=2)
    note(axes[0][0], "Tests whether the historic \"pathological folds\" (EA 1, EA 6, "
                     "IP 5, IP 2) are simply the high-variance ones.", dy=-0.20)
    save(fig, OUT, "fig_r1_seed_sd")


def fig_r1_checkpoint_gap():
    """What the model-selection bug cost, per model family."""
    runs = load(R1_RUNS, "R1 individual runs")
    if runs is None or "final_minus_best_mae" not in runs.columns:
        print("  skip fig_r1_checkpoint_gap: no final_minus_best_mae column")
        return
    models = models_in(runs)
    data = [runs[runs.model == m].final_minus_best_mae.dropna().values for m in models]
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=True,
                    medianprops=dict(color=INK, lw=1.6),
                    flierprops=dict(marker="o", ms=3, mfc=MUTED, mec="none"))
    for patch, model in zip(bp["boxes"], models):
        patch.set_facecolor(MODEL_COLORS.get(model, GREY))
        patch.set_alpha(0.85)
        patch.set_edgecolor("none")
    ax.axhline(0, color=INK, lw=1.0, zorder=1)
    ax.set_xticklabels([label_of(m) for m in models], fontsize=9)
    ax.set_ylabel("final-model MAE − best-checkpoint MAE  (eV)")
    ax.set_title("Cost of the model-selection bug, by model family")
    means = "   ".join(f"{label_of(m)}: {np.mean(d):+.4f}" for m, d in zip(models, data) if len(d))
    note(ax, f"Positive = the bug would have made this model look worse.  means: {means}",
         dy=-0.18)
    save(fig, OUT, "fig_r1_checkpoint_gap")


# --------------------------------------------------------------------------- #
def _fold_groups() -> dict[int, str] | None:
    comp = load(R3_FOLDS, "R3 fold composition")
    if comp is None or "fold_group" not in comp.columns:
        return None
    return dict(zip(comp.fold.astype(int), comp.fold_group))


def fig_r3_architecture(stat="median"):
    """R3 architecture recovery, reported separately for the two fold groups."""
    cells = load(R3_CELLS, "R3 cells")
    groups = _fold_groups()
    if cells is None or groups is None:
        return
    cells = cells.copy()
    cells["fold_group"] = cells.fold.map(groups)
    models, targets = models_in(cells), sorted(cells.target.unique())
    order = ["S_within_scaffold", "D_cross_scaffold"]
    present = [g for g in order if g in set(cells.fold_group)]
    titles = {"S_within_scaffold": "S — new substituents, familiar core",
              "D_cross_scaffold": "D — new scaffolds"}

    fig, axes = plt.subplots(1, len(present), figsize=(6.2 * len(present), 4.2),
                             squeeze=False)
    for ax, grp in zip(axes[0], present):
        _grouped_metric_axes(ax, cells[cells.fold_group == grp], "delta_r2",
                             models, targets, titles.get(grp, grp),
                             f"ΔR²  ({stat} across folds in group)", stat=stat)
    axes[0][0].legend(ncol=2, fontsize=8.5, loc="lower left")
    note(axes[0][0], "B-heldout clustered. The two groups are not exchangeable, so they "
                     "are never pooled into one number.", dy=-0.20)
    save(fig, OUT, "fig_r3_architecture" + ("_mean" if stat == "mean" else ""))


def fig_r3_chemistry_floor():
    """R3 group-mean R² against the fold-specific B-blind floor."""
    cells = load(R3_CELLS, "R3 cells")
    if cells is None or "null_group_mean_r2" not in cells.columns:
        return
    groups = _fold_groups() or {}
    targets = sorted(cells.target.unique())
    models = models_in(cells)
    fig, axes = plt.subplots(1, len(targets), figsize=(6.2 * len(targets), 4.4),
                             squeeze=False)
    for ax, tgt in zip(axes[0], targets):
        sub = cells[cells.target == tgt]
        folds = sorted(sub.fold.unique())
        floor = [sub[sub.fold == f].null_group_mean_r2.iloc[0] for f in folds]
        ax.plot(folds, floor, color=GREY, lw=1.6, ls="--", zorder=5,
                label="B-blind null floor")
        for model in models:
            ms = sub[sub.model == model].sort_values("fold")
            ax.plot(ms.fold, ms.group_mean_r2, marker=MODEL_MARKERS.get(model, "o"),
                    color=MODEL_COLORS.get(model, GREY), label=label_of(model), zorder=6)
        for f in folds:
            if groups.get(f) == "S_within_scaffold":
                ax.axvspan(f - 0.45, f + 0.45, color=CREAM, zorder=0)
        ax.set_xticks(folds)
        ax.set_xlabel("fold  (shaded = within-scaffold group S)")
        ax.set_ylabel("group-mean R²")
        ax.set_ylim(min(-0.1, min(floor) - 0.1), 1.05)
        ax.set_title(f"{tgt} — B-heldout chemistry vs floor")
    axes[0][0].legend(loc="lower left", fontsize=8)
    note(axes[0][0], "The B-blind floor varies strongly by fold, so raw R² is not "
                     "comparable between folds — read the gap, not the level.", dy=-0.20)
    save(fig, OUT, "fig_r3_chemistry_floor")


def fig_ab_comparison(stat="median"):
    """Does the architecture advantage hold on both chemical axes?

    Three bars per model, never two: the B-heldout folds are NOT exchangeable, so
    they are shown as their two groups rather than pooled into a single number.
      A       - A-heldout, 9 folds, one held-out donor each
      B (S)   - B-heldout group S, new side chains on cores already in training
      B (D)   - B-heldout group D, whole new scaffold families
    """
    r1, r3 = load(R1_CELLS, "R1 cells"), load(R3_CELLS, "R3 cells")
    if r1 is None or r3 is None:
        return
    groups = _fold_groups()
    if groups is None:
        print("  skip fig_ab_comparison: fold composition unavailable, refusing to pool B folds")
        return
    r3 = r3.copy()
    r3["fold_group"] = r3.fold.map(groups)

    panels = [
        ("A-heldout", r1, ""),
        ("B-heldout  (S)", r3[r3.fold_group == "S_within_scaffold"], "..."),
        ("B-heldout  (D)", r3[r3.fold_group == "D_cross_scaffold"], "///"),
    ]
    targets = sorted(set(r1.target) & set(r3.target))
    models = [m for m in MODEL_ORDER if m in set(r1.model) & set(r3.model)]
    alphas = [1.0, 0.72, 0.45]

    fig, axes = plt.subplots(1, len(targets), figsize=(6.6 * len(targets), 4.4),
                             squeeze=False)
    for ax, tgt in zip(axes[0], targets):
        width = 0.8 / len(panels)
        for j, (name, frame, hatch) in enumerate(panels):
            vals, errs = [], []
            for model in models:
                sub = frame[(frame.model == model) & (frame.target == tgt)]
                vals.append(_agg(sub.delta_r2, stat) if not sub.empty else np.nan)
                col = "delta_r2_seed_sd"
                errs.append(_agg(sub[col], stat) if col in sub.columns and not sub.empty else 0.0)
            pos = np.arange(len(models)) + (j - (len(panels) - 1) / 2) * width
            ax.bar(pos, vals, width * 0.88, hatch=hatch, alpha=alphas[j],
                   color=[MODEL_COLORS.get(m, GREY) for m in models],
                   edgecolor="white", linewidth=0.6, zorder=3)
            ax.errorbar(pos, vals, yerr=errs, fmt="none", ecolor=INK,
                        elinewidth=1.0, capsize=2.5, zorder=4)
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels([label_of(m) for m in models], fontsize=8.5, rotation=15,
                           ha="right")
        ax.set_ylabel(f"ΔR²  ({stat} across folds)")
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{tgt} — architecture recovery")
        ax.legend(handles=[Line2D([0], [0], color=MUTED, lw=8, alpha=a, label=n)
                           for (n, _, _), a in zip(panels, alphas)],
                  fontsize=8, loc="lower left", ncol=1)
    note(axes[0][0], "The B-heldout folds are split into their two groups rather than pooled — "
                     "they are not exchangeable. Absolute scores are not directly comparable "
                     "across splits; read the ordering of models within each group of bars.",
         dy=-0.30)
    save(fig, OUT, "fig_ab_comparison" + ("_mean" if stat == "mean" else ""))


def fig_r3_novelty(bins=(0.0, 0.35, 0.45, 0.55, 0.70, 1.01)):
    """Performance versus how unfamiliar the held-out monomer is.

    Computed from the frozen clustered split plus the regenerated NPZs, so it
    needs RDKit and the prediction files. This is the analysis the A-heldout
    split cannot support (one held-out monomer per fold).
    """
    split_path = ROOT / "metadata" / "splits" / "monomer_b_heldout_clustered.json"
    pred_dir = ROOT / "predictions" / "regen_v1" / "ea_ip_lomo_b_clustered"
    data_csv = ROOT / "data" / "ea_ip.csv"
    if not (split_path.is_file() and pred_dir.is_dir() and data_csv.is_file()):
        print("  skip fig_r3_novelty: split metadata or predictions not found")
        return
    try:
        from rdkit import Chem, DataStructs, RDLogger
        from rdkit.Chem import AllChem
    except ImportError:
        print("  skip fig_r3_novelty: RDKit not available")
        return
    RDLogger.DisableLog("rdApp.*")

    folds = json.loads(split_path.read_text())["folds"]
    df = pd.read_csv(data_csv)
    b_col = df.smiles_B.astype(str).to_numpy()

    all_b = sorted({s for f in folds for s in f["held_out_monomer_B"]})
    fps = {}
    for smi in all_b:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps[smi] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

    # max similarity of each held-out monomer to that fold's training monomers
    novelty: dict[tuple[int, str], float] = {}
    for f in folds:
        train_fps = [fps[s] for s in f["train_monomer_B"] if s in fps]
        for smi in f["held_out_monomer_B"]:
            if smi in fps and train_fps:
                novelty[(int(f["fold"]), smi)] = max(
                    DataStructs.BulkTanimotoSimilarity(fps[smi], train_fps))

    records = []
    for npz_path in sorted(pred_dir.glob("*.npz")):
        stem = npz_path.stem.split("__")
        if len(stem) < 6:
            continue
        target, model = stem[1], stem[2]
        fold = int(stem[4].replace("fold", ""))
        with np.load(npz_path, allow_pickle=True) as arch:
            if "y_pred" not in arch.files:
                continue
            y_true = arch["y_true"].astype(float).ravel()
            y_pred = arch["y_pred"].astype(float).ravel()
            idx = arch["test_indices"].astype(int).ravel()
        sims = np.array([novelty.get((fold, b), np.nan) for b in b_col[idx]])
        records.append(pd.DataFrame({"model": model, "target": target, "fold": fold,
                                     "abs_err": np.abs(y_pred - y_true), "sim": sims}))
    if not records:
        print("  skip fig_r3_novelty: no usable prediction files")
        return
    rows = pd.concat(records, ignore_index=True).dropna(subset=["sim"])
    rows["bin"] = pd.cut(rows.sim, bins=list(bins), right=False)

    targets = sorted(rows.target.unique())
    models = models_in(rows)
    fig, axes = plt.subplots(1, len(targets), figsize=(6.2 * len(targets), 4.2),
                             squeeze=False)
    for ax, tgt in zip(axes[0], targets):
        sub = rows[rows.target == tgt]
        cats = [c for c in sub.bin.cat.categories if not sub[sub.bin == c].empty]
        xs = np.arange(len(cats))
        for model in models:
            ms = sub[sub.model == model]
            means = [ms[ms.bin == c].abs_err.mean() for c in cats]
            ax.plot(xs, means, marker=MODEL_MARKERS.get(model, "o"),
                    color=MODEL_COLORS.get(model, GREY), label=label_of(model))
        counts = [sub[sub.bin == c].abs_err.size for c in cats]
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{c.left:.2f}–{c.right:.2f}\nn={n:,}"
                            for c, n in zip(cats, counts)], fontsize=8)
        ax.set_xlabel("max Tanimoto of held-out B monomer to training  (left = more novel)")
        ax.set_ylabel("mean absolute error  (eV)")
        ax.set_title(f"{tgt} — error versus novelty")
    axes[0][0].legend(fontsize=8)
    note(axes[0][0], "Morgan r=2, 2048-bit. This analysis is impossible on the A-heldout "
                     "split, which holds out one monomer per fold.", dy=-0.30)
    save(fig, OUT, "fig_r3_novelty")



def _error_panel(ax, cells, tgt, metric, null_col, models, unit="eV"):
    sub = cells[cells.target == tgt]
    folds = sorted(sub.fold.unique())
    if null_col in sub.columns:
        floor = [sub[sub.fold == f][null_col].iloc[0] for f in folds]
        ax.plot(folds, floor, color=GREY, lw=1.6, ls="--", zorder=5,
                label="null (ignores held-out monomer)")
    for model in models:
        ms = sub[sub.model == model].sort_values("fold")
        ax.plot(ms.fold, ms[metric], marker=MODEL_MARKERS.get(model, "o"),
                color=MODEL_COLORS.get(model, GREY), label=label_of(model), zorder=6)
    ax.set_xticks(folds)
    ax.set_xlabel("fold")
    ax.set_ylabel(f"{metric.upper()}  ({unit})")
    ax.set_ylim(bottom=0)
    ax.set_title(f"{tgt} — {metric.upper()}, absolute error")


def fig_error_absolute(split="r1"):
    """MAE and RMSE in eV — the only chemically interpretable numbers, against the null.

    R2-family metrics are normalised by each fold's own target variance and are therefore
    not comparable between folds. These are.
    """
    path = R1_CELLS if split == "r1" else R3_CELLS
    cells = load(path, f"{split.upper()} cells")
    if cells is None:
        return
    have = [m for m in ("mae", "rmse") if m in cells.columns]
    if not have:
        print(f"  skip fig_{split}_error_absolute: no mae/rmse columns — re-run the analysis")
        return
    targets = sorted(cells.target.unique())
    models = models_in(cells)
    fig, axes = plt.subplots(len(have), len(targets),
                             figsize=(5.8 * len(targets), 3.6 * len(have)), squeeze=False)
    for r, metric in enumerate(have):
        null_col = {"mae": "null_mae", "rmse": "null_rmse"}[metric]
        for c, tgt in enumerate(targets):
            _error_panel(axes[r][c], cells, tgt, metric, null_col, models)
    axes[0][0].legend(fontsize=8, loc="upper left")
    missing = "  (null RMSE needs a re-run of audit_b_heldout_design.py)" \
        if "rmse" in have and "null_rmse" not in cells.columns else ""
    note(axes[-1][0], f"Absolute error in eV, comparable across folds and targets.{missing}",
         dy=-0.24)
    save(fig, OUT, f"fig_{split}_error_absolute")


def fig_skill_vs_null(split="r1"):
    """Skill against the null: 1 - MSE_model/MSE_null. 0 = no better than knowing nothing."""
    path = R1_CELLS if split == "r1" else R3_CELLS
    cells = load(path, f"{split.upper()} cells")
    if cells is None:
        return
    have = [c for c in ("skill_group_mean", "skill_overall") if c in cells.columns]
    if not have:
        print(f"  skip fig_{split}_skill: no skill columns — re-run the analysis")
        return
    targets = sorted(cells.target.unique())
    models = models_in(cells)
    fig, axes = plt.subplots(len(have), len(targets),
                             figsize=(5.8 * len(targets), 3.6 * len(have)), squeeze=False)
    titles = {"skill_group_mean": "chemistry placement (group means)",
              "skill_overall": "all test rows"}
    for r, col in enumerate(have):
        for c, tgt in enumerate(targets):
            ax = axes[r][c]
            sub = cells[cells.target == tgt]
            folds = sorted(sub.fold.unique())
            ax.axhline(0, color=INK, lw=1.2, zorder=5)
            ax.axhline(1, color=GREY, lw=0.8, ls=":", zorder=4)
            for model in models:
                ms = sub[sub.model == model].sort_values("fold")
                ax.plot(ms.fold, ms[col], marker=MODEL_MARKERS.get(model, "o"),
                        color=MODEL_COLORS.get(model, GREY), label=label_of(model), zorder=6)
            ax.set_xticks(folds)
            ax.set_xlabel("fold")
            ax.set_ylabel("skill vs null")
            ax.set_ylim(min(-0.25, float(sub[col].min()) - 0.05), 1.05)
            ax.set_title(f"{tgt} — {titles.get(col, col)}")
    axes[0][0].legend(fontsize=8, loc="lower left")
    note(axes[-1][0], "0 = no better than a predictor that ignores the held-out monomer; "
                      "1 = perfect. Scale-free, so unlike R² these are comparable across "
                      "folds and targets.", dy=-0.26)
    save(fig, OUT, f"fig_{split}_skill_vs_null")



def fig_overall_performance(split="r1", stat="median"):
    """Plain bottom-line accuracy per model: overall R2 and RMSE, with seed error bars.

    This is the "how good is each model" figure — no decomposition, no null floors.
    """
    path = R1_CELLS if split == "r1" else R3_CELLS
    cells = load(path, f"{split.upper()} cells")
    if cells is None:
        return
    models, targets = models_in(cells), sorted(cells.target.unique())
    metrics = [m for m in ("overall_r2", "rmse", "mae") if m in cells.columns]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 4.2), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        _grouped_metric_axes(ax, cells, metric, models, targets,
                             METRIC_LABELS.get(metric, metric.upper()),
                             f"{stat} across folds" + ("  (eV)" if metric in ("rmse", "mae") else ""),
                             stat=stat)
        if metric == "overall_r2":
            ax.set_ylim(0, 1.05)
        else:
            ax.set_ylim(bottom=0)
    axes[0][0].legend(ncol=min(3, len(models)), loc="lower center",
                      bbox_to_anchor=(0.5 * len(metrics), -0.32), fontsize=8.5)
    label = "A-heldout" if split == "r1" else "B-heldout clustered"
    note(axes[0][0], f"{label} · 3-seed averaged predictions · bars are the {stat} across folds · "
                     f"error bars = {stat} per-fold seed SD", dy=-0.44)
    save(fig, OUT, f"fig_{split}_overall_performance" + ("_mean" if stat == "mean" else ""))



def fig_r3_overall_performance(stat="median"):
    """Bottom-line accuracy on the B split — R2, RMSE and MAE, split by fold group.

    The nine B folds are not exchangeable, so this never pools them: one row per
    fold group. Group S holds out new side chains on cores already in training;
    group D holds out whole new scaffold families.
    """
    cells = load(R3_CELLS, "R3 cells")
    groups = _fold_groups()
    if cells is None or groups is None:
        if cells is not None:
            print("  skip fig_r3_overall_performance: fold composition unavailable, "
                  "refusing to pool B folds")
        return
    cells = cells.copy()
    cells["fold_group"] = cells.fold.map(groups)

    metrics = [m for m in ("overall_r2", "rmse", "mae") if m in cells.columns]
    rows = [("S_within_scaffold", "S — new side chains, familiar core"),
            ("D_cross_scaffold", "D — new scaffold families")]
    rows = [(k, lab) for k, lab in rows if k in set(cells.fold_group)]
    models, targets = models_in(cells), sorted(cells.target.unique())

    fig, axes = plt.subplots(len(rows), len(metrics),
                             figsize=(4.5 * len(metrics), 3.7 * len(rows)),
                             squeeze=False)
    err_max = max([cells[m].max() for m in metrics if m in ("rmse", "mae")] or [1])
    for r, (key, label) in enumerate(rows):
        sub = cells[cells.fold_group == key]
        for c, metric in enumerate(metrics):
            ax = axes[r][c]
            _grouped_metric_axes(ax, sub, metric, models, targets,
                                 f"{label}" if c == 0 else METRIC_LABELS.get(metric, metric.upper()),
                                 f"{METRIC_LABELS.get(metric, metric.upper())}"
                                 + ("  (eV)" if metric in ("rmse", "mae") else ""),
                                 stat=stat)
            # shared scales down each column so the two groups are directly comparable
            if metric == "overall_r2":
                ax.set_ylim(0, 1.05)
            else:
                ax.set_ylim(0, err_max * 1.15)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(models), loc="lower center",
               bbox_to_anchor=(0.5, -0.02), fontsize=9, frameon=False)
    fig.text(0.5, -0.06,
             f"B-heldout clustered · 3-seed averaged predictions · bars are the {stat} across the "
             f"folds in each group · error bars = {stat} per-fold seed SD · y-axes shared down each "
             f"column, so the two rows are directly comparable.",
             ha="center", fontsize=8, color=MUTED, style="italic")
    fig.subplots_adjust(hspace=0.42)
    save(fig, OUT, "fig_r3_overall_performance" + ("_mean" if stat == "mean" else ""))


FIGURES = {
    "fig_r1_architecture": fig_r1_architecture,
    "fig_r1_chemistry_floor": fig_r1_chemistry_floor,
    "fig_r1_paired": fig_r1_paired,
    "fig_r1_seed_sd": fig_r1_seed_sd,
    "fig_r1_checkpoint_gap": fig_r1_checkpoint_gap,
    "fig_r3_architecture": fig_r3_architecture,
    "fig_r3_chemistry_floor": fig_r3_chemistry_floor,
    "fig_ab_comparison": fig_ab_comparison,
    "fig_r3_novelty": fig_r3_novelty,
    "fig_r1_error_absolute": lambda: fig_error_absolute("r1"),
    "fig_r3_error_absolute": lambda: fig_error_absolute("r3"),
    "fig_r1_overall_performance": lambda: fig_overall_performance("r1"),
    "fig_r3_overall_performance": fig_r3_overall_performance,
    "fig_r1_skill_vs_null": lambda: fig_skill_vs_null("r1"),
    "fig_r3_skill_vs_null": lambda: fig_skill_vs_null("r3"),
    # mean-across-folds counterparts — medians are robust to the pathological folds,
    # means are what a reader assumes; a gap between them signals fold heterogeneity
    "fig_r1_architecture_mean": lambda: fig_r1_architecture("mean"),
    "fig_r3_architecture_mean": lambda: fig_r3_architecture("mean"),
    "fig_ab_comparison_mean": lambda: fig_ab_comparison("mean"),
    "fig_r1_overall_performance_mean": lambda: fig_overall_performance("r1", "mean"),
    "fig_r3_overall_performance_mean": lambda: fig_r3_overall_performance("mean"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", choices=sorted(FIGURES))
    args = ap.parse_args()
    apply_style()
    print(f"writing to {OUT}")
    for key in (args.only or list(FIGURES)):
        print(f"[{key}]")
        try:
            FIGURES[key]()
        except Exception as exc:            # keep going as results trickle in
            print(f"  failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
