#!/usr/bin/env python3
"""Regenerate the three data figures used in week_review_monomerB_split.pptx.

Every number is read from its source rather than hard-coded, so re-running this
after the underlying analyses change keeps the figures honest.

    fig1_run_to_run_variance   <- analysis/model_diagnostics/_noise_floor_results.csv
    fig2_variance_by_axis      <- analysis/model_diagnostics/_dataset_design_audit.md  (§0.2)
    fig3_scaffold_cluster_sizes<- data/ea_ip.csv, via RDKit Murcko scaffolds

Usage:
    python "29-07-2026 supervisor_update/figures_deck.py"
    python "29-07-2026 supervisor_update/figures_deck.py" --only fig2
"""

from __future__ import annotations

import argparse
import re
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from figstyle import (BERRY, TEAL, ROSE, INK, MUTED, CREAM, apply_style, save, note)
import matplotlib.pyplot as plt

DIAG = ROOT / "analysis" / "model_diagnostics"
NOISE_CSV = DIAG / "_noise_floor_results.csv"
DESIGN_MD = DIAG / "_dataset_design_audit.md"
DATA_CSV = ROOT / "data" / "ea_ip.csv"
OUT = HERE / "figures"


# --------------------------------------------------------------------------- #
def read_md_table(path: Path, required: set[str], occurrence: int = 0) -> pd.DataFrame:
    """Pull a pipe table out of a markdown file by the columns it must contain."""
    lines = path.read_text().splitlines()
    matches = []
    for i, line in enumerate(lines):
        if not line.startswith("|"):
            continue
        cols = {c.strip() for c in line.strip("|").split("|")}
        if required.issubset(cols):
            block = [line]
            for cand in lines[i + 1:]:
                if not cand.startswith("|"):
                    break
                block.append(cand)
            matches.append(block)
    if len(matches) <= occurrence:
        raise SystemExit(f"table {sorted(required)} not found in {path.name}")
    frame = pd.read_csv(StringIO("\n".join(matches[occurrence])), sep="|",
                        skipinitialspace=True).iloc[:, 1:-1]
    frame.columns = [c.strip() for c in frame.columns]
    frame = frame.apply(lambda c: c.str.strip() if c.dtype == object else c)
    sep = frame.astype(str).apply(lambda c: c.str.fullmatch(r":?-+:?"), axis=0).all(axis=1)
    return frame[~sep].reset_index(drop=True)


# --------------------------------------------------------------------------- #
def fig1_run_to_run_variance():
    """Three identical runs per fold — the reason single-run results were withdrawn."""
    if not NOISE_CSV.is_file():
        print(f"  skip fig1: {NOISE_CSV.name} not found")
        return
    df = pd.read_csv(NOISE_CSV)
    folds = sorted(df.fold.unique())
    repeats = sorted(df.repeat.unique())
    colors = [BERRY, TEAL, ROSE]
    hatches = ['', '///', '...']

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    width = 0.8 / len(repeats)
    for j, rep in enumerate(repeats):
        vals = [df[(df.fold == f) & (df.repeat == rep)].group_mean_r2.iloc[0] for f in folds]
        pos = np.arange(len(folds)) + (j - (len(repeats) - 1) / 2) * width
        bars = ax.bar(pos, vals, width * 0.92, label=f"run {rep}",
                      color=colors[j % len(colors)], hatch=hatches[j % len(hatches)],
                      edgecolor="white", linewidth=0.6, zorder=3)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2, color=INK, weight="bold")

    ax.set_xticks(np.arange(len(folds)))
    ax.set_xticklabels([f"EA fold {f}" for f in folds])
    ax.set_ylabel("group-mean R²")
    ax.set_ylim(0, 1.06)
    ax.set_title("Three identical runs — same model, seed, split, code and GPU")
    ax.legend(ncol=len(repeats), loc="lower center", bbox_to_anchor=(0.5, -0.12))

    sds = df.groupby("fold").group_mean_r2.std(ddof=1)
    caption = "  ·  ".join(f"fold {f}: SD {sds[f]:.3f}" for f in folds)
    note(ax, f"{caption}    (source: {NOISE_CSV.name})", dy=-0.26)
    save(fig, OUT, "fig1_run_to_run_variance")


def fig2_variance_by_axis():
    """Share of target variance explained by each monomer role."""
    if not DESIGN_MD.is_file():
        print(f"  skip fig2: {DESIGN_MD.name} not found")
        return
    tbl = read_md_table(DESIGN_MD, {"target", "A_identity", "B_identity", "within_A_B_fracA"})
    for c in tbl.columns[1:]:
        tbl[c] = pd.to_numeric(tbl[c])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0),
                                  gridspec_kw={"width_ratios": [1.55, 1]})
    targets = tbl.target.tolist()
    width = 0.34
    for j, (col, label, colr) in enumerate([
            ("A_identity", "monomer A identity   (9 monomers)", BERRY),
            ("B_identity", "monomer B identity   (682 monomers)", TEAL)]):
        pos = np.arange(len(targets)) + (j - 0.5) * width
        bars = ax.bar(pos, tbl[col], width * 0.92, label=label, color=colr,
                      hatch=["", "///"][j], edgecolor="white", linewidth=0.6, zorder=3)
        ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=2, color=INK, weight="bold")
    ax.set_xticks(np.arange(len(targets)))
    ax.set_xticklabels(targets)
    ax.set_ylabel("share of total variance")
    ax.set_ylim(0, max(tbl[["A_identity", "B_identity"]].max()) * 1.28)
    ax.set_title("Comparable variance, wildly different monomer counts")
    ax.legend(loc="upper left", fontsize=8.5)

    arch = tbl.set_index("target").within_A_B_fracA * 100
    bars = ax2.bar(arch.index, arch.values, 0.45, color=ROSE, zorder=3)
    ax2.bar_label(bars, fmt="%.2f%%", fontsize=9, padding=2, color=INK, weight="bold")
    ax2.set_ylabel("% of total variance")
    ax2.set_ylim(0, max(arch.values) * 1.5)
    ax2.set_title("Architecture signal, within (A, B, fracA)")
    note(ax, f"source: {DESIGN_MD.name} §0.2")
    save(fig, OUT, "fig2_variance_by_axis")


def fig3_scaffold_cluster_sizes(top_n: int = 10):
    """Murcko cluster sizes among the B monomers — why no disjoint split exists."""
    if not DATA_CSV.is_file():
        print(f"  skip fig3: {DATA_CSV} not found")
        return
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError:
        print("  skip fig3: RDKit not available in this environment")
        return
    RDLogger.DisableLog("rdApp.*")

    b_values = sorted(pd.read_csv(DATA_CSV).smiles_B.astype(str).unique())
    clusters: dict[str, list[str]] = {}
    for smi in b_values:
        mol = Chem.MolFromSmiles(smi)
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else "INVALID"
        clusters.setdefault(scaf or f"ACYCLIC::{smi}", []).append(smi)

    sizes = sorted((len(v) for v in clusters.values()), reverse=True)
    top = sizes[:top_n]
    n_scaf, n_mon = len(clusters), len(b_values)
    singletons = sum(1 for s in sizes if s == 1)
    share = sum(top[:2]) / n_mon

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ordinals = ["1st", "2nd", "3rd"] + [f"{i}th" for i in range(4, top_n + 1)]
    bars = ax.bar(ordinals[:len(top)], top, 0.6, color=BERRY, zorder=3)
    ax.bar_label(bars, fmt="%d", fontsize=9, padding=2, color=INK, weight="bold")
    ax.set_ylabel("monomers in the family")
    ax.set_ylim(0, top[0] * 1.16)
    ax.set_title(f"{top_n} largest Murcko scaffold families among {n_mon} B monomers")
    ax.text(0.97, 0.93, f"top two = {share:.1%}\nof all B monomers",
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            color=BERRY, weight="bold",
            bbox=dict(boxstyle="round,pad=0.45", facecolor=CREAM, edgecolor="none"))
    note(ax, f"{n_scaf} distinct scaffolds, {singletons} singletons  ·  "
             f"a balanced scaffold-disjoint 9-fold split is impossible  (source: data/ea_ip.csv)")
    save(fig, OUT, "fig3_scaffold_cluster_sizes")


FIGURES = {
    "fig1": fig1_run_to_run_variance,
    "fig2": fig2_variance_by_axis,
    "fig3": fig3_scaffold_cluster_sizes,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", choices=sorted(FIGURES), help="subset to build")
    args = ap.parse_args()
    apply_style()
    chosen = args.only or sorted(FIGURES)
    print(f"writing to {OUT}")
    for key in chosen:
        print(f"[{key}]")
        FIGURES[key]()


if __name__ == "__main__":
    main()
