#!/usr/bin/env python3
"""
Complete report-ready analysis of EA/IP model comparison.

Run from project root:
    python3 scripts/python/analyze_ea_ip_report.py
"""

import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parents[2]
CSV_PATH    = PROJECT_DIR / "plots" / "ea_ip_random_vs_monomer" / "ea_ip_random_vs_monomer_consolidated.csv"
OUT_DIR     = PROJECT_DIR / "analysis" / "ea_ip_report"
for d in [OUT_DIR / "tables", OUT_DIR / "plots", OUT_DIR / "reports"]:
    d.mkdir(parents=True, exist_ok=True)

# ── Paper reference values ─────────────────────────────────────────────────────
PAPER_REF = pd.DataFrame([
    {"comparison": "D-MPNN",  "split": "random",  "target": "EA vs SHE (eV)", "paper_rmse": 0.17},
    {"comparison": "D-MPNN",  "split": "random",  "target": "IP vs SHE (eV)", "paper_rmse": 0.16},
    {"comparison": "D-MPNN",  "split": "monomer", "target": "EA vs SHE (eV)", "paper_rmse": 0.20},
    {"comparison": "D-MPNN",  "split": "monomer", "target": "IP vs SHE (eV)", "paper_rmse": 0.20},
    {"comparison": "wD-MPNN", "split": "random",  "target": "EA vs SHE (eV)", "paper_rmse": 0.03},
    {"comparison": "wD-MPNN", "split": "random",  "target": "IP vs SHE (eV)", "paper_rmse": 0.03},
    {"comparison": "wD-MPNN", "split": "monomer", "target": "EA vs SHE (eV)", "paper_rmse": 0.10},
    {"comparison": "wD-MPNN", "split": "monomer", "target": "IP vs SHE (eV)", "paper_rmse": 0.09},
])

PROP_ORDER = ["EA vs SHE (eV)", "IP vs SHE (eV)"]
PROP_SHORT = {"EA vs SHE (eV)": "EA", "IP vs SHE (eV)": "IP"}

# ── Plot style ─────────────────────────────────────────────────────────────────
def _style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 9, "legend.fontsize": 8,
        "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
    })

# ── 1. Load & clean ───────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["poly_type"] = df["poly_type"].astype(str).str.lower().isin(["true", "1", "yes"])
    df["has_data"]  = df["rmse_mean"].notna()

    def _family(row):
        if row["category"] == "Identity":  return "Identity"
        if row["category"] == "Tabular":   return "Tabular"
        if row["model"] in ("wDMPNN", "HPG"): return "Graph (topology-aware)"
        return "Graph (standard GNN)"

    df["family"] = df.apply(_family, axis=1)
    return df

# ── 2. Summary tables ─────────────────────────────────────────────────────────
def family_best_table(df: pd.DataFrame) -> pd.DataFrame:
    dv = df[df["has_data"]].copy()
    rows = []
    for (family, split, target), grp in dv.groupby(["family", "split", "target"]):
        best = grp.sort_values("rmse_mean").iloc[0]
        rows.append({"family": family, "split": split, "target": PROP_SHORT[target],
                     "best_model": best["model_label"],
                     "RMSE": round(best["rmse_mean"], 4),
                     "±": round(best["rmse_std"], 4),
                     "MAE": round(best["mae_mean"], 4),
                     "R²": round(best["r2_mean"], 4)})
    return pd.DataFrame(rows).sort_values(["split", "target", "RMSE"])

def pt_effect_table(df: pd.DataFrame) -> pd.DataFrame:
    dv = df[df["has_data"]].copy()
    rows = []
    for _, row_base in dv[~dv["poly_type"]].iterrows():
        pt_match = dv[
            dv["poly_type"] &
            (dv["model"]  == row_base["model"]) &
            (dv["split"]  == row_base["split"]) &
            (dv["target"] == row_base["target"])
        ]
        if pt_match.empty:
            continue
        row_pt = pt_match.sort_values("rmse_mean").iloc[0]
        delta  = row_base["rmse_mean"] - row_pt["rmse_mean"]
        rows.append({
            "base_model":    row_base["model_label"],
            "pt_model":      row_pt["model_label"],
            "split":         row_base["split"],
            "target":        PROP_SHORT[row_base["target"]],
            "RMSE_base":     round(row_base["rmse_mean"], 4),
            "RMSE_+PT":      round(row_pt["rmse_mean"],  4),
            "delta_RMSE":    round(delta, 4),
            "improvement_%": round(100 * delta / row_base["rmse_mean"], 1),
        })
    return pd.DataFrame(rows).sort_values("delta_RMSE", ascending=False)

def paper_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    dv = df[df["has_data"]].copy()
    rows = []
    for _, ref in PAPER_REF.iterrows():
        model_filter = "DMPNN" if ref["comparison"] == "D-MPNN" else "wDMPNN"
        sub = dv[
            (dv["model"]  == model_filter) &
            (dv["split"]  == ref["split"]) &
            (dv["target"] == ref["target"]) &
            (~dv["poly_type"])
        ]
        if sub.empty:
            rows.append({"comparison": ref["comparison"], "split": ref["split"],
                         "target": PROP_SHORT[ref["target"]], "our_variant": "—",
                         "our_RMSE": None, "paper_RMSE": ref["paper_rmse"],
                         "diff": None, "rel_%": None})
            continue
        best = sub.sort_values("rmse_mean").iloc[0]
        diff = best["rmse_mean"] - ref["paper_rmse"]
        rows.append({"comparison": ref["comparison"], "split": ref["split"],
                     "target": PROP_SHORT[ref["target"]], "our_variant": best["model_label"],
                     "our_RMSE": round(best["rmse_mean"], 4), "paper_RMSE": ref["paper_rmse"],
                     "diff": round(diff, 4), "rel_%": round(100 * diff / ref["paper_rmse"], 1)})
    return pd.DataFrame(rows)

def generalization_gap_table(df: pd.DataFrame) -> pd.DataFrame:
    dv = df[df["has_data"] & ~df["poly_type"]].copy()
    rows = []
    for (label, target), grp in dv.groupby(["model_label", "target"]):
        rand = grp[grp["split"] == "random"]
        mono = grp[grp["split"] == "monomer"]
        if rand.empty or mono.empty:
            continue
        r, m = rand.iloc[0]["rmse_mean"], mono.iloc[0]["rmse_mean"]
        rows.append({"model": label, "target": PROP_SHORT[target],
                     "family": grp.iloc[0]["family"],
                     "RMSE_random": round(r, 4), "RMSE_monomer": round(m, 4),
                     "gap": round(m - r, 4), "gap_%": round(100 * (m - r) / r, 1)})
    return pd.DataFrame(rows).sort_values(["target", "gap_%"], ascending=[True, False])

# ── 3. Plots ──────────────────────────────────────────────────────────────────
MODEL_ORDER_BASE = [
    "Identity (mix)", "Identity (interact)",
    "DMPNN (mix)", "DMPNN (interact)",
    "GAT (mix)", "GAT (interact)",
    "GIN (mix)", "GIN (interact)",
    "wDMPNN",
    "HPG", "HPG +desc",
    "Linear", "RF", "XGB",
]
CONDITIONS    = [("random","EA vs SHE (eV)"),("random","IP vs SHE (eV)"),
                 ("monomer","EA vs SHE (eV)"),("monomer","IP vs SHE (eV)")]
COND_LABELS   = ["Rand EA","Rand IP","Mono EA","Mono IP"]
COND_COLORS   = ["#4e79a7","#f28e2b","#59a14f","#e15759"]

FAMILY_SHADES = {
    "Identity":    ("#ede7f6", 0, 2),
    "Std GNNs":    ("#e3f2fd", 2, 9),
    "Topo-aware":  ("#fff3e0", 9, 11),
    "Tabular":     ("#fce4ec", 11, 14),
}

def plot_performance_by_family(df: pd.DataFrame):
    _style()
    dv = df[df["has_data"] & ~df["poly_type"]].copy()
    avail = dv["model_label"].unique()
    order = [m for m in MODEL_ORDER_BASE if m in avail]

    n = len(order)
    rmse = np.full((n, 4), np.nan)
    err  = np.full((n, 4), np.nan)
    for ci, (split, target) in enumerate(CONDITIONS):
        sub = dv[(dv["split"] == split) & (dv["target"] == target)]
        for mi, ml in enumerate(order):
            row = sub[sub["model_label"] == ml]
            if not row.empty:
                rmse[mi, ci] = row.iloc[0]["rmse_mean"]
                err[mi, ci]  = row.iloc[0]["rmse_std"]

    fig, ax = plt.subplots(figsize=(15, 5))
    x, w = np.arange(n), 0.18
    for ci, (label, color) in enumerate(zip(COND_LABELS, COND_COLORS)):
        mask = ~np.isnan(rmse[:, ci])
        ax.bar(x[mask] + (ci - 1.5) * w, rmse[mask, ci], w,
               yerr=err[mask, ci], capsize=2, label=label, color=color,
               alpha=0.85, ecolor="grey", error_kw={"linewidth": 0.7})

    for name, (shade, lo, hi) in FAMILY_SHADES.items():
        hi_clamp = min(hi, n)
        ax.axvspan(lo - 0.5, hi_clamp - 0.5, alpha=0.12, color=shade, zorder=0)
        ax.text((lo + hi_clamp) / 2 - 0.5, ax.get_ylim()[1] * 0.97, name,
                ha="center", va="top", fontsize=7, color="#555555")

    ax.set_xticks(x); ax.set_xticklabels(order, rotation=38, ha="right")
    ax.set_ylabel("RMSE (eV)"); ax.set_title("EA/IP Prediction RMSE by Model (base variants, no PT)")
    ax.legend(loc="upper right", ncol=2); ax.set_xlim(-0.6, n - 0.4)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "plots" / "01_performance_by_family.png", bbox_inches="tight")
    plt.close()
    print("  ✓ 01_performance_by_family.png")

def plot_pt_gain(pt_df: pd.DataFrame):
    """2×2 grid: rows = split (random/monomer), cols = target (EA/IP).
    Each base model appears exactly once per panel."""
    _style()
    splits  = ["random", "monomer"]
    targets = ["EA", "IP"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ri, split in enumerate(splits):
        for ci, target_short in enumerate(targets):
            ax  = axes[ri, ci]
            sub = pt_df[(pt_df["split"] == split) & (pt_df["target"] == target_short)].copy()

            if sub.empty:
                ax.text(0.5, 0.5, "No results yet", transform=ax.transAxes,
                        ha="center", va="center", color="grey")
                ax.set_title(f"PT effect — {split.capitalize()} | {target_short}")
                ax.axis("off"); continue

            sub = sub.sort_values("delta_RMSE", ascending=True)
            colors = ["#2ca02c" if d > 0 else "#d62728" for d in sub["delta_RMSE"]]
            bars = ax.barh(range(len(sub)), sub["delta_RMSE"], color=colors, alpha=0.85)
            ax.set_yticks(range(len(sub)))
            ax.set_yticklabels(
                [f"{r['base_model']}\n→ {r['pt_model']}" for _, r in sub.iterrows()],
                fontsize=8)
            ax.axvline(0, color="k", linewidth=0.8)
            ax.set_xlabel("RMSE reduction (eV)  [positive = improved by PT]")
            ax.set_title(f"PT effect — {split.capitalize()} split | {target_short}")
            for bar, (_, row) in zip(bars, sub.iterrows()):
                w = bar.get_width()
                sign = "+" if w >= 0 else ""
                ax.text(w + (0.0006 if w >= 0 else -0.0006),
                        bar.get_y() + bar.get_height() / 2,
                        f"{sign}{row['improvement_%']:.0f}%",
                        va="center", ha="left" if w >= 0 else "right", fontsize=7)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "plots" / "02_pt_gain.png", bbox_inches="tight")
    plt.close()
    print("  ✓ 02_pt_gain.png")

def plot_generalization_gap(gap_df: pd.DataFrame):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for ai, target_short in enumerate(["EA", "IP"]):
        ax  = axes[ai]
        sub = gap_df[gap_df["target"] == target_short].sort_values("RMSE_random")
        if sub.empty:
            ax.set_title(f"Generalisation gap — {target_short}"); continue

        x = np.arange(len(sub))
        ax.bar(x - 0.22, sub["RMSE_random"],  0.4, label="Random split",         color="#4e79a7", alpha=0.85)
        ax.bar(x + 0.22, sub["RMSE_monomer"], 0.4, label="Monomer held-out",      color="#f28e2b", alpha=0.85)
        for xi, (_, row) in enumerate(sub.iterrows()):
            ax.plot([xi - 0.22, xi + 0.22], [row["RMSE_random"], row["RMSE_monomer"]],
                    "k-", alpha=0.25, linewidth=1)
            ax.text(xi + 0.22, row["RMSE_monomer"] + 0.003, f'+{row["gap_%"]:.0f}%',
                    ha="center", fontsize=6.5, color="#555555")

        ax.set_xticks(x); ax.set_xticklabels(sub["model"], rotation=38, ha="right")
        ax.set_ylabel("RMSE (eV)"); ax.set_title(f"Generalisation Gap — {target_short}")
        ax.legend()
    plt.tight_layout()
    fig.savefig(OUT_DIR / "plots" / "03_generalization_gap.png", bbox_inches="tight")
    plt.close()
    print("  ✓ 03_generalization_gap.png")

def plot_hpg_comparison(df: pd.DataFrame):
    _style()
    hpg = df[(df["model"] == "HPG") & df["has_data"]].copy()
    # best standard GNN per split/target (no PT)
    gnn_ref = (df[
                   (df["family"] == "Graph (standard GNN)") &
                   df["has_data"] & ~df["poly_type"]
               ]
               .groupby(["split", "target"])
               .apply(lambda g: g.sort_values("rmse_mean").iloc[0])
               .reset_index(drop=True))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors_t = ["#4e79a7", "#f28e2b"]

    for ai, split in enumerate(["random", "monomer"]):
        ax      = axes[ai]
        hpg_sub = hpg[hpg["split"] == split]
        variants = sorted(hpg_sub["model_label"].unique())
        x = np.arange(len(variants))

        for ti, target in enumerate(PROP_ORDER):
            vals = []; errs = []
            for v in variants:
                row = hpg_sub[(hpg_sub["model_label"] == v) & (hpg_sub["target"] == target)]
                vals.append(row.iloc[0]["rmse_mean"] if not row.empty else np.nan)
                errs.append(row.iloc[0]["rmse_std"]  if not row.empty else 0)
            ax.bar(x + (ti - 0.5) * 0.35, vals, 0.35,
                   yerr=errs, capsize=3, label=PROP_SHORT[target],
                   color=colors_t[ti], alpha=0.85, ecolor="grey", error_kw={"linewidth": 0.8})

        # reference lines: best standard GNN for this split
        for ti, target in enumerate(PROP_ORDER):
            ref = gnn_ref[(gnn_ref["split"] == split) & (gnn_ref["target"] == target)]
            if not ref.empty:
                ax.axhline(ref.iloc[0]["rmse_mean"], color=colors_t[ti],
                           linestyle="--", alpha=0.55, linewidth=1.2,
                           label=f"Best std GNN ({PROP_SHORT[target]})")

        ax.set_xticks(x); ax.set_xticklabels(variants, rotation=15, ha="right")
        ax.set_ylabel("RMSE (eV)"); ax.set_title(f"HPG variants — {split.capitalize()} split")
        ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "plots" / "04_hpg_comparison.png", bbox_inches="tight")
    plt.close()
    print("  ✓ 04_hpg_comparison.png")

def plot_tabular_comparison(df: pd.DataFrame):
    _style()
    tab = df[(df["category"] == "Tabular") & df["has_data"]].copy()
    colors_t = ["#4e79a7", "#f28e2b"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ai, split in enumerate(["random", "monomer"]):
        ax  = axes[ai]
        sub = tab[tab["split"] == split]
        variants = sorted(sub["model_label"].unique())
        x = np.arange(len(variants))
        for ti, target in enumerate(PROP_ORDER):
            vals = []; errs = []
            for v in variants:
                row = sub[(sub["model_label"] == v) & (sub["target"] == target)]
                vals.append(row.iloc[0]["rmse_mean"] if not row.empty else np.nan)
                errs.append(row.iloc[0]["rmse_std"]  if not row.empty else 0)
            ax.bar(x + (ti - 0.5) * 0.35, vals, 0.35,
                   yerr=errs, capsize=3, label=PROP_SHORT[target],
                   color=colors_t[ti], alpha=0.85, ecolor="grey", error_kw={"linewidth": 0.8})
        ax.set_xticks(x); ax.set_xticklabels(variants, rotation=20, ha="right")
        ax.set_ylabel("RMSE (eV)"); ax.set_title(f"Tabular models — {split.capitalize()} split")
        ax.legend()
    plt.tight_layout()
    fig.savefig(OUT_DIR / "plots" / "05_tabular_comparison.png", bbox_inches="tight")
    plt.close()
    print("  ✓ 05_tabular_comparison.png")

def plot_paper_comparison(cmp_df: pd.DataFrame):
    _style()
    valid = cmp_df[cmp_df["our_RMSE"].notna()].copy()
    colors  = {"D-MPNN": "#1f77b4", "wD-MPNN": "#2ca02c"}
    markers = {"random": "o", "monomer": "s"}

    fig, ax = plt.subplots(figsize=(7, 6))
    for _, row in valid.iterrows():
        ax.scatter(row["paper_RMSE"], row["our_RMSE"],
                   color=colors.get(row["comparison"], "grey"),
                   marker=markers.get(row["split"], "o"),
                   s=90, alpha=0.9, zorder=3, linewidths=0.5, edgecolors="white")
        ax.annotate(f"{row['comparison']} {row['split'][:4]} {row['target']}",
                    (row["paper_RMSE"], row["our_RMSE"]),
                    textcoords="offset points", xytext=(6, 3), fontsize=7, alpha=0.75)

    all_vals = list(valid["paper_RMSE"]) + list(valid["our_RMSE"])
    lo, hi = min(all_vals) * 0.85, max(all_vals) * 1.15
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.45, label="y = x (parity)")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

    legend_els = [
        mpatches.Patch(color="#1f77b4", label="D-MPNN"),
        mpatches.Patch(color="#2ca02c", label="wD-MPNN"),
        plt.Line2D([0],[0], marker="o", color="grey", ls="None", label="random split"),
        plt.Line2D([0],[0], marker="s", color="grey", ls="None", label="monomer split"),
        plt.Line2D([0],[0], ls="--", color="k", alpha=0.45, label="parity"),
    ]
    ax.legend(handles=legend_els, fontsize=8)
    ax.set_xlabel("Paper RMSE (eV)"); ax.set_ylabel("Our RMSE (eV)")
    ax.set_title("Our Results vs. Published Reference Values")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "plots" / "06_paper_comparison.png", bbox_inches="tight")
    plt.close()
    print("  ✓ 06_paper_comparison.png")

# ── 4. Reports ────────────────────────────────────────────────────────────────
def _fmt_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)

def _get(df, model, split, target_kw, metric="rmse_mean"):
    r = df[(df["model"] == model) & (df["split"] == split) &
           df["target"].str.startswith(target_kw) & ~df["poly_type"]]
    return r[metric].min() if not r.empty else None

def write_executive_summary(df, fam_df, pt_df, cmp_df):
    dv  = df[df["has_data"]].copy()
    NOW = datetime.now().strftime("%B %Y")

    dmpnn_rand_ea = _get(dv, "DMPNN", "random",  "EA")
    dmpnn_mono_ea = _get(dv, "DMPNN", "monomer", "EA")
    wdmpnn_mono_ea = _get(dv, "wDMPNN", "monomer", "EA")
    wdmpnn_mono_ip = _get(dv, "wDMPNN", "monomer", "IP")
    xgb_rand_ea   = _get(dv, "XGB",   "random",  "EA")
    ident_rand_ea = dv[(dv["model"] == "IdentityBaseline") & (dv["split"] == "random") &
                       dv["target"].str.startswith("EA")]["rmse_mean"].min()

    pt_gains    = pt_df[pt_df["delta_RMSE"] > 0]
    avg_pt_gain = pt_gains["improvement_%"].mean() if not pt_gains.empty else 0
    max_pt_row  = pt_df.sort_values("improvement_%", ascending=False).iloc[0] if not pt_df.empty else None

    def _f(v): return f"{v:.3f}" if v is not None else "N/A"

    lines = [
        "# Executive Summary — EA/IP Copolymer Property Prediction",
        f"\n**Date:** {NOW}  ",
        "**Dataset:** ea_ip (copolymer electron affinity & ionisation potential)  ",
        "**Targets:** EA vs SHE (eV), IP vs SHE (eV)  ",
        "**Evaluation:** 5-fold cross-validation, mean ± std; RMSE in eV (lower is better)  ",
        "**Split strategies:** random and monomer-held-out (a_held_out)",
        "\n---\n",
        "## 1. Scope",
        "A total of **" + str(dv["model_label"].nunique()) + " model variants** were evaluated, spanning:",
        "- Identity baseline (polymer type one-hot only)",
        "- Standard GNNs: DMPNN, GAT, GIN (with mix/interact copolymer modes, ±PT)",
        "- Topology-aware GNNs: wDMPNN, HPG (±desc, ±PT)",
        "- Classical ML tabular models: Linear regression, Random Forest, XGBoost",
        f"\n{(~df['has_data']).sum()} of {len(df)} model–split–target combinations are pending (results not yet available).",
        "\n---\n",
        "## 2. Key Findings",
        "",
        "### Finding 1 — Polymer topology is a dominant predictive signal",
        f"The **Identity baseline** (using only polymer-type one-hot encoding, no molecular features) achieves "
        f"RMSE ≈ {_f(ident_rand_ea)} eV for EA on the random split. This is substantially better than the "
        "paper's D-MPNN baseline (~0.17 eV), confirming that the alternating/block/random classification of "
        "the polymer architecture captures most of the variance in EA/IP under i.i.d. conditions.",
        "",
        "### Finding 2 — GNN +PT variants achieve near-perfect random-split performance",
        "GAT+PT and GIN+PT reach RMSE ≈ 0.022–0.023 eV on random-split EA/IP — a >3× reduction over "
        "the corresponding base variants. **This result warrants scrutiny**: if the random split does not "
        "stratify by polymer type, the train and test sets will share the same type distribution, trivially "
        "inflating PT-based performance. A stratified-by-type ablation is recommended.",
        "",
        "### Finding 3 — Standard GNNs match the published D-MPNN baseline",
        f"DMPNN (mix, no PT) achieves RMSE ≈ {_f(dmpnn_rand_ea)} eV (EA, random), closely matching "
        "the published D-MPNN reference of ~0.17 eV. The monomer-split RMSE "
        f"({_f(dmpnn_mono_ea)} eV EA) also aligns well with the paper's ~0.20 eV.",
        "",
        "### Finding 4 — wDMPNN approaches but does not match the published benchmark",
        f"wDMPNN achieves RMSE ≈ {_f(wdmpnn_mono_ea)} eV (EA) and {_f(wdmpnn_mono_ip)} eV (IP) "
        "on the monomer-held-out split. The published wD-MPNN values are ~0.10 and ~0.09 eV respectively. "
        "The ~3× gap may reflect differences in training epochs, learning rate schedule, or dataset preprocessing. "
        "**Random-split results for wDMPNN are not yet available.**",
        "",
        "### Finding 5 — HPG underperforms significantly",
        "HPG variants (original and +desc) achieve RMSE > 0.5 eV across all conditions — roughly 5–8× worse "
        "than standard GNNs. This is unexpected for a topology-aware hierarchical architecture and strongly "
        "suggests a hyperparameter tuning or data preparation issue (see Further Experiments).",
        "",
        "### Finding 6 — Tabular models match the paper D-MPNN baseline",
        f"XGBoost achieves RMSE ≈ {_f(xgb_rand_ea)} eV (EA, random), comparable to the published D-MPNN "
        "baseline without any graph representation. RF is similarly competitive. Linear regression is "
        "substantially weaker, confirming the need for non-linear models in this task.",
        "",
        "### Finding 7 — Generalisation gap is universal but varies by model",
        "Every model shows higher RMSE on the monomer-held-out split, confirming this is a harder "
        "generalisation task. The gap is smallest for wDMPNN (~1.6× increase) and largest for HPG (~1.0× "
        "in absolute terms but high baseline). Standard GNNs show ~1.5–1.8× increase.",
        "\n---\n",
        "## 3. Recommendations",
        "1. Validate the +PT random-split results with a polymer-type-stratified split.",
        "2. Prioritise running wDMPNN on random splits to directly compare with the ~0.03 eV benchmark.",
        "3. Investigate HPG hyperparameters (hidden dim, depth, LR) before reporting HPG results.",
        "4. Run all pending Tabular +PT experiments to quantify PT effect for classical ML.",
    ]
    (OUT_DIR / "reports" / "executive_summary.md").write_text("\n".join(lines))

def write_supervisor_summary(df, fam_df, pt_df, cmp_df, gap_df):
    dv  = df[df["has_data"]].copy()
    NOW = datetime.now().strftime("%B %Y")

    wdmpnn_mono = dv[(dv["model"] == "wDMPNN") & (dv["split"] == "monomer")]
    wea = wdmpnn_mono[wdmpnn_mono["target"].str.startswith("EA")]["rmse_mean"]
    wip = wdmpnn_mono[wdmpnn_mono["target"].str.startswith("IP")]["rmse_mean"]
    wea_s = f"{wea.values[0]:.3f}" if len(wea) else "N/A"
    wip_s = f"{wip.values[0]:.3f}" if len(wip) else "N/A"

    lines = [
        "# Supervisor-Facing Findings Summary",
        f"\n**Prepared:** {NOW}  ",
        "**Project context:** Benchmarking graph neural networks and classical ML for copolymer "
        "electrochemical property prediction (EA, IP vs SHE).  ",
        "**Evaluation protocol:** 5-fold CV; random and monomer-held-out splits; RMSE (eV).",
        "\n---\n",
        "## Model Families",
        "| Family | Models | Notes |",
        "|--------|--------|-------|",
        "| Identity baseline | IdentityBaseline (mix, interact) | Polymer type one-hot only |",
        "| Standard GNNs | DMPNN, GAT, GIN | mix/interact modes; ±polymer type |",
        "| Topology-aware GNNs | wDMPNN, HPG | Designed for polymer structure |",
        "| Tabular ML | Linear, RF, XGBoost | Molecular descriptors (AB, RDKit) |",
        "\n## Best Model per Family (RMSE, eV)\n",
        _fmt_table(fam_df),
        "\n## Polymer Type (PT) Effect — Selected Pairs\n",
        _fmt_table(pt_df[["base_model","pt_model","split","target",
                           "RMSE_base","RMSE_+PT","delta_RMSE","improvement_%"]].head(12)),
        "\n## Comparison with Published Values\n",
        _fmt_table(cmp_df),
        "\n## Generalisation Gap (random → monomer, base variants)\n",
        _fmt_table(gap_df),
        "\n---\n",
        "## Discussion Points for Supervisor Meeting",
        "",
        "**1. PT leakage risk (high priority):**  ",
        "Standard GNNs with polymer type (+PT) on random splits show RMSE ~0.022 eV — an order of "
        "magnitude better than without PT, and better than the wD-MPNN benchmark. This is likely "
        "inflated because random splits may place all three polymer types in both train and test. "
        "A polymer-type-stratified split is needed before reporting these numbers.",
        "",
        "**2. wDMPNN gap from benchmark:**  ",
        f"Our wDMPNN monomer-split results (EA: {wea_s} eV, IP: {wip_s} eV) are approximately 3× "
        "worse than the published benchmark. Likely causes: fewer training epochs, different LR schedule, "
        "or a dataset/preprocessing discrepancy. Recommend checking against the original paper's "
        "training configuration.",
        "",
        "**3. HPG poor performance:**  ",
        "HPG RMSE > 0.5 eV on all conditions suggests the model is not training effectively on this "
        "dataset. Recommend: (a) verify graph construction for copolymers, (b) sweep hidden dim "
        "(256–512) and depth (4–8), (c) increase training budget.",
        "",
        "**4. Tabular models are surprisingly competitive:**  ",
        "XGBoost and RF without graph features match the D-MPNN paper baseline (~0.17 eV). "
        "This is useful context: graph-based approaches need to clearly exceed this threshold "
        "to justify their added complexity.",
    ]
    (OUT_DIR / "reports" / "supervisor_summary.md").write_text("\n".join(lines))

def write_further_experiments(df, pt_df):
    missing = df[~df["has_data"]].groupby("model_label")["split"].apply(
        lambda s: ", ".join(sorted(set(s)))).reset_index()
    missing.columns = ["model_variant", "missing_splits"]

    lines = [
        "# Further Experiments Required",
        f"\n**Date:** {datetime.now().strftime('%B %Y')}",
        f"\n{(~df['has_data']).sum()} of {len(df)} model–split–target combinations have no results.\n",
        "## Missing Experiments\n",
        _fmt_table(missing),
        "\n---\n",
        "## Prioritised Experiment List",
        "",
        "### Priority 1 — Critical for paper comparison",
        "- **wDMPNN, random split**: Required to compare against the published ~0.03 eV benchmark.",
        "",
        "### Priority 2 — Required to complete the PT analysis",
        "- **DMPNN +PT, random split**: Needed to check consistency with GAT/GIN +PT results.",
        "- **HPG +desc +PT, random split**: Needed to complete the HPG variant grid.",
        "- **All Tabular +PT variants** (both splits): Quantify PT benefit for classical ML.",
        "",
        "### Priority 3 — Validation / ablation",
        "- **Polymer-type-stratified random split for all +PT models**: The current random split "
        "may not stratify by polymer type. Adding a split that ensures each type is equally "
        "represented in train/test is needed to confirm whether +PT improvements are real or artefactual.",
        "- **HPG hyperparameter sweep**: Current HPG RMSE > 0.5 eV. Recommended sweep:",
        "  - hidden_dim: [128, 256, 512]",
        "  - depth: [4, 6, 8]",
        "  - max_lr: [1e-3, 3e-3]",
        "  - dropout_ffn: [0.1, 0.2, 0.3]",
        "",
        "### Priority 4 — Interpretability",
        "- **Ablation: Identity baseline (PT only) vs GNN+PT**: The identity baseline already achieves "
        "RMSE ~0.07 eV. An ablation comparing GNN+PT against PT-only would isolate how much of the "
        "GNN+PT gain comes from molecular graph features versus polymer topology alone.",
        "",
        "## Summary Table",
        "| Experiment | Reason | Estimated runtime |",
        "|------------|--------|-------------------|",
        "| wDMPNN random split | Paper comparison | ~1.5h |",
        "| DMPNN +PT random split | Complete PT grid | ~2.5h |",
        "| Tabular +PT (all splits) | PT for classical ML | ~2h |",
        "| HPG +desc +PT random | Complete HPG grid | ~1h |",
        "| PT-stratified split ablation | Validate +PT gains | ~5h |",
        "| HPG hyperparameter sweep | Fix poor HPG perf. | ~10h |",
    ]
    (OUT_DIR / "reports" / "further_experiments.md").write_text("\n".join(lines))

def write_readme(df, fam_df):
    NOW = datetime.now().strftime("%B %d, %Y")
    lines = [
        "# EA/IP Copolymer Model Comparison — Analysis Report",
        f"\n**Generated:** {NOW}  ",
        "**Source:** `plots/ea_ip_random_vs_monomer/ea_ip_random_vs_monomer_consolidated.csv`  ",
        "**Script:** `scripts/python/analyze_ea_ip_report.py`",
        "\n---\n",
        "## Folder Structure",
        "```",
        "analysis/ea_ip_report/",
        "├── tables/",
        "│   ├── 01_schema_summary.txt          schema, coverage, missing entries",
        "│   ├── 02_best_by_family.csv          best model per family × split × target",
        "│   ├── 03_pt_effect.csv               RMSE delta from adding polymer type",
        "│   ├── 04_paper_comparison.csv        our results vs published D-MPNN / wD-MPNN",
        "│   └── 05_generalization_gap.csv      random vs monomer RMSE gap per model",
        "├── plots/",
        "│   ├── 01_performance_by_family.png   all base models, all conditions",
        "│   ├── 02_pt_gain.png                 RMSE reduction from adding PT",
        "│   ├── 03_generalization_gap.png      paired random vs monomer bars",
        "│   ├── 04_hpg_comparison.png          HPG variants + GNN reference lines",
        "│   ├── 05_tabular_comparison.png      tabular model comparison",
        "│   └── 06_paper_comparison.png        parity plot vs published values",
        "├── reports/",
        "│   ├── executive_summary.md           key findings, 1-page summary",
        "│   ├── supervisor_summary.md          formal report for supervisor",
        "│   └── further_experiments.md         prioritised list of pending work",
        "└── README.md                          this file",
        "```",
        "\n## Dataset Coverage",
        f"- **Total model–split–target combinations:** {len(df)}",
        f"- **With results:** {df['has_data'].sum()}",
        f"- **Pending (no results):** {(~df['has_data']).sum()}",
        "\n## Best Model per Family (RMSE, eV)\n",
        _fmt_table(fam_df),
        "\n## Paper Reference Values",
        "| Model | Split | EA RMSE | IP RMSE |",
        "|-------|-------|---------|---------|",
        "| D-MPNN (paper)  | random  | ~0.17 | ~0.16 |",
        "| D-MPNN (paper)  | monomer | ~0.20 | ~0.20 |",
        "| wD-MPNN (paper) | random  | ~0.03 | ~0.03 |",
        "| wD-MPNN (paper) | monomer | ~0.10 | ~0.09 |",
        "\n## Notes",
        "- All metrics: mean ± std over 5-fold CV; RMSE in eV; lower is better.",
        "- +PT = polymer type (alternating/block/random) added as one-hot encoding.",
        "- Missing entries indicate experiments not yet run.",
        "- See `reports/further_experiments.md` for prioritised list of pending work.",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines))

# ── 5. Schema summary text ─────────────────────────────────────────────────────
def schema_summary_text(df: pd.DataFrame) -> str:
    dv = df[df["has_data"]]
    missing = df[~df["has_data"]]
    lines = [
        "EA/IP Consolidated Results — Schema Summary",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Source: {CSV_PATH}",
        "",
        f"Total rows       : {len(df)}",
        f"Rows with data   : {len(dv)}",
        f"Missing rows     : {len(missing)}",
        "",
        "Columns:",
        "  category       — model family (Identity, Graph, Tabular)",
        "  model          — model name (DMPNN, GAT, GIN, wDMPNN, HPG, Linear, RF, XGB, IdentityBaseline)",
        "  copolymer_mode — mix | interact | desc | N/A",
        "  poly_type      — bool, True = poly_type one-hot added",
        "  model_label    — human-readable model variant label",
        "  split          — random | monomer",
        "  target         — EA vs SHE (eV) | IP vs SHE (eV)",
        "  mae_mean/std   — mean absolute error (eV)",
        "  rmse_mean/std  — root mean squared error (eV)",
        "  r2_mean/std    — coefficient of determination",
        "",
        "Unique values:",
    ]
    for col in ["category", "model", "copolymer_mode", "poly_type", "split", "target"]:
        lines.append(f"  {col:18s}: {sorted(df[col].dropna().unique().tolist())}")
    lines += [
        "",
        "Missing entries by model variant:",
    ]
    miss_grp = missing.groupby("model_label")["split"].apply(
        lambda s: ", ".join(sorted(set(s)))).reset_index()
    for _, row in miss_grp.iterrows():
        lines.append(f"  {row['model_label']:30s}: {row['split']}")
    return "\n".join(lines)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Loading {CSV_PATH.name} ...")
    df = load_data()
    print(f"  {len(df)} rows, {df['has_data'].sum()} with data, {(~df['has_data']).sum()} missing")

    print("\nComputing tables ...")
    fam_df  = family_best_table(df)
    pt_df   = pt_effect_table(df)
    cmp_df  = paper_comparison_table(df)
    gap_df  = generalization_gap_table(df)
    schema  = schema_summary_text(df)

    print("Saving tables ...")
    (OUT_DIR / "tables" / "01_schema_summary.txt").write_text(schema)
    fam_df.to_csv(OUT_DIR / "tables" / "02_best_by_family.csv", index=False)
    pt_df.to_csv( OUT_DIR / "tables" / "03_pt_effect.csv",      index=False)
    cmp_df.to_csv(OUT_DIR / "tables" / "04_paper_comparison.csv", index=False)
    gap_df.to_csv(OUT_DIR / "tables" / "05_generalization_gap.csv", index=False)
    for t in [fam_df, pt_df, cmp_df, gap_df]:
        pass  # already saved
    print("  ✓ 5 table files")

    print("\nGenerating plots ...")
    plot_performance_by_family(df)
    plot_pt_gain(pt_df)
    plot_generalization_gap(gap_df)
    plot_hpg_comparison(df)
    plot_tabular_comparison(df)
    plot_paper_comparison(cmp_df)

    print("\nWriting reports ...")
    write_executive_summary(df, fam_df, pt_df, cmp_df)
    write_supervisor_summary(df, fam_df, pt_df, cmp_df, gap_df)
    write_further_experiments(df, pt_df)
    write_readme(df, fam_df)
    print("  ✓ executive_summary.md")
    print("  ✓ supervisor_summary.md")
    print("  ✓ further_experiments.md")
    print("  ✓ README.md")

    print(f"\n{'='*55}")
    print(f"All outputs saved → {OUT_DIR}")
    print(f"  tables/  : {len(list((OUT_DIR/'tables').iterdir()))} files")
    print(f"  plots/   : {len(list((OUT_DIR/'plots').iterdir()))} files")
    print(f"  reports/ : {len(list((OUT_DIR/'reports').iterdir()))} files")

    # Print top-line numbers to stdout
    print(f"\n── Top-line RMSE summary (best per family, EA, random split) ──")
    subset = fam_df[(fam_df["split"] == "random") & (fam_df["target"] == "EA")]
    for _, row in subset.iterrows():
        print(f"  {row['family']:28s}  {row['best_model']:25s}  RMSE={row['RMSE']:.4f} ± {row['±']:.4f}")

    print(f"\n── Paper comparison (RMSE, eV) ──")
    for _, row in cmp_df.iterrows():
        our = f"{row['our_RMSE']:.4f}" if row["our_RMSE"] is not None else "N/A"
        diff = f"{row['diff']:+.4f} ({row['rel_%']:+.0f}%)" if row["diff"] is not None else "N/A"
        print(f"  {row['comparison']:8s} {row['split']:8s} {row['target']:4s}  "
              f"paper={row['paper_RMSE']:.3f}  ours={our}  diff={diff}")

if __name__ == "__main__":
    main()
