"""Generate analysis/model_diagnostics/13_final_followup_summary.md

Aggregates findings from Parts 1-3 into a structured summary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from . import config as _cfg
from .config import FOLD_MONOMER_NAMES, MODEL_DISPLAY, TARGETS


def run_followup_summary(
    path_df: pd.DataFrame | None = None,
    corr_df: pd.DataFrame | None = None,
    abla_df: pd.DataFrame | None = None,
):
    """Read CSVs from disk if DataFrames not provided."""
    p11 = _cfg.OUT_ROOT / "11_pathological_folds"
    p12 = _cfg.OUT_ROOT / "12_residual_correlation"
    p13 = _cfg.OUT_ROOT / "13_chemarch_residual_ablation"

    if path_df is None:
        f = p11 / "pathological_fold_metrics.csv"
        path_df = pd.read_csv(f) if f.exists() else pd.DataFrame()
    if corr_df is None:
        f = p12 / "residual_correlations.csv"
        corr_df = pd.read_csv(f) if f.exists() else pd.DataFrame()
    if abla_df is None:
        f = p13 / "chemarch_pre_vs_post.csv"
        abla_df = pd.read_csv(f) if f.exists() else pd.DataFrame()

    lines = []
    lines.append("# Final Follow-up Diagnostic Summary\n")
    lines.append("This report synthesises findings from three supplementary diagnostics:\n")
    lines.append("- **Part 1**: Absolute errors on pathological folds (11_pathological_folds)")
    lines.append("- **Part 2**: Residual correlation between models (12_residual_correlation)")
    lines.append("- **Part 3**: ChemArch backbone-only vs full residual ablation (13_chemarch_residual_ablation)\n")
    lines.append("---\n")

    # ── Q1: What causes ChemArch's LOMO failures? ────────────────────────────
    lines.append("## Q1: What causes ChemArch's catastrophic LOMO failures?\n")
    if len(path_df) > 0 and len(abla_df) > 0:
        ca_lomo = path_df[(path_df["model"] == "chemarch") &
                          (path_df["split"] == "monomer_heldout")]
        neg_r2 = ca_lomo[ca_lomo["R2"] < 0]

        lines.append(f"ChemArch has **{len(neg_r2)}** model-fold-target combinations with R² < 0 "
                     f"out of {len(ca_lomo)} total on monomer-heldout.\n")

        # Characterise: large errors vs small variance?
        if len(neg_r2) > 0:
            corr_ea = ca_lomo[ca_lomo["target"] == "EA"][["R2", "TargetStd"]].corr().iloc[0, 1]
            corr_ip = ca_lomo[ca_lomo["target"] == "IP"][["R2", "TargetStd"]].corr().iloc[0, 1]
            lines.append(f"R²–TargetStd correlation: EA = {corr_ea:.3f},  IP = {corr_ip:.3f}")
            lines.append(f"Median MAE in R² < 0 cases: {float(neg_r2['MAE'].median()):.4f} eV")
            lines.append(f"Median TargetStd in R² < 0 cases: {float(neg_r2['TargetStd'].median()):.4f} eV\n")
            if float(neg_r2['TargetStd'].median()) < 0.15:
                lines.append("**Primary driver: denominator collapse** — test sets with unusually "
                              "small variance make even moderate absolute errors catastrophic for R².")
            else:
                lines.append("**Primary driver: genuinely large absolute errors** on held-out monomer chemistry.")
            lines.append("")

        # Ablation evidence
        lines.append("### Evidence from residual ablation (Part 3)\n")
        if len(abla_df) > 0:
            lomo_abla = abla_df  # already filtered to monomer_heldout
            for tkey in TARGETS:
                t = lomo_abla[lomo_abla["target"] == tkey]
                if len(t) == 0:
                    continue
                full_r2  = t[t["mode"] == "full"]["R2_overall"].mean()
                back_r2  = t[t["mode"] == "backbone_only"]["R2_overall"].mean()
                delta    = full_r2 - back_r2
                full_mae = t[t["mode"] == "full"]["MAE_overall"].mean()
                back_mae = t[t["mode"] == "backbone_only"]["MAE_overall"].mean()
                lines.append(f"**{tkey}** — mean across folds:")
                lines.append(f"  - Full ChemArch:    R² = {full_r2:.3f},  MAE = {full_mae:.4f} eV")
                lines.append(f"  - Backbone only:    R² = {back_r2:.3f},  MAE = {back_mae:.4f} eV")
                lines.append(f"  - ΔR² (full−backbone): {delta:+.3f}")
                if delta < -0.05:
                    lines.append(f"  → The residual head **hurts** on monomer-heldout for {tkey}.")
                elif delta > 0.05:
                    lines.append(f"  → The residual head **helps** on monomer-heldout for {tkey}.")
                else:
                    lines.append(f"  → The residual head has **negligible effect** on {tkey}.")
                lines.append("")
    else:
        lines.append("_(pathological_fold_metrics.csv or chemarch_pre_vs_post.csv not found)_\n")

    # ── Q2: Is failure in backbone or residual head? ──────────────────────────
    lines.append("---\n")
    lines.append("## Q2: Is the LOMO failure in the chemistry baseline, the residual head, or architecture prediction?\n")
    if len(abla_df) > 0:
        for tkey in TARGETS:
            t = abla_df[abla_df["target"] == tkey]
            if len(t) == 0:
                continue
            lines.append(f"### {tkey}\n")
            lines.append("| Fold | Held-out monomer | Backbone R² | Full R² | ΔR² |")
            lines.append("|------|-----------------|-------------|---------|-----|")
            for fold in sorted(t["fold"].unique()):
                row_full = t[(t["fold"] == fold) & (t["mode"] == "full")]
                row_back = t[(t["fold"] == fold) & (t["mode"] == "backbone_only")]
                if len(row_full) == 0 or len(row_back) == 0:
                    continue
                r2f = float(row_full["R2_overall"].values[0])
                r2b = float(row_back["R2_overall"].values[0])
                mon = FOLD_MONOMER_NAMES.get(fold, f"fold{fold}")[:30]
                lines.append(f"| {fold} | {mon} | {r2b:.3f} | {r2f:.3f} | {r2f-r2b:+.3f} |")
            lines.append("")

            # Summary interpretation — compare paired fold values
            folds_t = sorted(t["fold"].unique())
            both_bad = 0
            back_ok_full_bad = 0
            back_bad_full_ok = 0
            both_ok = 0
            for f in folds_t:
                r2f = t[(t["fold"] == f) & (t["mode"] == "full")]["R2_overall"]
                r2b = t[(t["fold"] == f) & (t["mode"] == "backbone_only")]["R2_overall"]
                if len(r2f) == 0 or len(r2b) == 0:
                    continue
                r2f, r2b = float(r2f.values[0]), float(r2b.values[0])
                if r2b < 0 and r2f < 0:
                    both_bad += 1
                elif r2b >= 0 and r2f < 0:
                    back_ok_full_bad += 1
                elif r2b < 0 and r2f >= 0:
                    back_bad_full_ok += 1
                else:
                    both_ok += 1
            lines.append(f"- Folds where **backbone bad, residual rescues** (R²_back<0, R²_full≥0): {back_bad_full_ok}")
            lines.append(f"- Folds where **both bad** (R²<0 for both): {both_bad}")
            lines.append(f"- Folds where **residual hurts** (backbone OK, full bad): {back_ok_full_bad}")
            lines.append(f"- Folds where **both OK**: {both_ok}\n")
            if back_bad_full_ok > both_bad + back_ok_full_bad:
                lines.append("**Conclusion:** The **residual head rescues a poor backbone** on most folds. "
                              "The chemistry backbone alone cannot extrapolate; the architecture-conditioned "
                              "correction substantially recovers performance.\n")
            elif both_bad > back_bad_full_ok:
                lines.append("**Conclusion:** The failure is primarily in the **chemistry backbone** — "
                              "the encoder cannot extrapolate to unseen monomers, and the residual "
                              "head is insufficient to recover performance.\n")
            elif back_ok_full_bad > 0:
                lines.append("**Conclusion:** The residual head **introduces** degradation on some folds. "
                              "The backbone alone generalises better in those cases.\n")
            else:
                lines.append("**Conclusion:** Both components generalise adequately; "
                              "failures are fold-specific and likely reflect unusual monomer chemistry.\n")
    else:
        lines.append("_(chemarch_pre_vs_post.csv not found)_\n")

    # ── Q3: Are ChemArch and wDMPNN making complementary mistakes? ───────────
    lines.append("---\n")
    lines.append("## Q3: Are ChemArch and wDMPNN making complementary mistakes?\n")
    if len(corr_df) > 0:
        ca_wd = corr_df[(corr_df["model_A"] == "chemarch") &
                        (corr_df["model_B"] == "wdmpnn")]
        for rtype in ("overall", "group_mean", "arch_deviation"):
            sub = ca_wd[ca_wd["residual_type"] == rtype]
            if len(sub) == 0:
                continue
            mean_p = float(sub["pearson"].mean())
            mean_s = float(sub["spearman"].mean())
            mean_cov = float(sub["covariance"].mean())
            lines.append(f"**{rtype}** residuals — ChemArch vs wDMPNN:")
            lines.append(f"  - Mean Pearson r = {mean_p:.3f}")
            lines.append(f"  - Mean Spearman ρ = {mean_s:.3f}")
            lines.append(f"  - Mean covariance = {mean_cov:.4f}")
            if mean_p > 0.7:
                interpretation = "**largely the same mistakes** (high positive correlation)"
            elif mean_p < 0.3:
                interpretation = "**substantially complementary mistakes** (low correlation)"
            else:
                interpretation = "**partially complementary** (moderate correlation)"
            lines.append(f"  → {interpretation}\n")

        # Lomo specifically
        lomo = ca_wd[ca_wd["split"] == "monomer_heldout"]
        if len(lomo) > 0:
            lomo_overall = lomo[lomo["residual_type"] == "overall"]
            p_lomo = float(lomo_overall["pearson"].mean())
            lines.append(f"On **monomer-heldout** specifically: mean Pearson r = {p_lomo:.3f}")
            if p_lomo > 0.7:
                lines.append("Both models make similar extrapolation errors on held-out monomers.")
            else:
                lines.append("The two models diverge more on held-out monomers, "
                              "suggesting they encode different aspects of monomer chemistry.")
        lines.append("")
    else:
        lines.append("_(residual_correlations.csv not found)_\n")

    # ── Q4: Hypothesis — combined encoder ────────────────────────────────────
    lines.append("---\n")
    lines.append("## Q4: Does evidence support combining a graph encoder with an explicit chemistry-conditioned residual?\n")
    lines.append('**Hypothesis:** "A graph-based chemistry encoder combined with an explicit '
                 'chemistry-conditioned architecture residual head could potentially combine '
                 'the strengths of both models."\n')

    evidence_for = []
    evidence_against = []

    if len(corr_df) > 0:
        ca_wd = corr_df[(corr_df["model_A"] == "chemarch") &
                        (corr_df["model_B"] == "wdmpnn") &
                        (corr_df["residual_type"] == "overall")]
        mean_p = float(ca_wd["pearson"].mean()) if len(ca_wd) > 0 else np.nan
        if not np.isnan(mean_p):
            if mean_p < 0.6:
                evidence_for.append(
                    f"ChemArch and wDMPNN residuals have low–moderate overall correlation "
                    f"(r={mean_p:.2f}), indicating they encode partially complementary information."
                )
            else:
                evidence_against.append(
                    f"ChemArch and wDMPNN residuals are highly correlated (r={mean_p:.2f}), "
                    f"suggesting both models make similar errors — ensemble gains would be limited."
                )

    if len(abla_df) > 0:
        # Does residual help in-distribution?
        in_dist = abla_df  # all are monomer_heldout here; extend if group/pair available
        for tkey in TARGETS:
            t = in_dist[in_dist["target"] == tkey]
            if len(t) == 0:
                continue
            delta_arch = (t[t["mode"] == "full"]["R2_arch_dev"].values -
                          t[t["mode"] == "backbone_only"]["R2_arch_dev"].values)
            if len(delta_arch) > 0:
                mean_d = float(np.nanmean(delta_arch))
                if mean_d > 0.02:
                    evidence_for.append(
                        f"{tkey}: residual head improves arch-deviation R² by ΔR²={mean_d:+.3f} "
                        f"— the arch-conditioned correction provides signal beyond the backbone."
                    )
                elif mean_d < -0.02:
                    evidence_against.append(
                        f"{tkey}: residual head *hurts* arch-deviation R² by ΔR²={mean_d:+.3f} "
                        f"on monomer-heldout — the residual overfits to seen monomer chemistry."
                    )

    if len(path_df) > 0:
        wd = path_df[(path_df["model"] == "wdmpnn") & (path_df["split"] == "monomer_heldout")]
        ca = path_df[(path_df["model"] == "chemarch") & (path_df["split"] == "monomer_heldout")]
        if len(wd) > 0 and len(ca) > 0:
            wd_r2 = float(wd["R2"].mean())
            ca_r2 = float(ca["R2"].mean())
            if wd_r2 > ca_r2 + 0.05:
                evidence_for.append(
                    f"wDMPNN (mean R²={wd_r2:.2f}) substantially outperforms ChemArch "
                    f"(mean R²={ca_r2:.2f}) on LOMO — demonstrating that a better chemistry "
                    f"encoder would benefit the architecture-aware model."
                )

    lines.append("### Supporting evidence\n")
    if evidence_for:
        for e in evidence_for:
            lines.append(f"- {e}")
    else:
        lines.append("- No direct supporting evidence found from the available diagnostics.")
    lines.append("")

    lines.append("### Contrary evidence\n")
    if evidence_against:
        for e in evidence_against:
            lines.append(f"- {e}")
    else:
        lines.append("- No direct contrary evidence found.")
    lines.append("")

    lines.append("### Conclusion\n")
    if len(evidence_for) > len(evidence_against):
        lines.append(
            "The diagnostics **support** the hypothesis. The key finding is that wDMPNN's "
            "stronger chemistry encoder generalises better to unseen monomers, while ChemArch's "
            "architecture residual provides ordering signal within-distribution. A model that "
            "combines a wDMPNN-quality chemistry encoder (or graph-based backbone) with an "
            "explicit architecture-conditioned correction head could, in principle, capture both "
            "strengths. This hypothesis is mechanistically plausible but would require "
            "experimental validation through retraining."
        )
    elif len(evidence_against) > len(evidence_for):
        lines.append(
            "The diagnostics **do not strongly support** the hypothesis. The residual head "
            "does not consistently improve over the backbone, and the error correlation between "
            "ChemArch and wDMPNN is high — suggesting the two models are not learning "
            "sufficiently complementary representations to benefit from combination."
        )
    else:
        lines.append(
            "The evidence is **mixed**. The hypothesis is plausible but current diagnostics "
            "are insufficient to confirm or refute it decisively. A dedicated ablation with "
            "a wDMPNN backbone + architecture residual head would be required."
        )
    lines.append("")
    lines.append("---\n")
    lines.append("_Generated automatically by `analysis/diagnostics/followup_summary.py`_")

    out_path = _cfg.OUT_ROOT / "13_final_followup_summary.md"
    out_path.write_text("\n".join(lines))
    print(f"  Saved: {out_path}")
    return out_path
