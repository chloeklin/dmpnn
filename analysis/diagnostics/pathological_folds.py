"""Part 1: Absolute error metrics on monomer-heldout folds.

Computes MAE, RMSE, MedianAE, P95AE, MaxAE for all models × all folds × both
targets, alongside test target statistics (mean, std, range). Produces:
  - pathological_fold_metrics.csv
  - fold-wise bar plots (R², MAE, RMSE) per target
  - pathological_fold_report.md
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from . import config as _cfg
from .config import (
    COLORS, DPI, FOLD_MONOMER_NAMES, MARKERS, MODEL_DISPLAY, MODELS,
    PRED_ROOT, SPLIT_SUBDIRS, TARGETS, TARGET_TOKENS,
)
from .data_loading import load_dataset, load_split_meta

def _out_dir():
    return _cfg.OUT_ROOT / "11_pathological_folds"
N_LOMO_FOLDS = 9
SPLIT = "monomer_heldout"
PRED_DIR = PRED_ROOT / SPLIT_SUBDIRS[SPLIT]


# ── helpers ──────────────────────────────────────────────────────────────────

def _abs_errors(y_true: np.ndarray, y_pred: np.ndarray):
    ae = np.abs(y_true - y_pred)
    r2  = float(r2_score(y_true, y_pred))
    mae = float(np.mean(ae))
    rmse = float(np.sqrt(np.mean(ae ** 2)))
    med  = float(np.median(ae))
    p95  = float(np.percentile(ae, 95))
    mx   = float(np.max(ae))
    return dict(R2=r2, MAE=mae, RMSE=rmse, MedianAE=med, P95AE=p95, MaxAE=mx)


def _load_npz(model: str, tkey: str, fold: int) -> tuple[np.ndarray, np.ndarray] | None:
    seed = _cfg.ACTIVE_SEED
    fname = f"ea_ip__{TARGET_TOKENS[tkey]}__{model}__{SPLIT}__fold{fold}__s{seed}.npz"
    p = PRED_DIR / fname
    if not p.exists():
        return None
    arr = np.load(p, allow_pickle=True)
    return arr["y_true"].flatten().astype(float), arr["y_pred"].flatten().astype(float)


# ── main function ─────────────────────────────────────────────────────────────

def run_pathological_folds() -> pd.DataFrame:
    _out_dir().mkdir(parents=True, exist_ok=True)

    meta_folds = load_split_meta(SPLIT)
    fold_info = {f["fold"]: f for f in meta_folds}

    rows = []
    for tkey in TARGETS:
        for fold in range(N_LOMO_FOLDS):
            fi = fold_info.get(fold, {})
            held = fi.get("held_out_monomer_A", FOLD_MONOMER_NAMES.get(fold, f"fold{fold}"))

            # Collect y_true from any available model (same for all)
            yt_ref = None
            for m in MODELS:
                result = _load_npz(m, tkey, fold)
                if result is not None:
                    yt_ref = result[0]
                    break

            if yt_ref is None:
                continue

            t_mean = float(np.mean(yt_ref))
            t_std  = float(np.std(yt_ref))
            t_range = float(np.max(yt_ref) - np.min(yt_ref))

            for model in MODELS:
                result = _load_npz(model, tkey, fold)
                if result is None:
                    continue
                yt, yp = result
                m = _abs_errors(yt, yp)
                rows.append(dict(
                    split=SPLIT, target=tkey, fold=fold,
                    heldout_monomer=FOLD_MONOMER_NAMES.get(fold, held[:40]),
                    model=model,
                    **m,
                    TargetMean=t_mean, TargetStd=t_std, TargetRange=t_range,
                ))

    df = pd.DataFrame(rows)
    df.to_csv(_out_dir() / "pathological_fold_metrics.csv", index=False)
    print(f"  Saved: pathological_fold_metrics.csv  ({len(df)} rows)")

    _make_plots(df)
    _make_report(df)
    return df


# ── plotting ─────────────────────────────────────────────────────────────────

def _make_plots(df: pd.DataFrame):
    monomer_labels = [FOLD_MONOMER_NAMES.get(f, f"fold{f}") for f in range(N_LOMO_FOLDS)]
    # Shorten for axis labels
    short = [m[:22] + ("…" if len(m) > 22 else "") for m in monomer_labels]
    x = np.arange(N_LOMO_FOLDS)
    width = 0.18
    offsets = np.linspace(-(len(MODELS)-1)/2 * width, (len(MODELS)-1)/2 * width, len(MODELS))

    for tkey in TARGETS:
        sub = df[df["target"] == tkey]

        for metric in ("R2", "MAE", "RMSE"):
            fig, ax = plt.subplots(figsize=(13, 5))

            for i, model in enumerate(MODELS):
                msub = sub[sub["model"] == model].sort_values("fold")
                vals = []
                for fold in range(N_LOMO_FOLDS):
                    row = msub[msub["fold"] == fold]
                    vals.append(float(row[metric].values[0]) if len(row) else np.nan)

                ax.bar(x + offsets[i], vals, width=width,
                       color=COLORS[model], label=MODEL_DISPLAY[model], alpha=0.85)

            ax.set_xticks(x)
            ax.set_xticklabels(short, rotation=35, ha="right", fontsize=8)
            ax.set_xlabel("Held-out Monomer")
            ax.set_ylabel(metric)
            ax.set_title(f"Monomer-heldout {tkey}: {metric} per fold")
            ax.legend(fontsize=8)
            if metric == "R2":
                ax.axhline(0, color="k", lw=0.8, ls="--")

            plt.tight_layout()
            fname = f"monomer_heldout_{tkey.replace(' ', '_')}_{metric}.png"
            fig.savefig(_out_dir() / fname, dpi=DPI)
            plt.close(fig)

    # Highlight Fold 6 EA and hardest IP fold
    for tkey in TARGETS:
        sub = df[df["target"] == tkey]
        if tkey == "EA":
            highlight_fold = 6
            hl_label = "Fold 6 (benzothiadiazole)"
        else:
            # Hardest = highest MAE mean across models
            fold_mae = sub.groupby("fold")["MAE"].mean()
            highlight_fold = int(fold_mae.idxmax())
            hl_label = f"Fold {highlight_fold} ({FOLD_MONOMER_NAMES.get(highlight_fold, '')})"

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        for ax, metric in zip(axes, ("R2", "MAE", "RMSE")):
            for i, model in enumerate(MODELS):
                msub = sub[sub["model"] == model].sort_values("fold")
                vals = [float(msub[msub["fold"] == f][metric].values[0])
                        if len(msub[msub["fold"] == f]) else np.nan
                        for f in range(N_LOMO_FOLDS)]
                ax.plot(range(N_LOMO_FOLDS), vals,
                        marker=MARKERS[model], color=COLORS[model],
                        label=MODEL_DISPLAY[model], lw=1.5)

            ax.axvline(highlight_fold, color="red", lw=1.5, ls="--",
                       label=hl_label if metric == "R2" else None)
            ax.set_xticks(range(N_LOMO_FOLDS))
            ax.set_xticklabels([str(f) for f in range(N_LOMO_FOLDS)])
            ax.set_xlabel("Fold")
            ax.set_ylabel(metric)
            ax.set_title(f"{tkey}: {metric}")
            if metric == "R2":
                ax.axhline(0, color="k", lw=0.6, ls=":")
                ax.legend(fontsize=7)
        plt.suptitle(f"{tkey} — highlighted: {hl_label}", fontsize=10)
        plt.tight_layout()
        fname = f"monomer_heldout_{tkey.replace(' ', '_')}_highlight.png"
        fig.savefig(_out_dir() / fname, dpi=DPI)
        plt.close(fig)

    print(f"  Saved: fold-wise bar/line plots in {_out_dir()}")


# ── markdown report ───────────────────────────────────────────────────────────

def _make_report(df: pd.DataFrame):
    lines = ["# Pathological Folds Report — Monomer-heldout\n"]

    for tkey in TARGETS:
        sub = df[df["target"] == tkey]
        lines.append(f"\n## Target: {tkey}\n")

        lines.append("### Per-fold summary (mean across models)\n")
        fold_agg = (sub.groupby("fold")[["R2", "MAE", "RMSE", "TargetStd", "TargetRange"]]
                    .mean().round(4))
        lines.append(fold_agg.to_markdown())
        lines.append("\n")

        # Find worst fold per model by R²
        lines.append("### Worst fold per model (lowest R²)\n")
        lines.append("| Model | Fold | Held-out monomer | R² | MAE | RMSE | TargetStd |")
        lines.append("|-------|------|------------------|----|-----|------|-----------|")
        for model in MODELS:
            msub = sub[sub["model"] == model]
            if len(msub) == 0:
                continue
            worst = msub.loc[msub["R2"].idxmin()]
            lines.append(
                f"| {MODEL_DISPLAY[model]} | {int(worst.fold)} "
                f"| {worst.heldout_monomer[:35]} "
                f"| {worst.R2:.3f} | {worst.MAE:.4f} | {worst.RMSE:.4f} "
                f"| {worst.TargetStd:.4f} |"
            )
        lines.append("")

        # R² vs TargetStd scatter analysis
        lines.append("### R² vs Target Std (denominator collapse check)\n")
        # Correlation of R² with TargetStd across folds × models
        corr = sub[["R2", "TargetStd"]].corr().iloc[0, 1]
        lines.append(f"Pearson correlation between R² and test TargetStd: **{corr:.3f}**\n")
        lines.append("Interpretation:")
        if corr > 0.4:
            lines.append("- Positive correlation suggests denominator collapse (small variance → low R²)")
            lines.append("  even when absolute errors are comparable to other folds.")
        else:
            lines.append("- Low or negative correlation: R² failures reflect genuinely large errors,")
            lines.append("  not just denominator collapse.")
        lines.append("")

        # Compare fold 6 EA specifically
        if tkey == "EA":
            f6 = sub[sub["fold"] == 6]
            lines.append("### Fold 6 Detail (benzothiadiazole held-out, EA)\n")
            lines.append("| Model | R² | MAE | RMSE | MedianAE | P95AE | MaxAE | TargetStd |")
            lines.append("|-------|-----|-----|------|----------|-------|-------|-----------|")
            for model in MODELS:
                row = f6[f6["model"] == model]
                if len(row) == 0:
                    continue
                r = row.iloc[0]
                lines.append(
                    f"| {MODEL_DISPLAY[model]} | {r.R2:.3f} | {r.MAE:.4f} "
                    f"| {r.RMSE:.4f} | {r.MedianAE:.4f} | {r.P95AE:.4f} "
                    f"| {r.MaxAE:.4f} | {r.TargetStd:.4f} |"
                )
            lines.append("")
            ts = float(f6["TargetStd"].mean())
            mae_ca = float(f6[f6["model"] == "chemarch"]["MAE"].values[0]) if len(f6[f6["model"] == "chemarch"]) else float("nan")
            lines.append(f"TargetStd = {ts:.4f} eV  |  ChemArch MAE = {mae_ca:.4f} eV")
            if ts < 0.1:
                lines.append(f"\n**Denominator collapse: test std = {ts:.4f} eV is very small.**")
                lines.append("Even a MAE of 0.05 eV can produce R² ≪ 0.")
            else:
                lines.append(f"\n**TargetStd is {ts:.4f} eV — not trivially small.**")
                lines.append("Negative R² reflects genuinely poor predictions.")
            lines.append("")

    # Final diagnostic answer
    lines.append("---\n")
    lines.append("## Diagnostic Answer\n")
    lines.append("### Q: Is negative R² caused mainly by genuinely large errors, or amplified by small target variance?\n")

    for tkey in TARGETS:
        sub = df[df["target"] == tkey]
        neg_r2 = sub[sub["R2"] < 0]
        lines.append(f"**{tkey}:** {len(neg_r2)} model-fold combinations with R² < 0")
        if len(neg_r2) > 0:
            med_std = float(neg_r2["TargetStd"].median())
            med_mae = float(neg_r2["MAE"].median())
            corr = sub[["R2", "TargetStd"]].corr().iloc[0, 1]
            lines.append(f"- Median TargetStd for failing folds: {med_std:.4f} eV")
            lines.append(f"- Median MAE for failing folds: {med_mae:.4f} eV")
            lines.append(f"- R²–TargetStd correlation: {corr:.3f}")
            if corr > 0.4 and med_std < 0.15:
                lines.append("- **Verdict: DENOMINATOR COLLAPSE is the primary driver.**")
                lines.append("  Absolute errors are not catastrophically larger than passing folds;")
                lines.append("  the test variance is unusually small, inflating the negative R².")
            elif med_mae > 0.3:
                lines.append("- **Verdict: GENUINELY LARGE ERRORS are the primary driver.**")
                lines.append(f"  Median MAE = {med_mae:.3f} eV is large in absolute terms.")
            else:
                lines.append("- **Verdict: MIXED — both denominator shrinkage and elevated errors contribute.**")
        lines.append("")

    (_out_dir() / "pathological_fold_report.md").write_text("\n".join(lines))
    print(f"  Saved: pathological_fold_report.md")
