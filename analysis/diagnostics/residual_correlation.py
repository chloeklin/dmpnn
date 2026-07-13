"""Part 2: Residual correlation between models.

For every split × target × fold, computes:
  - overall residuals
  - group-mean residuals
  - architecture-deviation residuals

between all model pairs, with focus on ChemArch vs wDMPNN.

Outputs:
  - residual_correlations.csv
  - scatter/hexbin plots
  - largest_model_disagreements.csv
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.metrics import r2_score

from .config import (
    COLORS, DPI, FOLD_MONOMER_NAMES, MODEL_DISPLAY, MODELS,
    N_FOLDS, OUT_ROOT, SPLITS, TARGETS,
)
from .data_loading import load_all_meta, load_dataset, load_predictions_single
from .grouping import build_fold_df, filter_matched_groups

OUT_DIR = OUT_ROOT / "12_residual_correlation"
PAIRS = [("chemarch", "wdmpnn"), ("globalarch", "wdmpnn"), ("frac", "wdmpnn")]
PAIR_LABELS = {
    ("chemarch", "wdmpnn"):   ("ChemArch", "wDMPNN"),
    ("globalarch", "wdmpnn"): ("GlobalArch", "wDMPNN"),
    ("frac", "wdmpnn"):       ("Frac", "wDMPNN"),
}


# ── residual computation ──────────────────────────────────────────────────────

def _residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return y_true - y_pred


def _group_mean_residuals(fold_df: pd.DataFrame,
                           e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (group-mean e, arch-deviation e) arrays aligned to fold_df rows."""
    tmp = fold_df.copy()
    tmp["e"] = e
    grp_mean_e = tmp.groupby("group_key")["e"].transform("mean")
    arch_dev_e = tmp["e"] - grp_mean_e
    return grp_mean_e.values, arch_dev_e.values


def _pairwise_metrics(e1: np.ndarray, e2: np.ndarray) -> dict:
    mask = np.isfinite(e1) & np.isfinite(e2)
    e1, e2 = e1[mask], e2[mask]
    if len(e1) < 3:
        return dict(pearson=np.nan, spearman=np.nan, covariance=np.nan,
                    rmse_diff=np.nan, n=len(e1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pearson  = float(np.corrcoef(e1, e2)[0, 1])
        spearman = float(sp_stats.spearmanr(e1, e2).statistic)
    cov  = float(np.cov(e1, e2)[0, 1])
    rmse_diff = float(np.sqrt(np.mean((np.abs(e1) - np.abs(e2)) ** 2)))
    return dict(pearson=pearson, spearman=spearman,
                covariance=cov, rmse_diff=rmse_diff, n=len(e1))


# ── main function ─────────────────────────────────────────────────────────────

def run_residual_correlation(df: pd.DataFrame, meta: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    corr_rows = []
    # Accumulate all residuals for scatter plots and disagreement analysis
    all_resid: dict[tuple, list] = {p: [] for p in PAIRS}  # pair -> list of (e1, e2, sample info)

    for split in SPLITS:
        meta_folds = meta[split]
        for tkey in TARGETS:
            for fold in range(N_FOLDS[split]):
                # Load all models for this fold
                preds = {}
                for model in MODELS:
                    p = load_predictions_single(model, tkey, split, fold, meta_folds)
                    if p is not None:
                        preds[model] = p

                if len(preds) < 2:
                    continue

                # Use first available model to get indices / fold_df
                ref_model = next(iter(preds))
                ref = preds[ref_model]
                fdf = build_fold_df(df, ref["y_true"], ref["y_pred"], ref["global_idx"])

                # Compute residuals for each model
                resids = {}
                for model, p in preds.items():
                    resids[model] = _residuals(p["y_true"], p["y_pred"])

                for (m1, m2) in PAIRS:
                    if m1 not in resids or m2 not in resids:
                        continue
                    e1 = resids[m1]
                    e2 = resids[m2]

                    # 1. Overall
                    om = _pairwise_metrics(e1, e2)
                    corr_rows.append(dict(split=split, target=tkey, fold=fold,
                                          model_A=m1, model_B=m2, residual_type="overall", **om))

                    # 2. Group-mean
                    gm_e1, ad_e1 = _group_mean_residuals(fdf, e1)
                    gm_e2, ad_e2 = _group_mean_residuals(fdf, e2)
                    gm = _pairwise_metrics(gm_e1, gm_e2)
                    corr_rows.append(dict(split=split, target=tkey, fold=fold,
                                          model_A=m1, model_B=m2, residual_type="group_mean", **gm))

                    # 3. Architecture deviation
                    ad = _pairwise_metrics(ad_e1, ad_e2)
                    corr_rows.append(dict(split=split, target=tkey, fold=fold,
                                          model_A=m1, model_B=m2, residual_type="arch_deviation", **ad))

                    # Accumulate for scatter (overall residuals)
                    sample_info = {
                        "split": split, "target": tkey, "fold": fold,
                        "e1": e1, "e2": e2,
                        "y_true": ref["y_true"],
                        "global_idx": ref["global_idx"],
                        "fdf": fdf,
                    }
                    all_resid[(m1, m2)].append(sample_info)

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(OUT_DIR / "residual_correlations.csv", index=False)
    print(f"  Saved: residual_correlations.csv  ({len(corr_df)} rows)")

    # Scatter/hexbin plots
    _make_scatter_plots(df, all_resid, corr_df)

    # Largest disagreements
    disagree_df = _make_disagreement_table(df, all_resid)
    disagree_df.to_csv(OUT_DIR / "largest_model_disagreements.csv", index=False)
    print(f"  Saved: largest_model_disagreements.csv  ({len(disagree_df)} rows)")

    return corr_df, disagree_df


# ── scatter/hexbin plots ──────────────────────────────────────────────────────

def _make_scatter_plots(df: pd.DataFrame, all_resid: dict, corr_df: pd.DataFrame):
    """One figure per (model_pair, target): rows=splits, cols=overall/scatter+hexbin."""
    for (m1, m2) in PAIRS:
        lA, lB = PAIR_LABELS[(m1, m2)]
        data_list = all_resid[(m1, m2)]
        if not data_list:
            continue

        for tkey in TARGETS:
            t_data = [d for d in data_list if d["target"] == tkey]
            if not t_data:
                continue

            splits_present = sorted(set(d["split"] for d in t_data))
            n_splits = len(splits_present)
            fig, axes = plt.subplots(n_splits, 2, figsize=(10, 4 * n_splits))
            if n_splits == 1:
                axes = [axes]

            for row_i, split in enumerate(splits_present):
                sd = [d for d in t_data if d["split"] == split]
                e1_all = np.concatenate([d["e1"] for d in sd])
                e2_all = np.concatenate([d["e2"] for d in sd])

                # Pearson from corr_df
                sub = corr_df[(corr_df["model_A"] == m1) & (corr_df["model_B"] == m2) &
                              (corr_df["split"] == split) & (corr_df["target"] == tkey) &
                              (corr_df["residual_type"] == "overall")]
                pearson  = float(sub["pearson"].mean()) if len(sub) else np.nan
                spearman = float(sub["spearman"].mean()) if len(sub) else np.nan

                ax_sc  = axes[row_i][0]
                ax_hex = axes[row_i][1]

                # Scatter (downsample if large)
                n_pts = len(e1_all)
                if n_pts > 5000:
                    rng = np.random.default_rng(42)
                    idx = rng.choice(n_pts, 5000, replace=False)
                    e1s, e2s = e1_all[idx], e2_all[idx]
                else:
                    e1s, e2s = e1_all, e2_all

                ax_sc.scatter(e1s, e2s, alpha=0.2, s=8,
                              color=COLORS.get(m1, "steelblue"))
                # Regression line
                mask = np.isfinite(e1s) & np.isfinite(e2s)
                if mask.sum() >= 2:
                    m_fit, b_fit = np.polyfit(e1s[mask], e2s[mask], 1)
                    xl = np.linspace(e1s[mask].min(), e1s[mask].max(), 100)
                    ax_sc.plot(xl, m_fit * xl + b_fit, "r-", lw=1.5, label=f"fit")
                lim = max(np.percentile(np.abs(np.concatenate([e1s, e2s])), 99) * 1.1, 0.1)
                ax_sc.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, label="y=x")
                ax_sc.set_xlim(-lim, lim); ax_sc.set_ylim(-lim, lim)
                ax_sc.set_xlabel(f"{lA} residual (eV)")
                ax_sc.set_ylabel(f"{lB} residual (eV)")
                ax_sc.set_title(f"{split} | r={pearson:.3f}  ρ={spearman:.3f}  n={n_pts:,}")
                ax_sc.legend(fontsize=7)
                ax_sc.axhline(0, lw=0.5, ls=":", c="gray")
                ax_sc.axvline(0, lw=0.5, ls=":", c="gray")

                # Hexbin
                hb = ax_hex.hexbin(e1_all, e2_all, gridsize=60, cmap="viridis",
                                   mincnt=1, bins="log",
                                   extent=[-lim, lim, -lim, lim])
                ax_hex.plot([-lim, lim], [-lim, lim], "r--", lw=0.8)
                ax_hex.set_xlim(-lim, lim); ax_hex.set_ylim(-lim, lim)
                ax_hex.set_xlabel(f"{lA} residual (eV)")
                ax_hex.set_ylabel(f"{lB} residual (eV)")
                ax_hex.set_title(f"{split} hexbin (log count)")
                fig.colorbar(hb, ax=ax_hex, label="log(count)")

            plt.suptitle(f"Residual correlation: {lA} vs {lB}  —  {tkey}", fontsize=11)
            plt.tight_layout()
            fname = f"residual_scatter_{m1}_vs_{m2}_{tkey.replace(' ', '_')}.png"
            fig.savefig(OUT_DIR / fname, dpi=DPI)
            plt.close(fig)

    print(f"  Saved: residual scatter/hexbin plots in {OUT_DIR}")


# ── disagreement table ────────────────────────────────────────────────────────

def _make_disagreement_table(df: pd.DataFrame, all_resid: dict) -> pd.DataFrame:
    """Top-100 samples where ChemArch >> wDMPNN and vice versa."""
    pair = ("chemarch", "wdmpnn")
    data_list = all_resid[pair]
    if not data_list:
        return pd.DataFrame()

    # Pool all samples across splits/targets/folds
    records = []
    for d in data_list:
        e1 = d["e1"]  # chemarch
        e2 = d["e2"]  # wdmpnn
        yt = d["y_true"]
        gidx = d["global_idx"]
        for i, gi in enumerate(gidx):
            row = df.iloc[gi]
            ae1 = abs(float(e1[i]))
            ae2 = abs(float(e2[i]))
            records.append(dict(
                global_idx=int(gi),
                split=d["split"], target=d["target"], fold=d["fold"],
                smiles_A=row.get("smiles_A", ""),
                smiles_B=row.get("smiles_B", ""),
                fracA=float(row.get("fracA", np.nan)),
                poly_type=row.get("poly_type", ""),
                y_true=float(yt[i]),
                y_pred_chemarch=float(yt[i]) - float(e1[i]),
                y_pred_wdmpnn=float(yt[i]) - float(e2[i]),
                ae_chemarch=ae1,
                ae_wdmpnn=ae2,
                diff_ca_minus_wd=ae1 - ae2,  # positive → chemarch worse
            ))

    pool = pd.DataFrame(records)

    # Top 100 where ChemArch greatly OUTPERFORMS wDMPNN (smaller error)
    ca_better = (pool.sort_values("diff_ca_minus_wd")
                     .head(100)
                     .copy())
    ca_better["who_wins"] = "chemarch_better"

    # Top 100 where wDMPNN greatly OUTPERFORMS ChemArch
    wd_better = (pool.sort_values("diff_ca_minus_wd", ascending=False)
                     .head(100)
                     .copy())
    wd_better["who_wins"] = "wdmpnn_better"

    combined = pd.concat([ca_better, wd_better], ignore_index=True)
    return combined
