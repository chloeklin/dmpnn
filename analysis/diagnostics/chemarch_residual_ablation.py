"""Part 3: ChemArch backbone-only vs full prediction (monomer-heldout).

The Stage2D ChemArch (2d1_arch) forward computes:
    h_poly = h_mix + alpha_arch * r_arch(z)

"Backbone-only" disables the residual by zeroing the alpha parameters
at inference time, without retraining.  This is safe because:
  1. alpha is a Parameter (not part of the GNN encoder or prediction heads)
  2. Setting alpha = 0 gives h_poly = h_mix  (the fraction baseline)
  3. The same scaler/heads are used, so outputs are directly comparable

Outputs:
  - chemarch_pre_vs_post.csv
  - comparison plots (backbone vs full per split)
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, mean_absolute_error

# Make project root importable
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts" / "python"))

from .config import (
    COLORS, DPI, FOLD_MONOMER_NAMES, MODEL_DISPLAY, OUT_ROOT, TARGETS,
    TARGET_TOKENS, PRED_ROOT, SPLIT_SUBDIRS,
)
from .data_loading import load_dataset, load_split_meta
from .grouping import build_fold_df, filter_matched_groups

OUT_DIR = OUT_ROOT / "13_chemarch_residual_ablation"
CKPT_DIR = _ROOT / "checkpoints" / "ea_ip_lomo"
SPLIT = "monomer_heldout"
N_FOLDS = 9

# Checkpoint naming: ea_ip__<target>__copoly_stage2d_2d1_arch__a_held_out__rep<fold>
VARIANT = "2d1_arch"
VARIANT_DISPLAY = "ChemArch (2d1_arch)"


# ── helpers ───────────────────────────────────────────────────────────────────

def _ckpt_dir(target: str, fold: int) -> Path:
    return CKPT_DIR / f"ea_ip__{target}__copoly_stage2d_{VARIANT}__a_held_out__rep{fold}"


def _find_ckpt(ckpt_dir: Path) -> Path | None:
    logs_dir = ckpt_dir / "logs" / "checkpoints"
    if not logs_dir.exists():
        return None
    ckpts = sorted(logs_dir.glob("*.ckpt"), key=lambda f: f.stat().st_mtime)
    return ckpts[-1] if ckpts else None


def _load_chemarch_model(target: str, fold: int,
                          meta_folds: list, df: pd.DataFrame):
    """Load a ChemArch checkpoint and return (model, test_loader, scaler, test_idx).

    The scaler is reconstructed from training targets so that the model's
    UnscaleTransform parameters match what was used during training.
    """
    from chemprop import nn as cpnn
    from chemprop.data import CopolymerDataset, build_dataloader
    from chemprop.data.datapoints import MoleculeDatapoint
    from chemprop.featurizers.molgraph import SimpleMoleculeMolGraphFeaturizer
    from chemprop.models.copolymer import CopolymerMPNN
    from rdkit import Chem
    from chemprop.utils import make_mol

    ckpt_dir = _ckpt_dir(target, fold)
    if not ckpt_dir.exists():
        return None

    ckpt_path = _find_ckpt(ckpt_dir)
    if ckpt_path is None:
        return None

    # Fold test indices from metadata
    fold_info = {f["fold"]: f for f in meta_folds}
    fi = fold_info.get(fold)
    if fi is None:
        return None
    test_idx  = np.array(fi["global_test_indices"], dtype=int)
    all_idx   = np.arange(len(df))
    train_idx = np.setdiff1d(all_idx, test_idx)

    target_col = target  # e.g. "EA vs SHE (eV)"
    smiles_A = df["smiles_A"].values
    smiles_B = df["smiles_B"].values
    fracA    = df["fracA"].values.astype(float)
    fracB    = df["fracB"].values.astype(float)
    y_vals   = df[target_col].values.astype(float)
    arch_map = {"alternating": 0, "random": 1, "block": 2}
    arch_idx = df["poly_type"].map(arch_map).fillna(0).values.astype(int)

    feat = SimpleMoleculeMolGraphFeaturizer()

    def _make_dp(i, side="A"):
        smi = smiles_A[i] if side == "A" else smiles_B[i]
        mol = make_mol(smi, keep_h=False, add_h=False)
        # Arch index in x_d (shape [1]) — only on side A; targets only on side A
        if side == "A":
            return MoleculeDatapoint(
                mol=mol,
                y=np.array([y_vals[i]], dtype=float),
                x_d=np.array([arch_idx[i]], dtype=float),
            )
        else:
            return MoleculeDatapoint(mol=mol)

    def _make_ds(idx):
        dA = [_make_dp(j, "A") for j in idx]
        dB = [_make_dp(j, "B") for j in idx]
        return CopolymerDataset(dA, dB, fracA[idx], fracB[idx], feat)

    train_ds = _make_ds(train_idx)
    test_ds  = _make_ds(test_idx)

    # Reconstruct scaler from training targets
    scaler = train_ds.normalize_targets()

    # Build model skeleton (architecture must match checkpoint)
    mp  = cpnn.BondMessagePassing()
    agg = cpnn.MeanAggregation()
    d_mp = mp.output_dim

    output_transform = cpnn.UnscaleTransform.from_standard_scaler(scaler)
    ffn = cpnn.RegressionFFN(output_transform=output_transform, n_tasks=1, input_dim=d_mp)

    model = CopolymerMPNN(
        message_passing=mp, agg=agg, predictor=ffn,
        copolymer_mode=f"stage2d_{VARIANT}", batch_norm=False,
    )

    map_loc = torch.device("cpu")
    ckpt = torch.load(str(ckpt_path), map_location=map_loc, weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    test_loader = build_dataloader(test_ds, batch_size=256, num_workers=0, shuffle=False)

    return model, test_loader, scaler, test_idx, test_ds


def _predict_loader(model: torch.nn.Module, test_loader) -> np.ndarray:
    """Run inference and return flat numpy array of predictions (unscaled)."""
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            # CopolymerTrainingBatch namedtuple: bmg_A, bmg_B, fracA, fracB, X_d, Y, w, lt_mask, gt_mask
            bmg_A  = batch.bmg_A
            bmg_B  = batch.bmg_B
            fracA  = batch.fracA
            fracB  = batch.fracB
            X_d    = batch.X_d
            out, _ = model.forward_stage2d(bmg_A, bmg_B, fracA, fracB, X_d)
            # Apply output transform (unscale) — same as during trainer.predict
            out_t = model.predictor.output_transform(out)
            preds.append(out_t.cpu().numpy())
    return np.concatenate(preds).flatten()


def _zero_alpha(model: torch.nn.Module) -> None:
    """Zero out all alpha parameters in the stage2 aggregator (in-place)."""
    agg = model.stage2_aggregator
    with torch.no_grad():
        if hasattr(agg, "alpha") and agg.alpha is not None:
            agg.alpha.zero_()


def _restore_alpha(model: torch.nn.Module, alpha_backup: torch.Tensor) -> None:
    with torch.no_grad():
        model.stage2_aggregator.alpha.copy_(alpha_backup)


def _compute_metrics(yt: np.ndarray, yp: np.ndarray, label: str) -> dict:
    r2  = float(r2_score(yt, yp))
    mae = float(mean_absolute_error(yt, yp))
    return {"label": label, "R2_overall": r2, "MAE_overall": mae}


def _group_metrics(fdf: pd.DataFrame, yp: np.ndarray, label: str) -> dict:
    """Group-mean and arch-deviation R² / MAE."""
    fdf = fdf.copy()
    fdf["y_pred"] = yp
    gm_true = fdf.groupby("group_key")["y_true"].transform("mean").values
    gm_pred = fdf.groupby("group_key")["y_pred"].transform("mean").values
    ad_true = fdf["y_true"].values - gm_true
    ad_pred = yp - gm_pred
    r2_gm  = float(r2_score(gm_true, gm_pred))
    mae_gm = float(mean_absolute_error(gm_true, gm_pred))
    r2_ad  = float(r2_score(ad_true, ad_pred))
    mae_ad = float(mean_absolute_error(ad_true, ad_pred))
    return {
        "label": label,
        "R2_group_mean": r2_gm, "MAE_group_mean": mae_gm,
        "R2_arch_dev": r2_ad,   "MAE_arch_dev": mae_ad,
    }


# ── main function ─────────────────────────────────────────────────────────────

def run_chemarch_residual_ablation(df: pd.DataFrame) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    meta_folds = load_split_meta(SPLIT)

    rows = []
    fold_summaries = {}  # target -> list of (fold, metrics_full, metrics_backbone)

    for tkey in TARGETS:
        fold_summaries[tkey] = []
        target_col = TARGETS[tkey]  # e.g. "EA vs SHE (eV)"

        for fold in range(N_FOLDS):
            print(f"  [ChemArch ablation] {tkey} fold {fold}...", end=" ", flush=True)

            result = _load_chemarch_model(target_col, fold, meta_folds, df)
            if result is None:
                print("SKIPPED (no checkpoint)")
                continue

            model, test_loader, scaler, test_idx, test_ds = result

            # y_true (unscaled — from raw dataset)
            yt = df.iloc[test_idx][target_col].values.astype(float)

            # Full prediction (backbone + residual)
            yp_full = _predict_loader(model, test_loader)
            yp_full = yp_full[:len(yt)]  # safety trim

            # Backbone-only: zero alpha, predict again
            alpha_backup = model.stage2_aggregator.alpha.data.clone()
            _zero_alpha(model)
            yp_backbone = _predict_loader(model, test_loader)
            yp_backbone = yp_backbone[:len(yt)]
            _restore_alpha(model, alpha_backup)

            # Build fold_df for group-mean analysis
            fdf = build_fold_df(df, yt, yp_full, test_idx)

            # Metrics
            m_full = _compute_metrics(yt, yp_full, "full")
            m_back = _compute_metrics(yt, yp_backbone, "backbone")
            gm_full = _group_metrics(fdf, yp_full, "full")
            fdf_back = build_fold_df(df, yt, yp_backbone, test_idx)
            gm_back = _group_metrics(fdf_back, yp_backbone, "backbone")

            print(f"R2_full={m_full['R2_overall']:.3f}  R2_backbone={m_back['R2_overall']:.3f}")

            row_base = dict(
                target=tkey, fold=fold,
                heldout_monomer=FOLD_MONOMER_NAMES.get(fold, f"fold{fold}"),
            )
            for mode, m, gm in [("full", m_full, gm_full),
                                  ("backbone_only", m_back, gm_back)]:
                rows.append({**row_base, "mode": mode,
                              "R2_overall":    m["R2_overall"],
                              "MAE_overall":   m["MAE_overall"],
                              "R2_group_mean": gm["R2_group_mean"],
                              "MAE_group_mean":gm["MAE_group_mean"],
                              "R2_arch_dev":   gm["R2_arch_dev"],
                              "MAE_arch_dev":  gm["MAE_arch_dev"]})

            fold_summaries[tkey].append((fold, yp_full, yp_backbone, yt, m_full, m_back))

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_DIR / "chemarch_pre_vs_post.csv", index=False)
    print(f"\n  Saved: chemarch_pre_vs_post.csv  ({len(out_df)} rows)")

    _make_ablation_plots(out_df, fold_summaries)

    return out_df


# ── plots ─────────────────────────────────────────────────────────────────────

def _make_ablation_plots(df: pd.DataFrame, fold_summaries: dict):
    """Bar charts showing backbone vs full for each fold, per target."""
    for tkey in TARGETS:
        sub = df[df["target"] == tkey]
        if len(sub) == 0:
            continue

        folds = sorted(sub["fold"].unique())
        n = len(folds)
        monomer_labels = [FOLD_MONOMER_NAMES.get(f, f"fold{f}")[:18] for f in folds]
        x = np.arange(n)
        width = 0.35

        for metric, ylabel in [("R2_overall", "R²"),
                                ("MAE_overall", "MAE (eV)"),
                                ("R2_group_mean", "R² (group mean)"),
                                ("R2_arch_dev", "R² (arch deviation)")]:
            fig, ax = plt.subplots(figsize=(12, 4))
            full_vals = [float(sub[(sub["fold"] == f) & (sub["mode"] == "full")][metric].values[0])
                         if len(sub[(sub["fold"] == f) & (sub["mode"] == "full")]) else np.nan
                         for f in folds]
            back_vals = [float(sub[(sub["fold"] == f) & (sub["mode"] == "backbone_only")][metric].values[0])
                         if len(sub[(sub["fold"] == f) & (sub["mode"] == "backbone_only")]) else np.nan
                         for f in folds]

            ax.bar(x - width/2, back_vals, width, label="Backbone only (α=0)", color="#aec7e8", edgecolor="k", lw=0.5)
            ax.bar(x + width/2, full_vals, width, label="Full ChemArch", color=COLORS["chemarch"], edgecolor="k", lw=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(monomer_labels, rotation=35, ha="right", fontsize=8)
            ax.set_xlabel("Held-out Monomer (fold)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"ChemArch: Backbone vs Full  —  {tkey}  —  {ylabel}")
            ax.legend(fontsize=9)
            if "R2" in metric:
                ax.axhline(0, color="k", lw=0.8, ls="--")

            plt.tight_layout()
            fname = f"chemarch_backbone_vs_full_{tkey.replace(' ', '_')}_{metric}.png"
            fig.savefig(OUT_DIR / fname, dpi=DPI)
            plt.close(fig)

    # Summary delta plot: full - backbone
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, tkey in zip(axes, TARGETS):
        sub = df[df["target"] == tkey]
        if len(sub) == 0:
            continue
        folds = sorted(sub["fold"].unique())
        for metric, ls, color in [
            ("R2_overall",    "-",  "black"),
            ("R2_group_mean", "--", "royalblue"),
            ("R2_arch_dev",   ":",  "darkorange"),
        ]:
            deltas = []
            for f in folds:
                full = sub[(sub["fold"] == f) & (sub["mode"] == "full")][metric]
                back = sub[(sub["fold"] == f) & (sub["mode"] == "backbone_only")][metric]
                if len(full) and len(back):
                    deltas.append(float(full.values[0]) - float(back.values[0]))
                else:
                    deltas.append(np.nan)
            ax.plot(folds, deltas, marker="o", ls=ls, color=color,
                    label=f"ΔR² {metric.replace('R2_', '')}")
        ax.axhline(0, color="gray", lw=0.8, ls=":")
        ax.set_xlabel("Fold")
        ax.set_ylabel("ΔR² (full − backbone)")
        ax.set_title(f"{tkey}: residual contribution ΔR²")
        ax.legend(fontsize=8)
    plt.suptitle("ChemArch: Change in R² from adding the residual head (α·r_arch)", fontsize=10)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "chemarch_delta_r2.png", dpi=DPI)
    plt.close(fig)

    print(f"  Saved: ablation comparison plots in {OUT_DIR}")
