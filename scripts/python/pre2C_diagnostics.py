#!/usr/bin/env python3
"""Pre-2C diagnostics for HPG-2Stage typed interaction feasibility.

Tests whether typed interaction decomposition (BB/BF/FF) is statistically
justified on the EA/IP monomer-disjoint dataset.

Diagnostics:
  1. Per-fold interaction benefit (Frac vs Interact-learned)
  2. BB/BF/FF channel identifiability from monomer SMILES
  3. Incremental residual explanation beyond composition

Usage:
    python scripts/python/pre2C_diagnostics.py

Outputs saved to: pre2C_diagnostics/
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = ROOT / "data" / "ea_ip.csv"
SPLITS_JSON = ROOT / "results" / "splits" / "ea_ip_aheldout_seed42.json"
PREDICTIONS_DIR = ROOT / "predictions" / "DMPNN"
OUTPUT_DIR = ROOT / "pre2C_diagnostics"

TARGETS = ["EA vs SHE (eV)", "IP vs SHE (eV)"]
TARGET_SHORT = {"EA vs SHE (eV)": "EA", "IP vs SHE (eV)": "IP"}
N_FOLDS = 5

# Prediction file patterns
PRED_FRAC_PATTERN = "ea_ip__{target}__copoly_mix_meta__poly_type__a_held_out__split{fold}.npz"
PRED_INTERACT_PATTERN = "ea_ip__{target}__copoly_mix_pair_meta__fusion_scalar_residual_fusion__poly_type__a_held_out__split{fold}.npz"

# Plot style
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ══════════════════════════════════════════════════════════════════════════════
# Utility functions
# ══════════════════════════════════════════════════════════════════════════════

def load_inputs() -> Tuple[pd.DataFrame, Dict]:
    """Load dataset CSV and fold assignments."""
    df = pd.read_csv(DATA_CSV)
    with open(SPLITS_JSON) as f:
        splits = json.load(f)
    return df, splits


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def load_predictions(pattern: str, target: str, fold: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load y_true and y_pred from a prediction .npz file."""
    fname = pattern.format(target=target, fold=fold)
    path = PREDICTIONS_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    data = np.load(path, allow_pickle=True)
    y_true = data["y_true"].squeeze()
    y_pred = data["y_pred"].squeeze()
    return y_true, y_pred


# ══════════════════════════════════════════════════════════════════════════════
# Diagnostic 1 — Per-fold interaction benefit
# ══════════════════════════════════════════════════════════════════════════════

def diagnostic1_interaction_benefit() -> pd.DataFrame:
    """Compute per-fold RMSE difference between Frac and Interact-learned.

    ΔRMSE_k = RMSE_Frac_k − RMSE_InteractLearned_k
    Positive = interaction helps.
    """
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 1 — Per-fold interaction benefit")
    print("=" * 70)

    rows = []
    for target in TARGETS:
        short = TARGET_SHORT[target]
        for fold in range(N_FOLDS):
            yt_frac, yp_frac = load_predictions(PRED_FRAC_PATTERN, target, fold)
            yt_int, yp_int = load_predictions(PRED_INTERACT_PATTERN, target, fold)

            rmse_frac = compute_rmse(yt_frac, yp_frac)
            rmse_int = compute_rmse(yt_int, yp_int)
            delta = rmse_frac - rmse_int

            rows.append({
                "target": short,
                "fold": fold,
                "RMSE_Frac": rmse_frac,
                "RMSE_Interact": rmse_int,
                "delta_RMSE": delta,
            })

    df_rmse = pd.DataFrame(rows)

    # Save fold-wise RMSE
    df_rmse.to_csv(OUTPUT_DIR / "diagnostic1_fold_rmse.csv", index=False)

    # Summary per target
    summary_rows = []
    for short in ["EA", "IP"]:
        sub = df_rmse[df_rmse["target"] == short]
        deltas = sub["delta_RMSE"].values
        win_rate = float(np.mean(deltas > 0))
        summary_rows.append({
            "target": short,
            "mean_delta_RMSE": deltas.mean(),
            "std_delta_RMSE": deltas.std(),
            "win_rate": win_rate,
        })
        print(f"  {short}: mean ΔRMSE = {deltas.mean():.6f} ± {deltas.std():.6f}, "
              f"win rate = {win_rate:.0%} ({int(win_rate * N_FOLDS)}/{N_FOLDS} folds)")

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUTPUT_DIR / "diagnostic1_delta_rmse.csv", index=False)

    # Plot 1: ΔRMSE per fold
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, short in zip(axes, ["EA", "IP"]):
        sub = df_rmse[df_rmse["target"] == short]
        colors = ["#66bb6a" if d > 0 else "#ef5350" for d in sub["delta_RMSE"]]
        ax.bar(sub["fold"], sub["delta_RMSE"], color=colors, edgecolor="white", width=0.6)
        ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Fold")
        ax.set_ylabel("ΔRMSE (Frac − Interact)" if ax == axes[0] else "")
        ax.set_title(f"{short} — per-fold interaction benefit", fontweight="bold")
        ax.set_xticks(range(N_FOLDS))
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "diagnostic1_delta_rmse_by_fold.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: Summary bar
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(2)
    means = [df_summary.loc[df_summary["target"] == t, "mean_delta_RMSE"].values[0] for t in ["EA", "IP"]]
    stds = [df_summary.loc[df_summary["target"] == t, "std_delta_RMSE"].values[0] for t in ["EA", "IP"]]
    colors = ["#42a5f5", "#ef5350"]
    ax.bar(x, means, yerr=stds, color=colors, edgecolor="white", width=0.5, capsize=5)
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(["EA", "IP"])
    ax.set_ylabel("Mean ΔRMSE (Frac − Interact)")
    ax.set_title("Interaction benefit summary", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "diagnostic1_delta_rmse_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return df_rmse


# ══════════════════════════════════════════════════════════════════════════════
# Diagnostic 2 — BB/BF/FF channel identifiability
# ══════════════════════════════════════════════════════════════════════════════

def assign_backbone_functional_atoms(smiles: str) -> Tuple[Optional[int], Optional[int]]:
    """Assign atoms to backbone vs functional groups using a simple heuristic.

    NOTE: This is a DIAGNOSTIC PROXY only — not a final chemical decomposition.
    The heuristic identifies ring-system atoms as "backbone" and non-ring atoms
    as "functional group" atoms. This is a coarse approximation suitable for
    testing whether typed decomposition has statistical signal, not for drawing
    definitive chemical conclusions.

    Returns (n_backbone, n_functional) or (None, None) on parse failure.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    # Count heavy atoms (exclude H and dummy atoms [*])
    heavy_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() > 0 and a.GetAtomicNum() != 0]
    # Also exclude wildcard/dummy atoms (atomic num 0)
    heavy_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() > 0]
    n_total = len(heavy_atoms)

    if n_total == 0:
        return None, None

    # Identify ring atoms as backbone
    ring_info = mol.GetRingInfo()
    ring_atom_set = set()
    for ring in ring_info.AtomRings():
        ring_atom_set.update(ring)

    # Filter to heavy atoms only
    heavy_idx = {a.GetIdx() for a in heavy_atoms}
    backbone_idx = ring_atom_set & heavy_idx

    if len(backbone_idx) == 0:
        # No rings: use longest chain of heavy atoms as backbone proxy
        # Simple heuristic: atoms in the longest path
        from rdkit.Chem import rdmolops
        try:
            # Get the longest path in the molecular graph
            # Use BFS from each atom to find the longest shortest path
            adj = Chem.GetAdjacencyMatrix(mol)
            # Only consider heavy atoms
            heavy_list = sorted(heavy_idx)
            if len(heavy_list) <= 1:
                return n_total, 0
            # Simple: treat half as backbone (arbitrary for acyclic monomers)
            n_backbone = max(1, n_total // 2)
            n_functional = n_total - n_backbone
            return n_backbone, n_functional
        except Exception:
            return n_total, 0

    n_backbone = len(backbone_idx)
    n_functional = n_total - n_backbone

    return n_backbone, n_functional


def compute_monomer_channel_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute backbone/functional proportions for each unique monomer."""
    all_monomers = set(df["smiles_A"].unique()) | set(df["smiles_B"].unique())
    rows = []
    for smi in sorted(all_monomers):
        n_bb, n_fn = assign_backbone_functional_atoms(smi)
        if n_bb is None:
            rows.append({"smiles": smi, "n_backbone": np.nan, "n_functional": np.nan,
                         "p_backbone": np.nan, "p_functional": np.nan})
        else:
            total = n_bb + n_fn
            rows.append({
                "smiles": smi,
                "n_backbone": n_bb,
                "n_functional": n_fn,
                "p_backbone": n_bb / total if total > 0 else 0.0,
                "p_functional": n_fn / total if total > 0 else 0.0,
            })
    return pd.DataFrame(rows)


def compute_polymer_channel_features(df: pd.DataFrame, monomer_stats: pd.DataFrame) -> pd.DataFrame:
    """Compute BB/BF/FF channel proportions for each polymer."""
    # Build lookup
    lookup = monomer_stats.set_index("smiles")

    pB_A = df["smiles_A"].map(lookup["p_backbone"]).values
    pF_A = df["smiles_A"].map(lookup["p_functional"]).values
    pB_B = df["smiles_B"].map(lookup["p_backbone"]).values
    pF_B = df["smiles_B"].map(lookup["p_functional"]).values

    fracA = df["fracA"].values
    fracB = df["fracB"].values

    pB_poly = fracA * pB_A + fracB * pB_B
    pF_poly = fracA * pF_A + fracB * pF_B

    X_BB = pB_poly ** 2
    X_BF = 2 * pB_poly * pF_poly
    X_FF = pF_poly ** 2

    return pd.DataFrame({
        "X_BB": X_BB,
        "X_BF": X_BF,
        "X_FF": X_FF,
        "pB_poly": pB_poly,
        "pF_poly": pF_poly,
    })


def compute_vif(X: np.ndarray) -> np.ndarray:
    """Compute Variance Inflation Factors for each column of X."""
    n_cols = X.shape[1]
    vifs = np.zeros(n_cols)
    for i in range(n_cols):
        y_i = X[:, i]
        X_others = np.delete(X, i, axis=1)
        reg = LinearRegression().fit(X_others, y_i)
        r2 = reg.score(X_others, y_i)
        vifs[i] = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
    return vifs


def diagnostic2_identifiability(df: pd.DataFrame) -> Dict:
    """Test whether BB/BF/FF channels are identifiable."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 2 — BB/BF/FF channel identifiability")
    print("=" * 70)

    # Monomer-level stats
    monomer_stats = compute_monomer_channel_stats(df)
    monomer_stats.to_csv(OUTPUT_DIR / "diagnostic2_monomer_channel_stats.csv", index=False)
    print(f"  Unique monomers: {len(monomer_stats)}")
    print(f"  Parse failures: {monomer_stats['p_backbone'].isna().sum()}")

    # Polymer-level channel features
    chan_df = compute_polymer_channel_features(df, monomer_stats)
    chan_df.to_csv(OUTPUT_DIR / "diagnostic2_polymer_channel_features.csv", index=False)

    # Drop NaN rows
    valid_mask = chan_df.notna().all(axis=1)
    chan_valid = chan_df[valid_mask].values  # [X_BB, X_BF, X_FF, pB, pF]
    X_raw = chan_valid[:, :3]  # [X_BB, X_BF, X_FF]
    print(f"  Valid polymers: {valid_mask.sum()} / {len(df)}")

    # Correlation matrix
    corr = np.corrcoef(X_raw.T)
    print(f"\n  Correlation matrix [X_BB, X_BF, X_FF]:")
    for i, name in enumerate(["X_BB", "X_BF", "X_FF"]):
        print(f"    {name}: [{corr[i, 0]:.4f}, {corr[i, 1]:.4f}, {corr[i, 2]:.4f}]")

    # Matrix rank of raw 3-channel
    rank_raw = np.linalg.matrix_rank(X_raw, tol=1e-10)
    print(f"\n  Rank of raw [X_BB, X_BF, X_FF]: {rank_raw}")

    # Reparameterized: drop X_FF
    X_reparam = X_raw[:, :2]  # [X_BB, X_BF]
    cond_number = np.linalg.cond(X_reparam)
    print(f"  Condition number of [X_BB, X_BF]: {cond_number:.2f}")

    # VIF
    vifs = compute_vif(X_reparam)
    print(f"  VIF(X_BB) = {vifs[0]:.2f}")
    print(f"  VIF(X_BF) = {vifs[1]:.2f}")

    # Channel variance/range
    summary_rows = []
    for i, name in enumerate(["X_BB", "X_BF", "X_FF"]):
        col = X_raw[:, i]
        summary_rows.append({
            "channel": name,
            "mean": col.mean(),
            "std": col.std(),
            "min": col.min(),
            "max": col.max(),
            "variance": col.var(),
        })
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUTPUT_DIR / "diagnostic2_channel_summary.csv", index=False)
    print(f"\n  Channel statistics:")
    print(df_summary.to_string(index=False))

    # Warnings
    if cond_number > 30:
        print(f"\n  ⚠️  WARNING: Condition number {cond_number:.1f} > 30 — potential multicollinearity")
    if any(v > 5 for v in vifs):
        print(f"\n  ⚠️  WARNING: VIF > 5 detected — potential multicollinearity")

    # ── Plots ──

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        pd.DataFrame(corr, index=["X_BB", "X_BF", "X_FF"], columns=["X_BB", "X_BF", "X_FF"]),
        annot=True, fmt=".3f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax,
        square=True, linewidths=0.5,
    )
    ax.set_title("Channel correlation matrix", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "diagnostic2_correlation_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Histograms
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, (i, name) in zip(axes, enumerate(["X_BB", "X_BF", "X_FF"])):
        ax.hist(X_raw[:, i], bins=50, color=["#42a5f5", "#66bb6a", "#ef5350"][i],
                edgecolor="white", alpha=0.85)
        ax.set_xlabel(name)
        ax.set_ylabel("Count" if i == 0 else "")
        ax.set_title(name, fontweight="bold")
    fig.suptitle("Channel feature distributions (diagnostic proxy)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "diagnostic2_channel_histograms.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # PCA scatter
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_raw)

    poly_types = df.loc[valid_mask, "poly_type"].values
    unique_types = sorted(set(poly_types))
    cmap = plt.colormaps.get_cmap("tab10").resampled(len(unique_types))

    fig, ax = plt.subplots(figsize=(7, 5))
    for i_t, pt in enumerate(unique_types):
        mask = poly_types == pt
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=8, alpha=0.4,
                   color=cmap(i_t), label=pt)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title("PCA of [X_BB, X_BF, X_FF] colored by poly_type", fontweight="bold")
    ax.legend(fontsize=8, markerscale=2, framealpha=0.8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "diagnostic2_pca_channel_space.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "rank_raw": rank_raw,
        "cond_number": cond_number,
        "vif_BB": vifs[0],
        "vif_BF": vifs[1],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Diagnostic 3 — Incremental residual explanation beyond composition
# ══════════════════════════════════════════════════════════════════════════════

def diagnostic3_incremental_residual_explanation(
    df: pd.DataFrame, splits: Dict
) -> pd.DataFrame:
    """Test whether channel features explain Frac residuals beyond composition."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 3 — Incremental residual explanation beyond composition")
    print("=" * 70)

    # Compute monomer + polymer channel features
    monomer_stats = compute_monomer_channel_stats(df)
    chan_df = compute_polymer_channel_features(df, monomer_stats)

    # Build poly_type dummies
    poly_type_dummies = pd.get_dummies(df["poly_type"], prefix="pt", drop_first=True)

    # Reconstruct residuals from predictions using fold assignments
    residuals = {short: np.full(len(df), np.nan) for short in ["EA", "IP"]}
    predictions_frac = {short: np.full(len(df), np.nan) for short in ["EA", "IP"]}

    for target in TARGETS:
        short = TARGET_SHORT[target]
        for fold in range(N_FOLDS):
            test_idx = np.array(splits["folds"][fold]["test_indices"])
            yt, yp = load_predictions(PRED_FRAC_PATTERN, target, fold)
            # Residual = true - predicted (from Frac model)
            res = yt - yp
            residuals[short][test_idx] = res
            predictions_frac[short][test_idx] = yp

    # Build residual dataset
    res_df = pd.DataFrame({
        "fracA": df["fracA"].values,
        "X_BB": chan_df["X_BB"].values,
        "X_BF": chan_df["X_BF"].values,
        "X_FF": chan_df["X_FF"].values,
        "residual_EA": residuals["EA"],
        "residual_IP": residuals["IP"],
        "poly_type": df["poly_type"].values,
    })

    # Add poly_type dummies
    for col in poly_type_dummies.columns:
        res_df[col] = poly_type_dummies[col].values

    # Drop rows with missing residuals or channel features
    valid_mask = res_df[["fracA", "X_BB", "X_BF", "residual_EA", "residual_IP"]].notna().all(axis=1)
    res_valid = res_df[valid_mask].reset_index(drop=True)
    res_valid.to_csv(OUTPUT_DIR / "diagnostic3_residual_dataset.csv", index=False)
    print(f"  Valid samples for regression: {len(res_valid)} / {len(df)}")

    # Assign each valid sample to its fold
    fold_assignment = np.full(len(df), -1, dtype=int)
    for fold in range(N_FOLDS):
        test_idx = np.array(splits["folds"][fold]["test_indices"])
        fold_assignment[test_idx] = fold
    fold_valid = fold_assignment[valid_mask.values]

    # Fit OLS per fold and overall
    pt_cols = [c for c in res_valid.columns if c.startswith("pt_")]

    results_rows = []
    coeff_rows = []

    for short in ["EA", "IP"]:
        y_col = f"residual_{short}"

        for fold in range(N_FOLDS):
            fold_mask = fold_valid == fold
            if fold_mask.sum() < 10:
                continue

            y = res_valid.loc[fold_mask, y_col].values

            # M1: residual ~ fracA
            X_m1 = res_valid.loc[fold_mask, ["fracA"]].values
            reg_m1 = LinearRegression().fit(X_m1, y)
            r2_m1 = reg_m1.score(X_m1, y)

            # M2: residual ~ fracA + X_BB + X_BF
            X_m2 = res_valid.loc[fold_mask, ["fracA", "X_BB", "X_BF"]].values
            reg_m2 = LinearRegression().fit(X_m2, y)
            r2_m2 = reg_m2.score(X_m2, y)

            delta_r2 = r2_m2 - r2_m1

            results_rows.append({
                "target": short,
                "fold": fold,
                "R2_M1": r2_m1,
                "R2_M2": r2_m2,
                "delta_R2": delta_r2,
            })

            # Coefficients for channel features
            coeff_rows.append({
                "target": short,
                "fold": fold,
                "coeff_fracA": reg_m2.coef_[0],
                "coeff_X_BB": reg_m2.coef_[1],
                "coeff_X_BF": reg_m2.coef_[2],
                "intercept": reg_m2.intercept_,
            })

    df_r2 = pd.DataFrame(results_rows)
    df_r2.to_csv(OUTPUT_DIR / "diagnostic3_incremental_r2.csv", index=False)

    df_coeff = pd.DataFrame(coeff_rows)
    df_coeff.to_csv(OUTPUT_DIR / "diagnostic3_coefficients.csv", index=False)

    # Print summary
    for short in ["EA", "IP"]:
        sub = df_r2[df_r2["target"] == short]
        dr2 = sub["delta_R2"].values
        print(f"  {short}: mean ΔR² = {dr2.mean():.6f} ± {dr2.std():.6f}")
        print(f"       R²(M1) = {sub['R2_M1'].mean():.6f}, R²(M2) = {sub['R2_M2'].mean():.6f}")

        # Coefficient sign stability
        sub_c = df_coeff[df_coeff["target"] == short]
        bb_signs = np.sign(sub_c["coeff_X_BB"].values)
        bf_signs = np.sign(sub_c["coeff_X_BF"].values)
        print(f"       X_BB sign: {'+' if bb_signs.mean() > 0 else '-'} "
              f"(stable in {int(np.abs(bb_signs.sum()))}/{N_FOLDS} folds)")
        print(f"       X_BF sign: {'+' if bf_signs.mean() > 0 else '-'} "
              f"(stable in {int(np.abs(bf_signs.sum()))}/{N_FOLDS} folds)")

    # ── Plots ──

    # Plot 1: ΔR² by fold
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, short in zip(axes, ["EA", "IP"]):
        sub = df_r2[df_r2["target"] == short]
        colors = ["#66bb6a" if d > 0 else "#ef5350" for d in sub["delta_R2"]]
        ax.bar(sub["fold"], sub["delta_R2"], color=colors, edgecolor="white", width=0.6)
        ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Fold")
        ax.set_ylabel("ΔR² (M2 − M1)" if ax == axes[0] else "")
        ax.set_title(f"{short} — incremental R² from channels", fontweight="bold")
        ax.set_xticks(range(N_FOLDS))
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "diagnostic3_delta_r2_by_fold.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: Coefficient plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, short in zip(axes, ["EA", "IP"]):
        sub_c = df_coeff[df_coeff["target"] == short]
        x = np.arange(N_FOLDS)
        w = 0.35
        ax.bar(x - w / 2, sub_c["coeff_X_BB"], width=w, color="#42a5f5",
               edgecolor="white", label="X_BB")
        ax.bar(x + w / 2, sub_c["coeff_X_BF"], width=w, color="#66bb6a",
               edgecolor="white", label="X_BF")
        ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Fold")
        ax.set_ylabel("OLS coefficient" if ax == axes[0] else "")
        ax.set_title(f"{short} — channel coefficients", fontweight="bold")
        ax.set_xticks(x)
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "diagnostic3_channel_coefficients.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Plot 3 & 4: Scatter of strongest channel vs residual
    for short in ["EA", "IP"]:
        y_col = f"residual_{short}"
        # Determine strongest channel by mean |coefficient|
        sub_c = df_coeff[df_coeff["target"] == short]
        mean_abs_bb = sub_c["coeff_X_BB"].abs().mean()
        mean_abs_bf = sub_c["coeff_X_BF"].abs().mean()
        strongest = "X_BB" if mean_abs_bb > mean_abs_bf else "X_BF"

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(res_valid[strongest], res_valid[y_col], s=4, alpha=0.2, color="#42a5f5")
        ax.set_xlabel(strongest)
        ax.set_ylabel(f"Residual ({short})")
        ax.set_title(f"{short} residual vs {strongest} (diagnostic proxy)", fontweight="bold")

        # Add regression line
        x_plot = res_valid[strongest].values.reshape(-1, 1)
        y_plot = res_valid[y_col].values
        valid_xy = ~(np.isnan(x_plot.squeeze()) | np.isnan(y_plot))
        if valid_xy.sum() > 10:
            reg = LinearRegression().fit(x_plot[valid_xy], y_plot[valid_xy])
            xs = np.linspace(x_plot[valid_xy].min(), x_plot[valid_xy].max(), 100)
            ax.plot(xs, reg.predict(xs.reshape(-1, 1)), "r-", linewidth=2, alpha=0.8)

        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"diagnostic3_residual_scatter_{short}.png",
                    dpi=200, bbox_inches="tight")
        plt.close(fig)

    return df_r2


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading inputs...")
    df, splits = load_inputs()
    print(f"  Dataset: {len(df)} samples, {df['smiles_A'].nunique()} unique A, "
          f"{df['smiles_B'].nunique()} unique B")

    # Diagnostic 1
    df_rmse = diagnostic1_interaction_benefit()

    # Diagnostic 2
    d2_results = diagnostic2_identifiability(df)

    # Diagnostic 3
    df_r2 = diagnostic3_incremental_residual_explanation(df, splits)

    # ══════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("FINAL SUMMARY")
    print("═" * 70)

    # Diagnostic 1 summary
    print("\nDiagnostic 1 — Interaction benefit:")
    for short in ["EA", "IP"]:
        sub = df_rmse[df_rmse["target"] == short]
        deltas = sub["delta_RMSE"].values
        wr = float(np.mean(deltas > 0))
        print(f"  {short}: win rate = {wr:.0%}, mean ΔRMSE = {deltas.mean():.6f}")

    # Diagnostic 2 summary
    print(f"\nDiagnostic 2 — Channel identifiability:")
    print(f"  Raw [X_BB, X_BF, X_FF] rank: {d2_results['rank_raw']}")
    print(f"  Condition number [X_BB, X_BF]: {d2_results['cond_number']:.2f}")
    print(f"  VIF(X_BB) = {d2_results['vif_BB']:.2f}, VIF(X_BF) = {d2_results['vif_BF']:.2f}")
    if d2_results['cond_number'] > 30:
        print("  ⚠️  Condition number > 30 — multicollinearity warning")
    if d2_results['vif_BB'] > 5 or d2_results['vif_BF'] > 5:
        print("  ⚠️  VIF > 5 — multicollinearity warning")

    # Diagnostic 3 summary
    print(f"\nDiagnostic 3 — Incremental R²:")
    for short in ["EA", "IP"]:
        sub = df_r2[df_r2["target"] == short]
        dr2 = sub["delta_R2"].values
        print(f"  {short}: mean ΔR² = {dr2.mean():.6f} ± {dr2.std():.6f}")
    if all(df_r2.groupby("target")["delta_R2"].mean() < 0.001):
        print("  ⚠️  ΔR² near zero — channels may not add value beyond composition")

    # Decision rule
    print("\n" + "─" * 70)
    print("DECISION RULE:")
    print("─" * 70)

    ip_deltas = df_rmse[df_rmse["target"] == "IP"]["delta_RMSE"].values
    ip_wr = float(np.mean(ip_deltas > 0))
    cond_ok = d2_results["cond_number"] < 30
    vif_ok = d2_results["vif_BB"] < 5 and d2_results["vif_BF"] < 5
    ip_dr2 = df_r2[df_r2["target"] == "IP"]["delta_R2"].mean()

    if ip_wr >= 0.6 and (cond_ok or vif_ok) and ip_dr2 > 0.001:
        print("  ✅ PROCEED with full Stage 2C implementation")
        print("     - IP interaction benefit positive in most folds")
        print("     - Channel design matrix identifiable after reparameterization")
        print("     - Channels explain residual variance beyond composition")
    elif ip_wr >= 0.4 or ip_dr2 > 0:
        print("  ⚡ Consider MINIMAL 2C ABLATION")
        print("     - Diagnostics are mixed; run a small-scale test first")
    else:
        print("  ❌ SKIP/DEPRIORITIZE Stage 2C")
        print("     - Interaction benefit inconsistent")
        print("     - Design matrix ill-conditioned or channels add no value")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
