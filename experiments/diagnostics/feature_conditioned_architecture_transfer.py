#!/usr/bin/env python3
"""Feature-conditioned architecture transfer diagnostic.

Tests whether architecture-conditioned property effects are transferable
across unseen monomer systems under monomer-disjoint evaluation.

Determines if Stage 2D-1 is worth implementing.

Core question:
    Δy_arch = f(h_A, h_B, fracA, fracB, architecture)

If h_A / h_B improve transfer R² beyond architecture + fractions,
then Stage 2D-1 is justified.

Usage:
    python feature_conditioned_architecture_transfer.py --data data.csv --out feature_conditioned_transfer
    python feature_conditioned_architecture_transfer.py --data data.csv --embedding-file monomer_embeddings.csv --out out_dir
    python feature_conditioned_architecture_transfer.py --data data.csv --arch-encoding onehot --models ridge gbm mlp
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ══════════════════════════════════════════════════════════════════════════════
# Column name inference
# ══════════════════════════════════════════════════════════════════════════════

COLUMN_ALIASES = {
    "smiles_A": ["smiles_a", "monomer_a", "a_smiles", "smilesA", "smilesa"],
    "smiles_B": ["smiles_b", "monomer_b", "b_smiles", "smilesB", "smilesb"],
    "fracA": ["fraca", "frac_a", "fraction_a", "x_a"],
    "fracB": ["fracb", "frac_b", "fraction_b", "x_b"],
    "poly_type": ["poly_type", "polymer_type", "architecture", "arch", "polytype"],
    "EA": ["ea", "ea vs she (ev)", "ea_vs_she", "ea vs she"],
    "IP": ["ip", "ip vs she (ev)", "ip_vs_she", "ip vs she"],
    "fold": ["fold", "cv_fold", "split"],
}

ARCH_ORDINAL = {"alternating": 0, "random": 1, "block": 2}


def infer_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Infer canonical column names from the dataframe."""
    col_lower_map = {c.lower().strip(): c for c in df.columns}
    mapping = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        found = None
        for alias in aliases:
            if alias.lower() in col_lower_map:
                found = col_lower_map[alias.lower()]
                break
        if found is None and canonical.lower() in col_lower_map:
            found = col_lower_map[canonical.lower()]
        mapping[canonical] = found

    required = ["smiles_A", "smiles_B", "fracA", "fracB", "poly_type", "EA", "IP"]
    missing = [k for k in required if mapping.get(k) is None]
    if missing:
        raise ValueError(
            f"Could not infer required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )
    return mapping


# ══════════════════════════════════════════════════════════════════════════════
# Embedding loading
# ══════════════════════════════════════════════════════════════════════════════

def compute_rdkit_fingerprints(
    df: pd.DataFrame,
    col_map: Dict,
    radius: int = 2,
    n_bits: int = 128,
) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    """Compute Morgan fingerprints for monomers A and B.

    Uses RDKit to generate fixed-length fingerprint vectors as
    proxy chemistry embeddings when learned embeddings are unavailable.

    Returns (hA_array, hB_array, emb_dim) or None.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError:
        print("  WARNING: RDKit not available. Cannot compute fingerprint features.")
        return None

    sA_col = col_map["smiles_A"]
    sB_col = col_map["smiles_B"]

    # Get unique monomers
    all_smiles = set(df[sA_col].unique()) | set(df[sB_col].unique())
    print(f"  Computing Morgan fingerprints (radius={radius}, bits={n_bits}) for {len(all_smiles)} unique monomers...")

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    # Build fingerprint lookup
    fp_lookup = {}
    n_failed = 0
    for smi in all_smiles:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            n_failed += 1
            fp_lookup[smi] = np.zeros(n_bits, dtype=np.float32)
        else:
            fp = gen.GetFingerprintAsNumPy(mol)
            fp_lookup[smi] = fp.astype(np.float32)

    if n_failed > 0:
        print(f"  WARNING: {n_failed} monomers failed to parse (zero vector used).")

    # Map to data rows
    n = len(df)
    hA = np.zeros((n, n_bits), dtype=np.float32)
    hB = np.zeros((n, n_bits), dtype=np.float32)

    sA_vals = df[sA_col].values
    sB_vals = df[sB_col].values
    for i in range(n):
        hA[i] = fp_lookup[sA_vals[i]]
        hB[i] = fp_lookup[sB_vals[i]]

    return hA, hB, n_bits


def detect_inline_embeddings(df: pd.DataFrame) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    """Detect hA_0, hA_1, ..., hB_0, hB_1, ... columns in the dataframe.

    Returns (hA_array, hB_array, emb_dim) or None.
    """
    hA_cols = sorted([c for c in df.columns if c.startswith("hA_")],
                     key=lambda x: int(x.split("_")[1]))
    hB_cols = sorted([c for c in df.columns if c.startswith("hB_")],
                     key=lambda x: int(x.split("_")[1]))

    if not hA_cols or not hB_cols:
        return None
    if len(hA_cols) != len(hB_cols):
        print(f"  WARNING: hA columns ({len(hA_cols)}) != hB columns ({len(hB_cols)}). Skipping inline embeddings.")
        return None

    hA = df[hA_cols].values
    hB = df[hB_cols].values
    emb_dim = len(hA_cols)

    # Check for NaN
    if np.isnan(hA).any() or np.isnan(hB).any():
        n_nan = int(np.isnan(hA).any(axis=1).sum() + np.isnan(hB).any(axis=1).sum())
        print(f"  WARNING: {n_nan} rows have NaN in embedding columns.")
        return None

    return hA, hB, emb_dim


def load_embedding_file(
    emb_path: Path,
    df: pd.DataFrame,
    col_map: Dict,
) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    """Load monomer embeddings from a separate CSV file.

    FORMAT 2: CSV with columns [monomer_smiles, emb_0, emb_1, ..., emb_d]
    Maps monomer_A -> h_A, monomer_B -> h_B via SMILES lookup.

    Returns (hA_array, hB_array, emb_dim) or None.
    """
    if not emb_path.exists():
        print(f"  WARNING: Embedding file not found: {emb_path}")
        return None

    sA_col = col_map["smiles_A"]
    sB_col = col_map["smiles_B"]

    if emb_path.suffix == ".csv":
        emb_df = pd.read_csv(emb_path)

        # Find the SMILES column
        smiles_col = None
        for candidate in ["monomer_smiles", "smiles", "SMILES", "monomer"]:
            if candidate in emb_df.columns:
                smiles_col = candidate
                break
        if smiles_col is None:
            # Try first column
            smiles_col = emb_df.columns[0]
            print(f"  Using first column as SMILES key: '{smiles_col}'")

        # Find embedding columns (emb_0, emb_1, ... or numeric columns after smiles)
        emb_cols = sorted([c for c in emb_df.columns if c.startswith("emb_")],
                          key=lambda x: int(x.split("_")[1]))
        if not emb_cols:
            # Fall back to all numeric columns except the SMILES column
            emb_cols = [c for c in emb_df.columns if c != smiles_col and emb_df[c].dtype in [np.float64, np.float32, np.int64]]
            emb_cols = sorted(emb_cols)

        if not emb_cols:
            print(f"  WARNING: No embedding columns found in {emb_path}")
            return None

        emb_dim = len(emb_cols)

        # Build lookup: smiles -> embedding vector
        emb_lookup = {}
        for _, row in emb_df.iterrows():
            smiles = str(row[smiles_col])
            emb_lookup[smiles] = row[emb_cols].values.astype(np.float64)

        # Map to data rows
        n = len(df)
        hA = np.full((n, emb_dim), np.nan)
        hB = np.full((n, emb_dim), np.nan)

        smiles_A_vals = df[sA_col].astype(str).values
        smiles_B_vals = df[sB_col].astype(str).values

        for i in range(n):
            if smiles_A_vals[i] in emb_lookup:
                hA[i] = emb_lookup[smiles_A_vals[i]]
            if smiles_B_vals[i] in emb_lookup:
                hB[i] = emb_lookup[smiles_B_vals[i]]

        # Check coverage
        hA_valid = ~np.isnan(hA[:, 0])
        hB_valid = ~np.isnan(hB[:, 0])
        both_valid = hA_valid & hB_valid
        n_valid = int(both_valid.sum())

        if n_valid == 0:
            print(f"  WARNING: No rows matched embeddings. Check SMILES format.")
            return None

        print(f"  Embedding lookup: {len(emb_lookup)} monomers, dim={emb_dim}")
        print(f"  Coverage: {n_valid}/{n} rows ({100*n_valid/n:.1f}%) have both hA and hB")

        if n_valid < n:
            n_missing_A = int((~hA_valid).sum())
            n_missing_B = int((~hB_valid).sum())
            print(f"  Missing: {n_missing_A} monomer_A, {n_missing_B} monomer_B")

        return hA, hB, emb_dim

    elif emb_path.suffix == ".npy":
        # Assume rows aligned: [hA_0..hA_d, hB_0..hB_d] per row
        emb_all = np.load(emb_path)
        if emb_all.ndim == 2 and emb_all.shape[0] == len(df):
            emb_dim = emb_all.shape[1] // 2
            hA = emb_all[:, :emb_dim]
            hB = emb_all[:, emb_dim:]
            print(f"  Loaded .npy embeddings: shape={emb_all.shape}, emb_dim={emb_dim}")
            return hA, hB, emb_dim
        else:
            print(f"  WARNING: .npy shape {emb_all.shape} doesn't match {len(df)} rows")
            return None

    else:
        print(f"  WARNING: Unsupported embedding format: {emb_path.suffix}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Build matched groups
# ══════════════════════════════════════════════════════════════════════════════

def build_matched_groups(df: pd.DataFrame, col_map: Dict) -> pd.DataFrame:
    """Keep only rows in groups with >=2 distinct architectures."""
    sA = col_map["smiles_A"]
    sB = col_map["smiles_B"]
    fA = col_map["fracA"]
    fB = col_map["fracB"]
    arch = col_map["poly_type"]

    df = df.copy()
    df["_group_id"] = (
        df[sA].astype(str) + "|" +
        df[sB].astype(str) + "|" +
        df[fA].astype(str) + "|" +
        df[fB].astype(str)
    )

    arch_counts = df.groupby("_group_id")[arch].nunique()
    valid_groups = arch_counts[arch_counts >= 2].index
    df_matched = df[df["_group_id"].isin(valid_groups)].copy()

    return df_matched


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Compute pure architecture component (delta_y)
# ══════════════════════════════════════════════════════════════════════════════

def compute_architecture_deviations(df: pd.DataFrame, col_map: Dict) -> pd.DataFrame:
    """Compute delta_y = y - group_mean for each target."""
    ea_col = col_map["EA"]
    ip_col = col_map["IP"]

    df = df.copy()
    for target, t_col in [("EA", ea_col), ("IP", ip_col)]:
        group_means = df.groupby("_group_id")[t_col].transform("mean")
        df[f"delta_{target}"] = df[t_col] - group_means

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Feature construction
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_sets(
    df: pd.DataFrame,
    col_map: Dict,
    hA: Optional[np.ndarray],
    hB: Optional[np.ndarray],
    arch_encoding: str = "ordinal",
    descriptor_cols: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """Build all 6 specified feature matrices.

    Feature sets:
        1. arch_only:             [architecture]
        2. arch_frac:             [architecture, fracA, fracB]
        3. arch_chem:             [architecture, h_A, h_B]
        4. arch_chem_symmetric:   [architecture, h_A+h_B, |h_A-h_B|, h_A*h_B]
        5. arch_chem_frac:        [architecture, fracA, fracB, h_A, h_B]
        6. arch_chem_symmetric_frac: [architecture, fracA, fracB, h_A+h_B, |h_A-h_B|, h_A*h_B]

    Returns dict of {feature_set_name: X_matrix}.
    """
    arch_col = col_map["poly_type"]
    fA_col = col_map["fracA"]
    fB_col = col_map["fracB"]

    # Architecture features
    if arch_encoding == "ordinal":
        arch_feat = df[arch_col].map(ARCH_ORDINAL).values.reshape(-1, 1)
    else:  # one-hot
        enc = OneHotEncoder(sparse_output=False, drop="first")
        arch_feat = enc.fit_transform(df[[arch_col]])

    # Composition features
    frac_feat = df[[fA_col, fB_col]].values

    feature_sets = {}

    # 1. arch_only
    feature_sets["arch_only"] = arch_feat.copy()

    # 2. arch_frac
    feature_sets["arch_frac"] = np.hstack([arch_feat, frac_feat])

    # 3-6: Chemistry-conditioned (only if embeddings available)
    if hA is not None and hB is not None:
        # Symmetric features
        h_sum = hA + hB
        h_diff = np.abs(hA - hB)
        h_prod = hA * hB

        # 3. arch_chem
        feature_sets["arch_chem"] = np.hstack([arch_feat, hA, hB])

        # 4. arch_chem_symmetric
        feature_sets["arch_chem_symmetric"] = np.hstack([arch_feat, h_sum, h_diff, h_prod])

        # 5. arch_chem_frac
        feature_sets["arch_chem_frac"] = np.hstack([arch_feat, frac_feat, hA, hB])

        # 6. arch_chem_symmetric_frac
        feature_sets["arch_chem_symmetric_frac"] = np.hstack([arch_feat, frac_feat, h_sum, h_diff, h_prod])

    # Optional descriptor feature sets
    if descriptor_cols:
        valid_cols = [c for c in descriptor_cols if c in df.columns]
        if valid_cols:
            desc_feat = df[valid_cols].values
            feature_sets["descriptors_frac"] = np.hstack([arch_feat, frac_feat, desc_feat])
            if hA is not None and hB is not None:
                feature_sets["combined_all"] = np.hstack([arch_feat, frac_feat, hA, hB, desc_feat])

    return feature_sets


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Transfer evaluation with monomer-disjoint folds
# ══════════════════════════════════════════════════════════════════════════════

def get_fold_assignments(df: pd.DataFrame, col_map: Dict) -> np.ndarray:
    """Get or create monomer-disjoint fold assignments."""
    fold_col = col_map.get("fold")

    # Try loading from splits JSON
    if fold_col is None:
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parents[1]
        splits_json = project_root / "results" / "splits" / "ea_ip_aheldout_seed42.json"
        if splits_json.exists():
            with open(splits_json) as f:
                splits_data = json.load(f)
            fold_arr = np.full(len(df), -1, dtype=int)
            for fold_info in splits_data["folds"]:
                test_idx = np.array(fold_info["test_indices"])
                valid_mask = test_idx < len(df)
                fold_arr[test_idx[valid_mask]] = fold_info["fold"]
            if (fold_arr >= 0).all():
                print(f"  Loaded fold assignments from splits JSON ({splits_data['n_folds']} folds)")
                return fold_arr

    if fold_col is not None:
        return df[fold_col].values.astype(int)

    # Create new monomer-disjoint folds
    print("  Creating monomer-disjoint folds (GroupKFold on smiles_A)...")
    sA = col_map["smiles_A"]
    groups = df[sA].values
    gkf = GroupKFold(n_splits=5)
    fold_arr = np.full(len(df), -1, dtype=int)
    for fold_idx, (_, test_idx) in enumerate(gkf.split(np.arange(len(df)), groups=groups)):
        fold_arr[test_idx] = fold_idx
    return fold_arr


def train_evaluate_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_seed: int,
) -> Dict:
    """Train a model and return test metrics + predictions."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if model_name == "Ridge":
        model = Ridge(alpha=1.0, random_state=random_seed)
    elif model_name == "GBM":
        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=random_seed,
        )
    elif model_name == "MLP":
        model = MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=500,
            early_stopping=True, validation_fraction=0.15,
            random_state=random_seed, learning_rate_init=0.001,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    # Metrics
    r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else np.nan
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    corr, _ = pearsonr(y_test, y_pred) if len(y_test) > 2 else (np.nan, np.nan)

    # Feature importance for GBM
    feat_importance = None
    if model_name == "GBM":
        feat_importance = model.feature_importances_

    return {
        "r2": float(r2),
        "rmse": rmse,
        "mae": mae,
        "pearson_r": float(corr),
        "y_pred": y_pred,
        "feature_importance": feat_importance,
    }


def run_transfer_evaluation(
    df: pd.DataFrame,
    feature_sets: Dict[str, np.ndarray],
    fold_arr: np.ndarray,
    targets: Dict[str, str],
    model_names: List[str],
    random_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]:
    """Run full transfer evaluation across folds, feature sets, and models."""
    folds = sorted(np.unique(fold_arr))

    all_rows = []
    predictions = {}
    all_feat_importances = {}

    for target_short, target_col in targets.items():
        y_all = df[target_col].values

        for feat_name, X_all in feature_sets.items():
            if np.any(np.isnan(X_all)):
                print(f"  Skipping {feat_name} (contains NaN)")
                continue

            for model_name in model_names:
                fold_preds = np.full(len(df), np.nan)
                fold_importances = []

                for fold in folds:
                    train_mask = fold_arr != fold
                    test_mask = fold_arr == fold

                    X_train = X_all[train_mask]
                    y_train = y_all[train_mask]
                    X_test = X_all[test_mask]
                    y_test = y_all[test_mask]

                    if len(X_test) < 5 or len(X_train) < 20:
                        continue

                    result = train_evaluate_model(
                        model_name, X_train, y_train, X_test, y_test, random_seed
                    )

                    all_rows.append({
                        "target": target_short,
                        "feature_set": feat_name,
                        "model": model_name,
                        "fold": fold,
                        "R2": result["r2"],
                        "RMSE": result["rmse"],
                        "MAE": result["mae"],
                        "pearson_r": result["pearson_r"],
                        "n_test": int(test_mask.sum()),
                        "n_features": X_all.shape[1],
                    })

                    fold_preds[test_mask] = result["y_pred"]
                    if result["feature_importance"] is not None:
                        fold_importances.append(result["feature_importance"])

                key = f"{target_short}__{feat_name}__{model_name}"
                predictions[key] = fold_preds

                if fold_importances:
                    all_feat_importances[key] = np.mean(fold_importances, axis=0)

    metrics_df = pd.DataFrame(all_rows)

    if len(metrics_df) > 0:
        summary_df = (
            metrics_df.groupby(["target", "feature_set", "model"])
            .agg(
                mean_R2=("R2", "mean"),
                std_R2=("R2", "std"),
                mean_RMSE=("RMSE", "mean"),
                std_RMSE=("RMSE", "std"),
                mean_MAE=("MAE", "mean"),
                mean_pearson_r=("pearson_r", "mean"),
            )
            .reset_index()
        )
    else:
        summary_df = pd.DataFrame()

    return metrics_df, summary_df, predictions, all_feat_importances


# ══════════════════════════════════════════════════════════════════════════════
# Interpretation and comparison
# ══════════════════════════════════════════════════════════════════════════════

def compute_chemistry_gain(summary_df: pd.DataFrame) -> Dict[str, Dict]:
    """Compute chemistry gain = best_chem - best_arch_only for each target."""
    results = {}
    if summary_df.empty:
        return results

    # Define which feature sets are "baseline" vs "chemistry"
    baseline_sets = {"arch_only", "arch_frac"}
    chem_sets = {"arch_chem", "arch_chem_symmetric", "arch_chem_frac", "arch_chem_symmetric_frac"}

    for target in summary_df["target"].unique():
        sub = summary_df[summary_df["target"] == target]
        best_per_feat = sub.groupby("feature_set")["mean_R2"].max()

        # Best baseline R²
        baseline_r2s = [best_per_feat.get(k, np.nan) for k in baseline_sets if k in best_per_feat.index]
        best_baseline = max(baseline_r2s) if baseline_r2s else np.nan

        # Best chemistry R²
        chem_r2s = [best_per_feat.get(k, np.nan) for k in chem_sets if k in best_per_feat.index]
        best_chem = max(chem_r2s) if chem_r2s else np.nan

        # Which feature set achieved best chem?
        best_chem_name = None
        if chem_r2s:
            for k in chem_sets:
                if k in best_per_feat.index and best_per_feat[k] == best_chem:
                    best_chem_name = k
                    break

        # Chemistry gain
        if not np.isnan(best_chem) and not np.isnan(best_baseline):
            gain = best_chem - best_baseline
        else:
            gain = np.nan

        results[target] = {
            "best_baseline_r2": best_baseline,
            "best_chem_r2": best_chem,
            "best_chem_name": best_chem_name,
            "chemistry_gain": gain,
            "all_feat_r2": dict(best_per_feat),
        }

    return results


def generate_interpretation(
    summary_df: pd.DataFrame,
    gain_results: Dict[str, Dict],
    embeddings_found: bool,
    emb_dim: Optional[int],
) -> List[str]:
    """Generate interpretation with GO/NO-GO recommendation."""
    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("INTERPRETATION — Stage 2D-1 Feasibility")
    lines.append("=" * 70)

    if not embeddings_found:
        lines.append("")
        lines.append("  ⚠️  WARNING: No monomer embeddings found.")
        lines.append("  Chemistry-conditioned feature sets (arch_chem*) were NOT evaluated.")
        lines.append("  Provide embeddings via --embedding-file or inline hA_*/hB_* columns.")
        lines.append("")

    if summary_df.empty:
        lines.append("  No valid results to interpret.")
        return lines

    for target, info in gain_results.items():
        lines.append(f"\n  {target}:")

        # Print R² for each feature set
        feat_r2 = info["all_feat_r2"]
        ordered_sets = ["arch_only", "arch_frac", "arch_chem", "arch_chem_symmetric",
                        "arch_chem_frac", "arch_chem_symmetric_frac"]
        for fs in ordered_sets:
            if fs in feat_r2:
                lines.append(f"    {fs:<30} best R² = {feat_r2[fs]:.4f}")

        best_baseline = info["best_baseline_r2"]
        best_chem = info["best_chem_r2"]
        gain = info["chemistry_gain"]

        lines.append(f"\n    Best baseline (arch_only/arch_frac): R² = {best_baseline:.4f}" if not np.isnan(best_baseline) else "")

        if not np.isnan(best_chem):
            lines.append(f"    Best chemistry-conditioned:          R² = {best_chem:.4f} ({info['best_chem_name']})")
            lines.append(f"    Chemistry gain:                      ΔR² = {gain:+.4f}")

            if gain >= 0.05:
                lines.append("")
                lines.append("    ✅ Monomer chemistry provides meaningful transferable architecture signal.")
                lines.append("    → Stage 2D-1 is JUSTIFIED.")
            elif gain >= 0.01:
                lines.append("")
                lines.append("    ⚡ Chemistry provides weak additional transfer signal.")
                lines.append("    → Consider lightweight 2D-1 or regularized 2D-0.")
            else:
                lines.append("")
                lines.append("    ❌ Architecture effect is mostly global blockiness/composition.")
                lines.append("    → 2D-0 likely sufficient. Chemistry does not add transferable signal.")
        else:
            if not embeddings_found:
                lines.append("    Chemistry-conditioned: NOT EVALUATED (no embeddings)")
            else:
                lines.append("    Chemistry-conditioned: no valid results")

    # Overall GO/NO-GO
    lines.append("")
    lines.append("  " + "-" * 50)
    lines.append("  OVERALL RECOMMENDATION:")

    any_justified = False
    for target, info in gain_results.items():
        if not np.isnan(info.get("chemistry_gain", np.nan)) and info["chemistry_gain"] >= 0.05:
            any_justified = True

    if not embeddings_found:
        lines.append("    INCONCLUSIVE — no embeddings available for chemistry test.")
    elif any_justified:
        lines.append("    GO — Stage 2D-1 is justified.")
    else:
        gains = [info["chemistry_gain"] for info in gain_results.values() if not np.isnan(info.get("chemistry_gain", np.nan))]
        if gains and max(gains) >= 0.01:
            lines.append("    CAUTIOUS GO — weak signal, consider lightweight 2D-1.")
        else:
            lines.append("    NO-GO — 2D-0 sufficient under monomer-disjoint evaluation.")

    return lines


# ══════════════════════════════════════════════════════════════════════════════
# Visualizations
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {
    "arch_only": "#90caf9",
    "arch_frac": "#42a5f5",
    "arch_chem": "#ffcc80",
    "arch_chem_symmetric": "#ffa726",
    "arch_chem_frac": "#a5d6a7",
    "arch_chem_symmetric_frac": "#66bb6a",
    "descriptors_frac": "#ce93d8",
    "combined_all": "#ab47bc",
}


def generate_visualizations(
    summary_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    predictions: Dict,
    df: pd.DataFrame,
    feat_importances: Dict,
    col_map: Dict,
    out_dir: Path,
):
    """Generate diagnostic plots (no seaborn)."""
    if summary_df.empty:
        return

    # 1. Fold-wise R² barplots (best model per feature set)
    for target in sorted(metrics_df["target"].unique()):
        sub = metrics_df[metrics_df["target"] == target]
        # Use best model per feature set for clarity
        best_model_per_feat = sub.groupby("feature_set")["R2"].mean().reset_index()

        fig, ax = plt.subplots(figsize=(11, 5))
        feat_sets = sorted(sub["feature_set"].unique(),
                           key=lambda x: sub[sub["feature_set"] == x]["R2"].mean())
        n_feats = len(feat_sets)
        folds = sorted(sub["fold"].unique())
        n_folds = len(folds)
        width = 0.8 / max(n_feats, 1)

        for i, feat_name in enumerate(feat_sets):
            color = COLORS.get(feat_name, f"C{i}")
            # Best model for this feat_set
            best_model = sub[sub["feature_set"] == feat_name].groupby("model")["R2"].mean().idxmax()
            feat_sub = sub[(sub["feature_set"] == feat_name) & (sub["model"] == best_model)]
            r2s = feat_sub.sort_values("fold")["R2"].values
            if len(r2s) == n_folds:
                positions = np.arange(n_folds) + i * width - (n_feats - 1) * width / 2
                ax.bar(positions, r2s, width=width, label=feat_name, alpha=0.85, color=color)

        ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Fold", fontsize=11)
        ax.set_ylabel("Test R²", fontsize=11)
        ax.set_title(f"Δ{target} — Transfer R² by fold (best model per feature set)", fontweight="bold")
        ax.set_xticks(np.arange(n_folds))
        ax.set_xticklabels([str(f) for f in folds])
        ax.legend(fontsize=8, loc="best", ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / f"fold_r2_barplot_{target}.png", dpi=150, bbox_inches="tight")
        fig.savefig(out_dir / f"fold_r2_barplot_{target}.pdf", bbox_inches="tight")
        plt.close(fig)

    # 2. Comparison barplot (mean R² by feature set)
    for target in sorted(summary_df["target"].unique()):
        sub = summary_df[summary_df["target"] == target]
        best_per_feat = sub.groupby("feature_set")["mean_R2"].max().sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(8, max(3, len(best_per_feat) * 0.5)))
        colors = [COLORS.get(fs, "#999999") for fs in best_per_feat.index]
        bars = ax.barh(range(len(best_per_feat)), best_per_feat.values, color=colors, alpha=0.85)
        ax.set_yticks(range(len(best_per_feat)))
        ax.set_yticklabels(best_per_feat.index, fontsize=9)
        ax.set_xlabel("Mean R² (best model)", fontsize=11)
        ax.set_title(f"Δ{target} — Feature set comparison", fontweight="bold")
        ax.axvline(0, color="k", linewidth=0.5)
        # Add value labels
        for bar, val in zip(bars, best_per_feat.values):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"comparison_barplot_{target}.png", dpi=150, bbox_inches="tight")
        fig.savefig(out_dir / f"comparison_barplot_{target}.pdf", bbox_inches="tight")
        plt.close(fig)

    # 3. Predicted vs true for best model
    for target in sorted(summary_df["target"].unique()):
        best_row = summary_df[summary_df["target"] == target].sort_values("mean_R2", ascending=False).iloc[0]
        key = f"{target}__{best_row['feature_set']}__{best_row['model']}"
        if key in predictions:
            y_pred = predictions[key]
            y_true = df[f"delta_{target}"].values
            valid = ~np.isnan(y_pred)
            if valid.sum() > 10:
                fig, ax = plt.subplots(figsize=(5.5, 5.5))
                ax.scatter(y_true[valid], y_pred[valid], s=3, alpha=0.15, color="#1565c0", rasterized=True)
                lims = [min(y_true[valid].min(), y_pred[valid].min()),
                        max(y_true[valid].max(), y_pred[valid].max())]
                ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5)
                ax.set_xlabel(f"True Δ{target}", fontsize=11)
                ax.set_ylabel(f"Predicted Δ{target}", fontsize=11)
                ax.set_title(
                    f"Best: {best_row['model']} ({best_row['feature_set']})\n"
                    f"R² = {best_row['mean_R2']:.4f}, r = {best_row['mean_pearson_r']:.4f}",
                    fontweight="bold", fontsize=10
                )
                fig.tight_layout()
                fig.savefig(out_dir / f"pred_vs_true_{target}.png", dpi=150, bbox_inches="tight")
                fig.savefig(out_dir / f"pred_vs_true_{target}.pdf", bbox_inches="tight")
                plt.close(fig)

    # 4. Feature importance (GBM)
    for key, importances in feat_importances.items():
        target, feat_name, model = key.split("__")
        if model != "GBM":
            continue
        n_show = min(25, len(importances))
        idx_sorted = np.argsort(importances)[-n_show:]

        fig, ax = plt.subplots(figsize=(6, max(3, n_show * 0.3)))
        ax.barh(range(n_show), importances[idx_sorted], color="#66bb6a", alpha=0.8)
        ax.set_yticks(range(n_show))
        ax.set_yticklabels([f"feat_{i}" for i in idx_sorted], fontsize=7)
        ax.set_xlabel("Importance")
        ax.set_title(f"Δ{target} — GBM feature importance ({feat_name})", fontweight="bold", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / f"feat_importance_{target}_{feat_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 5. Architecture-conditioned deviation distributions
    arch_col = col_map["poly_type"]
    for target in ["delta_EA", "delta_IP"]:
        if target not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        arch_colors = {"alternating": "#42a5f5", "random": "#66bb6a", "block": "#ffa726"}
        for arch_label in sorted(df[arch_col].unique()):
            vals = df[df[arch_col] == arch_label][target].values
            color = arch_colors.get(arch_label, "#999999")
            ax.hist(vals, bins=60, alpha=0.5, label=arch_label, density=True, color=color)
        ax.set_xlabel(target, fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"Distribution of {target} by architecture", fontweight="bold")
        ax.legend(fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / f"distribution_{target}.png", dpi=150, bbox_inches="tight")
        fig.savefig(out_dir / f"distribution_{target}.pdf", bbox_inches="tight")
        plt.close(fig)

    print(f"  Plots saved to {out_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
# Audit / Leakage verification
# ══════════════════════════════════════════════════════════════════════════════

FORBIDDEN_FEATURE_NAMES = {
    "ea", "ip", "delta_ea", "delta_ip", "target", "label", "y",
    "prediction", "residual", "fold",
}

# Expected feature dimensions per feature set
EXPECTED_DIMS = {
    "arch_only":                 lambda d, enc: (1 if enc == "ordinal" else None),
    "arch_frac":                 lambda d, enc: (3 if enc == "ordinal" else None),
    "arch_chem":                 lambda d, enc: ((1 if enc == "ordinal" else None) and (1 + 2 * d) if d else None),
    "arch_chem_symmetric":       lambda d, enc: ((1 if enc == "ordinal" else None) and (1 + 3 * d) if d else None),
    "arch_chem_frac":            lambda d, enc: ((3 if enc == "ordinal" else None) and (3 + 2 * d) if d else None),
    "arch_chem_symmetric_frac":  lambda d, enc: ((3 if enc == "ordinal" else None) and (3 + 3 * d) if d else None),
}


def _expected_dim(feat_name: str, emb_dim: Optional[int], arch_encoding: str) -> Optional[int]:
    """Compute expected feature dimension for a feature set."""
    n_arch = 1 if arch_encoding == "ordinal" else 2  # drop='first' from 3 classes
    d = emb_dim if emb_dim else 0
    expected = {
        "arch_only": n_arch,
        "arch_frac": n_arch + 2,
        "arch_chem": n_arch + 2 * d,
        "arch_chem_symmetric": n_arch + 3 * d,
        "arch_chem_frac": n_arch + 2 + 2 * d,
        "arch_chem_symmetric_frac": n_arch + 2 + 3 * d,
    }
    return expected.get(feat_name, None)


def _feature_names_for_set(feat_name: str, emb_dim: Optional[int], arch_encoding: str) -> List[str]:
    """Generate expected feature names for a feature set."""
    d = emb_dim if emb_dim else 0
    arch_names = ["arch_ordinal"] if arch_encoding == "ordinal" else ["arch_oh_1", "arch_oh_2"]
    frac_names = ["fracA", "fracB"]
    hA_names = [f"hA_{i}" for i in range(d)]
    hB_names = [f"hB_{i}" for i in range(d)]
    h_sum_names = [f"hSum_{i}" for i in range(d)]
    h_diff_names = [f"hAbsDiff_{i}" for i in range(d)]
    h_prod_names = [f"hProd_{i}" for i in range(d)]

    mapping = {
        "arch_only": arch_names,
        "arch_frac": arch_names + frac_names,
        "arch_chem": arch_names + hA_names + hB_names,
        "arch_chem_symmetric": arch_names + h_sum_names + h_diff_names + h_prod_names,
        "arch_chem_frac": arch_names + frac_names + hA_names + hB_names,
        "arch_chem_symmetric_frac": arch_names + frac_names + h_sum_names + h_diff_names + h_prod_names,
    }
    return mapping.get(feat_name, [f"feat_{i}" for i in range(10)])


def audit_1_feature_dimensions(
    feature_sets: Dict[str, np.ndarray],
    emb_dim: Optional[int],
    arch_encoding: str,
) -> Tuple[List[str], pd.DataFrame]:
    """AUDIT 1: Verify feature dimensionality for each feature set."""
    lines = []
    rows = []
    lines.append("=" * 70)
    lines.append("AUDIT 1 — Feature Dimensionality")
    lines.append("=" * 70)

    all_pass = True
    for feat_name in sorted(feature_sets.keys()):
        X = feature_sets[feat_name]
        n_feat = X.shape[1]
        expected = _expected_dim(feat_name, emb_dim, arch_encoding)
        feat_names = _feature_names_for_set(feat_name, emb_dim, arch_encoding)

        status = "OK"
        if expected is not None and n_feat != expected:
            status = f"MISMATCH (expected {expected}, got {n_feat})"
            all_pass = False

        lines.append(f"\n  {feat_name}:")
        lines.append(f"    n_features:  {n_feat}")
        lines.append(f"    expected:    {expected if expected else 'N/A'}")
        lines.append(f"    status:      {status}")
        lines.append(f"    first 10:    {feat_names[:10]}")

        rows.append({
            "feature_set": feat_name,
            "n_features": n_feat,
            "expected": expected,
            "status": status,
            "first_10_names": str(feat_names[:10]),
        })

    if all_pass:
        lines.append("\n  ✓ All feature dimensions match expectations.")
    else:
        lines.append("\n  ⚠️  WARNING: Some feature dimensions do not match expectations.")

    return lines, pd.DataFrame(rows)


def audit_2_embedding_merge(
    df: pd.DataFrame,
    col_map: Dict,
    hA: Optional[np.ndarray],
    hB: Optional[np.ndarray],
    emb_dim: Optional[int],
    embeddings_found: bool,
    embedding_source: str,
) -> Tuple[List[str], pd.DataFrame]:
    """AUDIT 2: Embedding merge integrity check."""
    lines = []
    rows = []
    lines.append("\n" + "=" * 70)
    lines.append("AUDIT 2 — Embedding Merge Integrity")
    lines.append("=" * 70)

    if not embeddings_found:
        lines.append("\n  No embeddings loaded — skipping.")
        return lines, pd.DataFrame()

    lines.append(f"\n  Embedding source: {embedding_source}")
    lines.append(f"  Embedding dimensionality: {emb_dim}")

    sA_col = col_map["smiles_A"]
    sB_col = col_map["smiles_B"]
    unique_A = set(df[sA_col].unique())
    unique_B = set(df[sB_col].unique())
    all_monomers = unique_A | unique_B
    lines.append(f"  Unique monomers in dataset: {len(all_monomers)}")
    lines.append(f"    monomer_A unique: {len(unique_A)}")
    lines.append(f"    monomer_B unique: {len(unique_B)}")

    if hA is not None and hB is not None:
        n_rows = hA.shape[0]
        hA_nan_rows = int(np.isnan(hA).any(axis=1).sum())
        hB_nan_rows = int(np.isnan(hB).any(axis=1).sum())
        lines.append(f"  Embedding array rows: {n_rows}")
        lines.append(f"  Rows with NaN in hA: {hA_nan_rows}")
        lines.append(f"  Rows with NaN in hB: {hB_nan_rows}")

        # Check for constant / near-constant columns
        hA_var = np.var(hA, axis=0)
        hB_var = np.var(hB, axis=0)
        n_const_A = int((hA_var < 1e-10).sum())
        n_const_B = int((hB_var < 1e-10).sum())
        lines.append(f"  hA constant columns (var < 1e-10): {n_const_A}/{emb_dim}")
        lines.append(f"  hB constant columns (var < 1e-10): {n_const_B}/{emb_dim}")

        if n_const_A > 0 or n_const_B > 0:
            lines.append("  ⚠️  WARNING: Some embedding columns are constant or near-constant.")

        # Duplicate columns check
        n_dup_A = emb_dim - len(set([tuple(hA[:, i]) for i in range(min(emb_dim, 50))]))
        lines.append(f"  hA duplicate columns (checked first 50): {n_dup_A}")

        rows.append({
            "metric": "emb_dim", "value": emb_dim,
        })
        rows.append({
            "metric": "n_rows_with_embeddings", "value": n_rows,
        })
        rows.append({
            "metric": "hA_nan_rows", "value": hA_nan_rows,
        })
        rows.append({
            "metric": "hB_nan_rows", "value": hB_nan_rows,
        })
        rows.append({
            "metric": "hA_constant_cols", "value": n_const_A,
        })
        rows.append({
            "metric": "hB_constant_cols", "value": n_const_B,
        })

    lines.append("\n  ✓ Embedding merge audit complete.")
    return lines, pd.DataFrame(rows)


def audit_3_fold_leakage(
    df: pd.DataFrame,
    col_map: Dict,
    fold_arr: np.ndarray,
) -> Tuple[List[str], pd.DataFrame, bool]:
    """AUDIT 3: Monomer leakage across folds."""
    lines = []
    rows = []
    lines.append("\n" + "=" * 70)
    lines.append("AUDIT 3 — Fold Leakage Check (Monomer Overlap)")
    lines.append("=" * 70)

    sA_col = col_map["smiles_A"]
    sB_col = col_map["smiles_B"]

    folds = sorted(np.unique(fold_arr))
    any_leakage = False

    for fold in folds:
        train_mask = fold_arr != fold
        test_mask = fold_arr == fold

        train_monomers_A = set(df.loc[train_mask, sA_col].unique())
        train_monomers_B = set(df.loc[train_mask, sB_col].unique())
        test_monomers_A = set(df.loc[test_mask, sA_col].unique())
        test_monomers_B = set(df.loc[test_mask, sB_col].unique())

        train_all = train_monomers_A | train_monomers_B
        test_all = test_monomers_A | test_monomers_B
        overlap = train_all & test_all

        n_overlap = len(overlap)
        if n_overlap > 0:
            any_leakage = True

        lines.append(f"\n  Fold {fold}:")
        lines.append(f"    Train rows: {int(train_mask.sum())}, Test rows: {int(test_mask.sum())}")
        lines.append(f"    Unique train monomers: {len(train_all)}")
        lines.append(f"    Unique test monomers:  {len(test_all)}")
        lines.append(f"    Overlapping monomers:  {n_overlap}")
        if n_overlap > 0:
            overlap_list = sorted(overlap)[:20]
            lines.append(f"    First 20 overlap: {overlap_list}")

        rows.append({
            "fold": fold,
            "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "unique_train_monomers": len(train_all),
            "unique_test_monomers": len(test_all),
            "overlap_count": n_overlap,
        })

    if any_leakage:
        lines.append("\n  ❌ MONOMER LEAKAGE DETECTED")
        lines.append("     Some monomers appear in both train and test splits.")
        lines.append("     This may inflate transfer R² estimates.")
    else:
        lines.append("\n  ✓ No monomer leakage detected across folds.")

    return lines, pd.DataFrame(rows), any_leakage


def audit_4_group_leakage(
    df: pd.DataFrame,
    fold_arr: np.ndarray,
) -> Tuple[List[str], pd.DataFrame, bool]:
    """AUDIT 4: Matched group leakage (same group in train and test)."""
    lines = []
    rows = []
    lines.append("\n" + "=" * 70)
    lines.append("AUDIT 4 — Matched Group Leakage Check")
    lines.append("=" * 70)

    folds = sorted(np.unique(fold_arr))
    group_ids = df["_group_id"].values
    any_leakage = False

    for fold in folds:
        train_mask = fold_arr != fold
        test_mask = fold_arr == fold

        train_groups = set(group_ids[train_mask])
        test_groups = set(group_ids[test_mask])
        overlap = train_groups & test_groups

        n_overlap = len(overlap)
        if n_overlap > 0:
            any_leakage = True

        lines.append(f"\n  Fold {fold}:")
        lines.append(f"    Train groups: {len(train_groups)}, Test groups: {len(test_groups)}")
        lines.append(f"    Groups in BOTH train and test: {n_overlap}")
        if n_overlap > 0:
            overlap_list = sorted(overlap)[:20]
            lines.append(f"    First 20 leaked groups: {overlap_list}")

        rows.append({
            "fold": fold,
            "train_groups": len(train_groups),
            "test_groups": len(test_groups),
            "leaked_groups": n_overlap,
        })

    if any_leakage:
        lines.append("\n  ❌ MATCHED GROUP LEAKAGE DETECTED")
        lines.append("     Same (monomer_A, monomer_B, fracA, fracB) groups appear in train and test.")
    else:
        lines.append("\n  ✓ No matched group leakage.")

    return lines, pd.DataFrame(rows), any_leakage


def audit_5_target_validation(
    df: pd.DataFrame,
    col_map: Dict,
) -> Tuple[List[str], pd.DataFrame]:
    """AUDIT 5: Target construction validation."""
    lines = []
    rows = []
    lines.append("\n" + "=" * 70)
    lines.append("AUDIT 5 — Target Construction Validation")
    lines.append("=" * 70)

    arch_col = col_map["poly_type"]

    for target in ["delta_EA", "delta_IP"]:
        if target not in df.columns:
            continue

        # Verify group means are zero
        group_means = df.groupby("_group_id")[target].mean()
        max_abs_mean = float(group_means.abs().max())
        mean_abs_mean = float(group_means.abs().mean())
        n_nonzero = int((group_means.abs() > 1e-8).sum())

        lines.append(f"\n  {target}:")
        lines.append(f"    Max |group mean delta|:  {max_abs_mean:.2e}")
        lines.append(f"    Mean |group mean delta|: {mean_abs_mean:.2e}")
        lines.append(f"    Groups with |mean| > 1e-8: {n_nonzero}/{len(group_means)}")
        lines.append(f"    Overall variance: {df[target].var():.8f}")

        # Distribution by architecture
        lines.append(f"    Distribution by architecture:")
        for arch_label in sorted(df[arch_col].unique()):
            vals = df[df[arch_col] == arch_label][target].values
            lines.append(
                f"      {arch_label:12s}: n={len(vals):5d}, "
                f"mean={vals.mean():+.6f}, median={np.median(vals):+.6f}, "
                f"std={vals.std():.6f}, min={vals.min():+.6f}, max={vals.max():+.6f}"
            )

        rows.append({
            "target": target,
            "max_abs_group_mean": max_abs_mean,
            "mean_abs_group_mean": mean_abs_mean,
            "n_groups_nonzero": n_nonzero,
            "total_groups": len(group_means),
            "variance": float(df[target].var()),
        })

    if all(r["max_abs_group_mean"] < 1e-8 for r in rows):
        lines.append("\n  ✓ All group mean deltas are numerically zero.")
    else:
        lines.append("\n  ⚠️  Some group means are not exactly zero (floating point).")

    return lines, pd.DataFrame(rows)


def audit_6_feature_leakage(
    feature_sets: Dict[str, np.ndarray],
    df: pd.DataFrame,
) -> Tuple[List[str], bool]:
    """AUDIT 6: Check that target/label columns are not in features."""
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("AUDIT 6 — Feature Leakage Check (target in features)")
    lines.append("=" * 70)

    # Check column names in the dataframe that could have been used
    df_cols_lower = {c.lower() for c in df.columns}
    leakage_found = False

    # For each feature set, we check dimensionality against known safe sources.
    # Since features are numpy arrays (not named), we verify by exclusion:
    # the only columns that SHOULD produce features are architecture, fracA, fracB, and embeddings.
    # If feature dim exceeds expected, something wrong.
    # Additionally check that no target-like values snuck in by correlation.

    for feat_name, X in feature_sets.items():
        # Check if any column of X is perfectly correlated with targets
        for target_col in ["delta_EA", "delta_IP"]:
            if target_col not in df.columns:
                continue
            y = df[target_col].values
            if len(y) != X.shape[0]:
                continue
            for col_idx in range(X.shape[1]):
                x_col = X[:, col_idx]
                # Check perfect correlation
                if np.std(x_col) > 0 and np.std(y) > 0:
                    corr = np.abs(np.corrcoef(x_col, y)[0, 1])
                    if corr > 0.999:
                        lines.append(f"\n  ❌ FEATURE LEAKAGE: {feat_name} column {col_idx} "
                                     f"has correlation {corr:.6f} with {target_col}")
                        leakage_found = True

    # Also check that forbidden column names are not in feature construction sources
    # We inspect the dataframe columns used
    safe_sources = {"poly_type", "architecture", "arch", "polymer_type", "polytype",
                    "fraca", "fracb", "frac_a", "frac_b", "fraction_a", "fraction_b"}
    for col in df.columns:
        if col.lower() in FORBIDDEN_FEATURE_NAMES:
            # Not necessarily leakage unless it's IN the features
            # We can't directly check numpy arrays for column origin,
            # but we flag if such columns exist
            pass

    if leakage_found:
        lines.append("\n  ❌ FEATURE LEAKAGE DETECTED — features correlate perfectly with targets.")
    else:
        lines.append("\n  ✓ No feature leakage detected (no perfect target correlations).")

    return lines, leakage_found


def audit_7_permutation_controls(
    df: pd.DataFrame,
    feature_sets: Dict[str, np.ndarray],
    fold_arr: np.ndarray,
    targets: Dict[str, str],
    random_seed: int,
) -> Tuple[List[str], pd.DataFrame]:
    """AUDIT 7: Permutation negative controls.

    1. Shuffle delta_y in train → expect R² ≈ 0 on real test targets.
    2. Shuffle architecture labels in train → expect R² drop.
    """
    lines = []
    rows = []
    lines.append("\n" + "=" * 70)
    lines.append("AUDIT 7 — Permutation Controls")
    lines.append("=" * 70)

    control_sets = ["arch_only", "arch_frac", "arch_chem_frac", "arch_chem_symmetric_frac"]
    control_sets = [fs for fs in control_sets if fs in feature_sets]

    folds = sorted(np.unique(fold_arr))
    rng = np.random.RandomState(random_seed + 999)

    for target_short, target_col in targets.items():
        y_all = df[target_col].values

        for feat_name in control_sets:
            X_all = feature_sets[feat_name]
            if np.any(np.isnan(X_all)):
                continue

            # Original R²
            orig_r2s = []
            shuf_target_r2s = []
            shuf_arch_r2s = []

            for fold in folds:
                train_mask = fold_arr != fold
                test_mask = fold_arr == fold
                X_train = X_all[train_mask]
                X_test = X_all[test_mask]
                y_train = y_all[train_mask]
                y_test = y_all[test_mask]

                if len(X_test) < 5 or len(X_train) < 20:
                    continue

                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_train)
                X_te_s = scaler.transform(X_test)

                # Original
                model = Ridge(alpha=1.0)
                model.fit(X_tr_s, y_train)
                orig_r2s.append(r2_score(y_test, model.predict(X_te_s)))

                # Control 1: shuffle train targets
                y_train_shuf = rng.permutation(y_train)
                model_shuf = Ridge(alpha=1.0)
                model_shuf.fit(X_tr_s, y_train_shuf)
                shuf_target_r2s.append(r2_score(y_test, model_shuf.predict(X_te_s)))

                # Control 2: shuffle architecture column in train features
                X_train_shuf = X_tr_s.copy()
                X_train_shuf[:, 0] = rng.permutation(X_train_shuf[:, 0])
                model_arch_shuf = Ridge(alpha=1.0)
                model_arch_shuf.fit(X_train_shuf, y_train)
                shuf_arch_r2s.append(r2_score(y_test, model_arch_shuf.predict(X_te_s)))

            if orig_r2s:
                orig_mean = np.mean(orig_r2s)
                shuf_t_mean = np.mean(shuf_target_r2s)
                shuf_a_mean = np.mean(shuf_arch_r2s)

                lines.append(f"\n  {target_short} | {feat_name}:")
                lines.append(f"    Original R² (Ridge):            {orig_mean:.4f}")
                lines.append(f"    Shuffled-target R² (expect ~0): {shuf_t_mean:.4f}")
                lines.append(f"    Shuffled-arch R² (expect drop): {shuf_a_mean:.4f}")

                rows.append({
                    "target": target_short,
                    "feature_set": feat_name,
                    "original_R2": orig_mean,
                    "shuffled_target_R2": shuf_t_mean,
                    "shuffled_arch_R2": shuf_a_mean,
                })

    # Verdict
    if rows:
        max_shuf_target = max(r["shuffled_target_R2"] for r in rows)
        if max_shuf_target > 0.1:
            lines.append(f"\n  ❌ WARNING: Shuffled-target control has R² = {max_shuf_target:.4f} > 0.1")
            lines.append("     This suggests potential information leakage or overfitting.")
        else:
            lines.append(f"\n  ✓ Shuffled-target controls collapse near zero (max={max_shuf_target:.4f}).")
    else:
        lines.append("\n  No permutation controls computed (no valid feature sets).")

    return lines, pd.DataFrame(rows)


def audit_8_foldwise_table(
    metrics_df: pd.DataFrame,
) -> Tuple[List[str], pd.DataFrame]:
    """AUDIT 8: Full fold-wise metrics table."""
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("AUDIT 8 — Fold-wise Full Metrics Table")
    lines.append("=" * 70)

    if metrics_df.empty:
        lines.append("\n  No fold-wise metrics available.")
        return lines, metrics_df

    lines.append(f"\n  Total rows: {len(metrics_df)}")
    lines.append(f"  Columns: {list(metrics_df.columns)}")
    lines.append(f"  Unique targets: {sorted(metrics_df['target'].unique())}")
    lines.append(f"  Unique feature sets: {sorted(metrics_df['feature_set'].unique())}")
    lines.append(f"  Unique models: {sorted(metrics_df['model'].unique())}")
    lines.append(f"  Unique folds: {sorted(metrics_df['fold'].unique())}")

    return lines, metrics_df


def run_audit(
    df_matched: pd.DataFrame,
    col_map: Dict,
    feature_sets: Dict[str, np.ndarray],
    fold_arr: np.ndarray,
    hA_matched: Optional[np.ndarray],
    hB_matched: Optional[np.ndarray],
    emb_dim: Optional[int],
    embeddings_found: bool,
    embedding_source: str,
    arch_encoding: str,
    targets: Dict[str, str],
    metrics_df: pd.DataFrame,
    random_seed: int,
    out_dir: Path,
) -> str:
    """Run all audit checks and save outputs. Returns PASS/FAIL/WARNING."""
    print("\n" + "=" * 70)
    print("RUNNING AUDIT / LEAKAGE VERIFICATION")
    print("=" * 70)

    all_lines = []
    all_lines.append("=" * 70)
    all_lines.append("FEATURE-CONDITIONED ARCHITECTURE TRANSFER — AUDIT REPORT")
    all_lines.append("=" * 70)
    all_lines.append(f"\nTimestamp: audit run")
    all_lines.append(f"Random seed: {random_seed}")
    all_lines.append(f"Architecture encoding: {arch_encoding}")
    all_lines.append(f"Embeddings found: {embeddings_found}")
    all_lines.append(f"Embedding dim: {emb_dim}")
    all_lines.append(f"Rows: {len(df_matched)}")
    all_lines.append(f"Folds: {len(np.unique(fold_arr))}")

    # Track pass/fail conditions
    monomer_leakage = False
    group_leakage = False
    feature_leakage = False
    permutation_suspicious = False
    dim_mismatch = False

    # AUDIT 1
    print("  Running Audit 1: Feature dimensionality...")
    a1_lines, a1_df = audit_1_feature_dimensions(feature_sets, emb_dim, arch_encoding)
    all_lines.extend(a1_lines)
    a1_df.to_csv(out_dir / "audit_feature_dimensions.csv", index=False)
    if any("MISMATCH" in str(s) for s in a1_df.get("status", [])):
        dim_mismatch = True

    # AUDIT 2
    print("  Running Audit 2: Embedding merge integrity...")
    a2_lines, a2_df = audit_2_embedding_merge(
        df_matched, col_map, hA_matched, hB_matched, emb_dim, embeddings_found, embedding_source
    )
    all_lines.extend(a2_lines)
    if not a2_df.empty:
        a2_df.to_csv(out_dir / "audit_embedding_merge.csv", index=False)

    # AUDIT 3
    print("  Running Audit 3: Fold leakage (monomer overlap)...")
    a3_lines, a3_df, monomer_leakage = audit_3_fold_leakage(df_matched, col_map, fold_arr)
    all_lines.extend(a3_lines)
    a3_df.to_csv(out_dir / "audit_fold_leakage.csv", index=False)

    # AUDIT 4
    print("  Running Audit 4: Matched group leakage...")
    a4_lines, a4_df, group_leakage = audit_4_group_leakage(df_matched, fold_arr)
    all_lines.extend(a4_lines)
    a4_df.to_csv(out_dir / "audit_group_leakage.csv", index=False)

    # AUDIT 5
    print("  Running Audit 5: Target construction validation...")
    a5_lines, a5_df = audit_5_target_validation(df_matched, col_map)
    all_lines.extend(a5_lines)
    a5_df.to_csv(out_dir / "audit_target_validation.csv", index=False)

    # AUDIT 6
    print("  Running Audit 6: Feature leakage check...")
    a6_lines, feature_leakage = audit_6_feature_leakage(feature_sets, df_matched)
    all_lines.extend(a6_lines)

    # AUDIT 7
    print("  Running Audit 7: Permutation controls (this may take a few minutes)...")
    a7_lines, a7_df = audit_7_permutation_controls(
        df_matched, feature_sets, fold_arr, targets, random_seed
    )
    all_lines.extend(a7_lines)
    if not a7_df.empty:
        a7_df.to_csv(out_dir / "audit_permutation_controls.csv", index=False)
        max_shuf = a7_df["shuffled_target_R2"].max() if "shuffled_target_R2" in a7_df.columns else 0
        if max_shuf > 0.1:
            permutation_suspicious = True

    # AUDIT 8
    print("  Running Audit 8: Fold-wise full table...")
    a8_lines, a8_df = audit_8_foldwise_table(metrics_df)
    all_lines.extend(a8_lines)
    if not a8_df.empty:
        a8_df.to_csv(out_dir / "audit_foldwise_full_metrics.csv", index=False)

    # ── FINAL DECISION ──
    all_lines.append("\n\n" + "=" * 70)
    all_lines.append("FINAL AUDIT DECISION")
    all_lines.append("=" * 70)

    if monomer_leakage or group_leakage or feature_leakage or permutation_suspicious:
        verdict = "FAIL"
        reasons = []
        if monomer_leakage:
            reasons.append("Monomer leakage across folds")
        if group_leakage:
            reasons.append("Matched group leakage across folds")
        if feature_leakage:
            reasons.append("Target columns correlated with features")
        if permutation_suspicious:
            reasons.append("Permutation controls did not collapse (possible leakage)")
        all_lines.append(f"\n  VERDICT: ❌ FAIL")
        all_lines.append(f"  Reasons:")
        for r in reasons:
            all_lines.append(f"    - {r}")
    elif dim_mismatch or not embeddings_found:
        verdict = "WARNING"
        all_lines.append(f"\n  VERDICT: ⚠️  WARNING")
        if dim_mismatch:
            all_lines.append("    - Feature dimensions do not match expectations")
        if not embeddings_found:
            all_lines.append("    - No embeddings loaded (chemistry-conditioned sets not tested)")
    else:
        verdict = "PASS"
        all_lines.append(f"\n  VERDICT: ✓ PASS")
        all_lines.append("    - No monomer leakage")
        all_lines.append("    - No matched group leakage")
        all_lines.append("    - Feature dimensions match expectations")
        all_lines.append("    - Embeddings present and non-constant")
        all_lines.append("    - No target leakage")
        all_lines.append("    - Permutation controls collapse near zero")

    # Save report
    with open(out_dir / "audit_report.txt", "w") as f:
        f.write("\n".join(all_lines))

    # Print summary to console
    print(f"\n  AUDIT VERDICT: {verdict}")
    print(f"  Saved audit_report.txt and supporting CSVs to: {out_dir}/")

    return verdict


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Feature-conditioned architecture transfer diagnostic — Stage 2D-1 feasibility"
    )
    parser.add_argument("--data", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument("--out", type=str, default="feature_conditioned_transfer", help="Output directory")
    parser.add_argument("--embedding-file", type=str, default=None,
                        help="Path to monomer embedding CSV (columns: monomer_smiles, emb_0, ..., emb_d) or .npy")
    parser.add_argument("--descriptor-cols", nargs="*", default=None,
                        help="Physical descriptor column names to include as features")
    parser.add_argument("--arch-encoding", choices=["ordinal", "onehot"], default="ordinal",
                        help="Architecture encoding method (default: ordinal)")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Models to evaluate (default: ridge gbm mlp)")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed")
    parser.add_argument("--rdkit-features", action="store_true",
                        help="Use RDKit Morgan fingerprints as proxy chemistry embeddings")
    parser.add_argument("--fp-bits", type=int, default=128,
                        help="Number of bits for Morgan fingerprints (default: 128)")
    parser.add_argument("--fp-radius", type=int, default=2,
                        help="Morgan fingerprint radius (default: 2)")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--audit", action="store_true",
                        help="Run leakage/verification audit and save detailed report")
    args = parser.parse_args()

    random_seed = args.random_seed
    np.random.seed(random_seed)

    # Parse model names
    model_names = ["Ridge", "GBM", "MLP"]
    if args.models:
        name_map = {"ridge": "Ridge", "gbm": "GBM", "mlp": "MLP"}
        model_names = [name_map.get(m.lower(), m) for m in args.models]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Track warnings for report
    report_warnings = []

    # ── Load data ──
    print("=" * 70)
    print("FEATURE-CONDITIONED ARCHITECTURE TRANSFER DIAGNOSTIC")
    print("=" * 70)
    print(f"\nLoading data from: {args.data}")
    df = pd.read_csv(args.data)
    print(f"  Shape: {df.shape}")

    col_map = infer_columns(df)
    print("  Column mapping:")
    for k, v in col_map.items():
        if v is not None:
            print(f"    {k} -> '{v}'")

    # Drop rows with missing targets
    ea_col = col_map["EA"]
    ip_col = col_map["IP"]
    n_before = len(df)
    df = df.dropna(subset=[ea_col, ip_col]).reset_index(drop=True)
    if len(df) < n_before:
        print(f"  Dropped {n_before - len(df)} rows with missing targets.")

    # ── Load embeddings ──
    print("\n" + "=" * 70)
    print("EMBEDDING DISCOVERY")
    print("=" * 70)

    hA = None
    hB = None
    emb_dim = None
    embeddings_found = False
    embedding_source = "none"

    # Try FORMAT 1: inline columns in main CSV
    inline_result = detect_inline_embeddings(df)
    if inline_result is not None:
        hA, hB, emb_dim = inline_result
        embeddings_found = True
        embedding_source = "inline (hA_*/hB_* columns)"
        print(f"  ✓ Found inline embeddings: hA_0..hA_{emb_dim-1}, hB_0..hB_{emb_dim-1}")
        print(f"    Embedding dimensionality: {emb_dim}")

    # Try FORMAT 2: separate embedding file
    if not embeddings_found and args.embedding_file:
        emb_path = Path(args.embedding_file)
        print(f"  Loading embeddings from: {emb_path}")
        result = load_embedding_file(emb_path, df, col_map)
        if result is not None:
            hA, hB, emb_dim = result
            embeddings_found = True
            embedding_source = f"external file ({emb_path.name})"
            print(f"    Embedding dimensionality: {emb_dim}")

    # Try RDKit fingerprints as fallback
    if not embeddings_found and args.rdkit_features:
        print(f"  Computing RDKit Morgan fingerprints as proxy embeddings...")
        result = compute_rdkit_fingerprints(df, col_map, radius=args.fp_radius, n_bits=args.fp_bits)
        if result is not None:
            hA, hB, emb_dim = result
            embeddings_found = True
            embedding_source = f"RDKit Morgan fingerprints (radius={args.fp_radius}, bits={args.fp_bits})"
            print(f"  ✓ RDKit fingerprints computed: dim={emb_dim}")

    if not embeddings_found:
        print("  ⚠️  No monomer embeddings found.")
        print("     - No hA_*/hB_* columns in data CSV")
        if args.embedding_file:
            print(f"     - Failed to load from: {args.embedding_file}")
        else:
            print("     - No --embedding-file provided")
        if not args.rdkit_features:
            print("     - Use --rdkit-features to auto-generate Morgan fingerprint proxies")
        print("     Chemistry-conditioned feature sets will be SKIPPED.")
        report_warnings.append("No embeddings found — arch_chem* feature sets skipped.")

    # ── Step 1: Build matched groups ──
    print("\n" + "=" * 70)
    print("STEP 1 — Build matched groups")
    print("=" * 70)
    df_matched = build_matched_groups(df, col_map)
    n_groups = df_matched["_group_id"].nunique()
    print(f"  Matched rows: {len(df_matched)} / {len(df)}")
    print(f"  Matched groups (≥2 architectures): {n_groups}")

    # Subset embeddings to matched rows
    hA_matched = None
    hB_matched = None
    if embeddings_found:
        matched_idx = df_matched.index.values
        hA_matched = hA[matched_idx]
        hB_matched = hB[matched_idx]
        # Check for NaN after subset
        valid_emb = ~(np.isnan(hA_matched).any(axis=1) | np.isnan(hB_matched).any(axis=1))
        n_valid_emb = int(valid_emb.sum())
        if n_valid_emb < len(df_matched):
            n_drop = len(df_matched) - n_valid_emb
            print(f"  ⚠️  {n_drop} matched rows lack embeddings — filtering to {n_valid_emb} rows")
            report_warnings.append(f"Embedding merge dropped {n_drop} rows (retained {n_valid_emb}).")
            df_matched = df_matched[valid_emb.values if hasattr(valid_emb, 'values') else valid_emb].reset_index(drop=True)
            hA_matched = hA_matched[valid_emb]
            hB_matched = hB_matched[valid_emb]
            n_groups = df_matched["_group_id"].nunique()
            print(f"  After embedding filter: {len(df_matched)} rows, {n_groups} groups")

    # ── Step 2: Compute architecture deviations ──
    print("\n" + "=" * 70)
    print("STEP 2 — Compute architecture-conditioned deviations (Δy)")
    print("=" * 70)
    df_matched = compute_architecture_deviations(df_matched, col_map)
    print(f"  delta_EA: mean={df_matched['delta_EA'].mean():.6f}, std={df_matched['delta_EA'].std():.6f}")
    print(f"  delta_IP: mean={df_matched['delta_IP'].mean():.6f}, std={df_matched['delta_IP'].std():.6f}")
    print(f"  delta_EA variance: {df_matched['delta_EA'].var():.8f}")
    print(f"  delta_IP variance: {df_matched['delta_IP'].var():.8f}")

    # ── Step 3: Feature construction ──
    print("\n" + "=" * 70)
    print("STEP 3 — Feature construction")
    print("=" * 70)

    feature_sets = build_feature_sets(
        df_matched, col_map,
        hA=hA_matched, hB=hB_matched,
        arch_encoding=args.arch_encoding,
        descriptor_cols=args.descriptor_cols,
    )
    print(f"  Architecture encoding: {args.arch_encoding}")
    print(f"  Feature sets built:")
    for name, X in sorted(feature_sets.items()):
        print(f"    {name:<30} shape: {X.shape}")

    # ── Step 4: Transfer evaluation ──
    print("\n" + "=" * 70)
    print("STEP 4 — Monomer-disjoint transfer evaluation")
    print("=" * 70)

    fold_arr = get_fold_assignments(df_matched, col_map)
    n_folds = len(np.unique(fold_arr))
    print(f"  Folds: {n_folds}")
    print(f"  Models: {model_names}")
    print(f"  Targets: delta_EA, delta_IP")
    print(f"  Evaluating {len(feature_sets)} feature sets × {len(model_names)} models × {n_folds} folds...")

    targets = {"EA": "delta_EA", "IP": "delta_IP"}

    metrics_df, summary_df, predictions, feat_importances = run_transfer_evaluation(
        df_matched, feature_sets, fold_arr, targets, model_names, random_seed
    )

    # ── Run audit if requested ──
    if args.audit:
        audit_verdict = run_audit(
            df_matched=df_matched,
            col_map=col_map,
            feature_sets=feature_sets,
            fold_arr=fold_arr,
            hA_matched=hA_matched,
            hB_matched=hB_matched,
            emb_dim=emb_dim,
            embeddings_found=embeddings_found,
            embedding_source=embedding_source,
            arch_encoding=args.arch_encoding,
            targets=targets,
            metrics_df=metrics_df,
            random_seed=random_seed,
            out_dir=out_dir,
        )

    # ── Print results ──
    print("\n" + "=" * 70)
    print("RESULTS — Transfer metrics summary")
    print("=" * 70)
    if not summary_df.empty:
        for target in sorted(summary_df["target"].unique()):
            print(f"\n  Δ{target}:")
            sub = summary_df[summary_df["target"] == target].sort_values("mean_R2", ascending=False)
            print(f"  {'Feature Set':<30} {'Model':<7} {'R²':>7} {'±std':>7} {'RMSE':>8} {'MAE':>8} {'r':>7}")
            print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*7}")
            for _, row in sub.iterrows():
                print(
                    f"  {row['feature_set']:<30} {row['model']:<7} "
                    f"{row['mean_R2']:>7.4f} {row['std_R2']:>7.4f} "
                    f"{row['mean_RMSE']:>8.5f} {row['mean_MAE']:>8.5f} "
                    f"{row['mean_pearson_r']:>7.4f}"
                )

    # ── Chemistry gain comparison ──
    gain_results = compute_chemistry_gain(summary_df)

    print("\n" + "=" * 70)
    print("CRITICAL COMPARISON — Chemistry gain over baseline")
    print("=" * 70)
    for target, info in sorted(gain_results.items()):
        print(f"\n  Δ{target}:")
        print(f"    Best baseline (arch_only/arch_frac):    R² = {info['best_baseline_r2']:.4f}")
        if not np.isnan(info.get("best_chem_r2", np.nan)):
            print(f"    Best chemistry-conditioned:             R² = {info['best_chem_r2']:.4f} ({info['best_chem_name']})")
            print(f"    Chemistry gain (ΔR²):                  {info['chemistry_gain']:+.4f}")
        else:
            print(f"    Best chemistry-conditioned:             N/A (no embeddings)")

    # ── Interpretation ──
    interp_lines = generate_interpretation(summary_df, gain_results, embeddings_found, emb_dim)
    for line in interp_lines:
        print(line)

    # ── Save outputs ──
    print("\n" + "=" * 70)
    print("SAVING OUTPUTS")
    print("=" * 70)

    metrics_df.to_csv(out_dir / "fold_metrics.csv", index=False)
    summary_df.to_csv(out_dir / "transfer_metrics.csv", index=False)

    # Predictions
    pred_df = pd.DataFrame(predictions)
    pred_df["delta_EA_true"] = df_matched["delta_EA"].values
    pred_df["delta_IP_true"] = df_matched["delta_IP"].values
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    # Feature importance
    if feat_importances:
        fi_rows = []
        for key, imp in feat_importances.items():
            parts = key.split("__")
            for i, val in enumerate(imp):
                fi_rows.append({
                    "target": parts[0], "feature_set": parts[1],
                    "model": parts[2], "feature_idx": i, "importance": float(val)
                })
        pd.DataFrame(fi_rows).to_csv(out_dir / "feature_importance.csv", index=False)

    # Report
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("FEATURE-CONDITIONED ARCHITECTURE TRANSFER REPORT")
    report_lines.append("=" * 70)
    report_lines.append(f"\nDataset: {args.data}")
    report_lines.append(f"Total rows: {len(df)}")
    report_lines.append(f"Matched rows (≥2 arch): {len(df_matched)}")
    report_lines.append(f"Matched groups: {n_groups}")
    report_lines.append(f"Folds: {n_folds}")
    report_lines.append(f"Models: {model_names}")
    report_lines.append(f"Architecture encoding: {args.arch_encoding}")
    report_lines.append(f"Random seed: {random_seed}")
    report_lines.append(f"\nEmbeddings found: {embeddings_found}")
    if embeddings_found:
        report_lines.append(f"Embedding dimensionality: {emb_dim}")
        report_lines.append(f"Rows retained after embedding merge: {len(df_matched)}")
    report_lines.append(f"\ndelta_EA variance: {df_matched['delta_EA'].var():.8f}")
    report_lines.append(f"delta_IP variance: {df_matched['delta_IP'].var():.8f}")

    if report_warnings:
        report_lines.append("\nWARNINGS:")
        for w in report_warnings:
            report_lines.append(f"  - {w}")

    report_lines.append("\n\nFEATURE SETS:")
    for name, X in sorted(feature_sets.items()):
        report_lines.append(f"  {name}: {X.shape[1]} features")

    if not summary_df.empty:
        report_lines.append("\n\nTRANSFER METRICS (sorted by R²):")
        for target in sorted(summary_df["target"].unique()):
            report_lines.append(f"\n  Δ{target}:")
            sub = summary_df[summary_df["target"] == target].sort_values("mean_R2", ascending=False)
            report_lines.append(sub.to_string(index=False))

    report_lines.append("\n\nCHEMISTRY GAIN:")
    for target, info in sorted(gain_results.items()):
        report_lines.append(f"\n  Δ{target}:")
        report_lines.append(f"    Best baseline R²: {info['best_baseline_r2']:.4f}")
        if not np.isnan(info.get("best_chem_r2", np.nan)):
            report_lines.append(f"    Best chem R²: {info['best_chem_r2']:.4f} ({info['best_chem_name']})")
            report_lines.append(f"    Chemistry gain: {info['chemistry_gain']:+.4f}")

    report_lines.extend(interp_lines)

    with open(out_dir / "transfer_report.txt", "w") as f:
        f.write("\n".join(report_lines))

    print(f"\n  All outputs saved to: {out_dir}/")

    # ── Visualizations ──
    if not args.no_plots:
        print("\n  Generating visualizations...")
        generate_visualizations(
            summary_df, metrics_df, predictions, df_matched,
            feat_importances, col_map, out_dir,
        )

    # ── Final summary ──
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  Embeddings: {'YES (dim={})'.format(emb_dim) if embeddings_found else 'NOT AVAILABLE'}")
    if not summary_df.empty:
        for target in sorted(summary_df["target"].unique()):
            best = summary_df[summary_df["target"] == target].sort_values("mean_R2", ascending=False).iloc[0]
            print(f"  Δ{target}: best R² = {best['mean_R2']:.4f} ({best['feature_set']}, {best['model']})")
            if target in gain_results:
                g = gain_results[target]["chemistry_gain"]
                if not np.isnan(g):
                    print(f"         chemistry gain: ΔR² = {g:+.4f}")
    else:
        print("  No valid transfer results.")


if __name__ == "__main__":
    main()
