"""
Tabular Utilities for Molecular Property Prediction

This module provides utilities for processing tabular molecular data, including:
- RDKit descriptor computation
- Atom and bond feature extraction
- Feature engineering and preprocessing
- Model evaluation metrics
"""

# Standard library imports
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third-party imports
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler

# Lazy imports for heavy dependencies
RDKIT_AVAILABLE = False


try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdchem
    RDKIT_AVAILABLE = True
except ImportError:
    import warnings
    warnings.warn("RDKit not available. Some functionality will be limited.")


ATOM_FEAT_LEN = 100 + 6 + 5 + 4 + 5 + 5 + 1 + 1   # = 127
BOND_FEAT_LEN = 4 + 1 + 1 + 6                      # = 12
AB_POOLED_LEN = ATOM_FEAT_LEN + BOND_FEAT_LEN      # = 139


# -------------------------- RDKit descriptors ------------------------

# Initialize RDKit descriptors lazily
def _get_rdkit_descriptors() -> Tuple[List[Tuple[str, Callable]], List[str]]:
    """Lazily load RDKit descriptors and their names.
    
    Returns:
        Tuple containing:
            - List of (descriptor_name, descriptor_function) tuples
            - List of descriptor names
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for descriptor calculation")
    desc_list = list(Descriptors.descList)
    desc_names = [n for n, _ in desc_list]
    return desc_list, desc_names

# Initialize descriptors on first use
RD_DESC: List[Tuple[str, Callable]] = []
RD_DESC_NAMES: List[str] = []

def _init_rdkit_descriptors() -> None:
    """Initialize RDKit descriptors if not already done."""
    global RD_DESC, RD_DESC_NAMES
    if len(RD_DESC) == 0 or len(RD_DESC_NAMES) == 0:
        desc_list, desc_names = _get_rdkit_descriptors()
        RD_DESC.clear()
        RD_DESC.extend(desc_list)
        RD_DESC_NAMES.clear()
        RD_DESC_NAMES.extend(desc_names)

def compute_rdkit_desc(mol: Optional[Chem.Mol]) -> np.ndarray:
    """Compute RDKit descriptors for a molecule.
    
    Args:
        mol: RDKit molecule object or None
        
    Returns:
        np.ndarray: Array of descriptor values, with NaNs for failed calculations
        
    Note:
        Handles special case for 'SPS' descriptor with no heavy atoms.
    """
    _init_rdkit_descriptors()
    if mol is None:
        return np.full(len(RD_DESC), np.nan, dtype=float)
        
    out = []
    num_heavy = mol.GetNumHeavyAtoms()
    
    for name, func in RD_DESC:
        try:
            if name == "SPS" and num_heavy == 0:
                out.append(0.0)  # Special case for SPS with no heavy atoms
            else:
                out.append(float(func(mol)))
        except Exception as e:
            out.append(np.nan)
            
    return np.asarray(out, dtype=float)

def rdkit_block_from_smiles(smiles: List[str]) -> np.ndarray:
    """Compute RDKit descriptors for a list of SMILES strings.
    
    Args:
        smiles: List of SMILES strings
        
    Returns:
        np.ndarray: 2D array of shape (n_molecules, n_descriptors) with imputed values
        
    Note:
        - Invalid SMILES will result in rows of NaNs
        - Missing values are imputed with column medians
        - Inf values are converted to NaN before imputation
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for SMILES processing")
        
    if not isinstance(smiles, (list, np.ndarray)):
        raise ValueError("smiles must be a list or numpy array")
        
    # Convert SMILES to molecules
    mols = [Chem.MolFromSmiles(s) if isinstance(s, str) else None for s in smiles]
    
    # Compute descriptors
    try:
        M = np.vstack([compute_rdkit_desc(m) for m in mols])
    except Exception as e:
        raise RuntimeError(f"Error computing RDKit descriptors: {str(e)}")
    
    # Convert inf to nan, then impute missing values with column medians
    M = np.where(np.isfinite(M), M, np.nan)
    
    # Handle all-NaN columns by setting median to 0
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(M, axis=0)
        # Replace NaN medians (from all-NaN columns) with 0
        med = np.where(np.isnan(med), 0.0, med)
    
    inds = np.where(np.isnan(M))
    M[inds] = np.take(med, inds[1])
    
    return M


# ----------------------- Atom+Bond pooled features --------------------
# Constants for feature extraction
ATOM_DEGREE_MAX = 5  # clamp >=6 to 5
ATOM_HCOUNT_MAX = 4  # clamp >=5 to 4

# Define domains for one-hot encoding
if RDKIT_AVAILABLE:
    ATOM_CHIRALITY = [
        rdchem.ChiralType.CHI_UNSPECIFIED,
        rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        rdchem.ChiralType.CHI_OTHER,
    ]
    ATOM_HYBRID = [
        rdchem.HybridizationType.SP,
        rdchem.HybridizationType.SP2,
        rdchem.HybridizationType.SP3,
        rdchem.HybridizationType.SP3D,
        rdchem.HybridizationType.SP3D2,
    ]
    BOND_TYPES = [
        rdchem.BondType.SINGLE,
        rdchem.BondType.DOUBLE,
        rdchem.BondType.TRIPLE,
        rdchem.BondType.AROMATIC,
    ]
    BOND_STEREO = [
        rdchem.BondStereo.STEREONONE,
        rdchem.BondStereo.STEREOANY,
        rdchem.BondStereo.STEREOE,   # E
        rdchem.BondStereo.STEREOZ,   # Z
        rdchem.BondStereo.STEREOCIS,
        rdchem.BondStereo.STEREOTRANS,
    ]
else:
    # Define empty lists if RDKit is not available
    ATOM_CHIRALITY = []
    ATOM_HYBRID = []
    BOND_TYPES = []
    BOND_STEREO = []

# Domain for formal charge one-hot encoding
CHARGE_DOMAIN = [-2, -1, 0, 1, 2]


def one_hot_index(val: Any, domain: List[Any]) -> Optional[int]:
    """Get the index of a value in a domain, returning None if not found.
    
    Args:
        val: Value to find in the domain
        domain: List of possible values
        
    Returns:
        Index of the value in the domain, or None if not found
    """
    try:
        return domain.index(val)
    except ValueError:
        return None

def atom_feature_vector(atom: 'rdchem.Atom') -> np.ndarray:
    """Create a feature vector for an atom.
    
    Args:
        atom: RDKit atom object
        
    Returns:
        np.ndarray: Concatenated feature vector with the following components:
            - Atomic number one-hot (1-100)
            - Degree one-hot (0-5+)
            - Formal charge one-hot (-2 to +2)
            - Chirality one-hot
            - Hydrogen count one-hot (0-4+)
            - Hybridization one-hot
            - Aromatic flag
            - Scaled atomic mass
    
    Raises:
        ImportError: If RDKit is not available
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for atom feature calculation")
        
    if not isinstance(atom, rdchem.Atom):
        raise ValueError("atom must be an RDKit Atom object")

    # Atomic number one-hot Z = 1..100; outside -> all zeros
    Z_MAX = 100
    z = atom.GetAtomicNum()
    z_onehot = np.zeros(Z_MAX, dtype=float)  # indices 0..99 represent Z=1..100
    if 1 <= z <= Z_MAX:
        z_onehot[z - 1] = 1.0  # shift by -1

    # Degree 0..5, clamp >=6 to 5
    deg_onehot = np.zeros(ATOM_DEGREE_MAX + 1, dtype=float)
    d = min(max(int(atom.GetDegree()), 0), ATOM_DEGREE_MAX)
    deg_onehot[d] = 1.0

    # Formal charge one-hot {-2,-1,0,+1,+2}; outside -> zeros
    fc_onehot = np.zeros(len(CHARGE_DOMAIN), dtype=float)
    fc = atom.GetFormalCharge()
    if fc in CHARGE_DOMAIN:
        fc_onehot[CHARGE_DOMAIN.index(fc)] = 1.0

    # Chirality
    chiral_onehot = np.zeros(len(ATOM_CHIRALITY), dtype=float)
    ci = one_hot_index(atom.GetChiralTag(), ATOM_CHIRALITY)
    if ci is not None:
        chiral_onehot[ci] = 1.0

    # #Hs 0..4, clamp >=5 to 4
    hcount_onehot = np.zeros(ATOM_HCOUNT_MAX + 1, dtype=float)
    h = min(max(int(atom.GetTotalNumHs()), 0), ATOM_HCOUNT_MAX)
    hcount_onehot[h] = 1.0

    # Hybridization (only the 5 allowed; else zeros)
    hybrid_onehot = np.zeros(len(ATOM_HYBRID), dtype=float)
    hi = one_hot_index(atom.GetHybridization(), ATOM_HYBRID)
    if hi is not None:
        hybrid_onehot[hi] = 1.0

    # Additional atomic properties
    aromatic = np.array([1.0 if atom.GetIsAromatic() else 0.0], dtype=float)
    mass_scaled = np.array([atom.GetMass() / 100.0], dtype=float)  # Scale mass to ~0-2 range

    return np.concatenate([
        z_onehot, deg_onehot, fc_onehot, chiral_onehot,
        hcount_onehot, hybrid_onehot, aromatic, mass_scaled
    ], axis=0)


def bond_feature_vector(bond: 'rdchem.Bond') -> np.ndarray:
    """Create a feature vector for a bond.
    
    Args:
        bond: RDKit bond object
        
    Returns:
        np.ndarray: Concatenated feature vector with the following components:
            - Bond type one-hot (single, double, triple, aromatic)
            - Conjugation flag
            - Ring membership flag
            - Stereo configuration one-hot
    
    Raises:
        ImportError: If RDKit is not available
        ValueError: If bond is not an RDKit Bond object
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for bond feature calculation")
        
    if not isinstance(bond, rdchem.Bond):
        raise ValueError("bond must be an RDKit Bond object")

    # Bond type (single, double, triple, aromatic)
    bt_onehot = np.zeros(len(BOND_TYPES), dtype=float)
    bi = one_hot_index(bond.GetBondType(), BOND_TYPES)
    if bi is not None:
        bt_onehot[bi] = 1.0

    # Bond properties
    conj = np.array([1.0 if bond.GetIsConjugated() else 0.0], dtype=float)
    in_ring = np.array([1.0 if bond.IsInRing() else 0.0], dtype=float)

    # Stereochemistry
    stereo_onehot = np.zeros(len(BOND_STEREO), dtype=float)
    si = one_hot_index(bond.GetStereo(), BOND_STEREO)
    if si is not None:
        stereo_onehot[si] = 1.0

    return np.concatenate([bt_onehot, conj, in_ring, stereo_onehot], axis=0)

def pool_matrix(mat: np.ndarray, mode: str = 'mean') -> np.ndarray:
    """Pool a matrix along the first axis using the specified method.
    
    Args:
        mat: Input array of shape (n_samples, n_features)
        mode: Pooling method, either 'mean' or 'sum'
        
    Returns:
        Pooled array of shape (n_features,)
        
    Raises:
        ValueError: If mode is not 'mean' or 'sum' or if input is not 2D
    """
    if not isinstance(mat, np.ndarray) or mat.ndim != 2:
        raise ValueError("Input must be a 2D numpy array")
        
    if mode == 'mean':
        return np.mean(mat, axis=0)
    elif mode == 'sum':
        return np.sum(mat, axis=0)
    else:
        raise ValueError("mode must be either 'mean' or 'sum'")

def atom_bond_block_from_smiles(smiles: List[str], pool: str = 'mean', add_counts: bool = False) -> np.ndarray:
    """Generate atom and bond features for a list of SMILES strings.
    
    Args:
        smiles: List of SMILES strings
        pool: Pooling method for atom and bond features ('mean' or 'sum')
        add_counts: Whether to append atom and bond counts to the features
        
    Returns:
        np.ndarray: Array of shape (n_molecules, n_features) with pooled atom and bond features
        
    Raises:
        ImportError: If RDKit is not available
        ValueError: If input is not a list or if pooling fails
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for SMILES processing")
        
    if not isinstance(smiles, (list, np.ndarray)):
        raise ValueError("smiles must be a list or numpy array")
        
    feats = []
    
    for s in smiles:
        # Convert SMILES to molecule
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        
        # Handle invalid molecules
        if m is None or m.GetNumAtoms() == 0:
            vec = np.full(AB_POOLED_LEN + (2 if add_counts else 0), np.nan, dtype=float)
            feats.append(vec)
            continue

            
        try:
            # Compute atom features
            atom_features = [atom_feature_vector(a) for a in m.GetAtoms()]
            atom_mat = np.vstack(atom_features) if atom_features else np.zeros((1, 10))
            
            # Compute bond features
            if m.GetNumBonds() > 0:
                bond_features = [bond_feature_vector(b) for b in m.GetBonds()]
                bond_mat = np.vstack(bond_features)
            else:
                # For molecules with no bonds, use zeros with appropriate feature size
                bond_feat_size = 12  # Size of bond feature vector
                bond_mat = np.zeros((1, bond_feat_size))
                
            # Pool features
            atom_pooled = pool_matrix(atom_mat, pool)
            bond_pooled = pool_matrix(bond_mat, pool)
            
            # Combine features
            vec = np.concatenate([atom_pooled, bond_pooled])
            
            # Optionally add counts
            if add_counts:
                counts = np.array([m.GetNumAtoms(), m.GetNumBonds()], dtype=float)
                vec = np.concatenate([vec, counts])
                
            feats.append(vec)
            
        except Exception as e:
            # Fall back to canonical size even if mats aren't defined
            feats.append(np.full(AB_POOLED_LEN + (2 if add_counts else 0), np.nan, dtype=float))
    
    # Convert to numpy array and impute missing values
    try:
        M = np.vstack(feats)
        M = np.where(np.isfinite(M), M, np.nan)
        
        # Impute NaN values with column medians
        med = np.nanmedian(M, axis=0)
        inds = np.where(np.isnan(M))
        M[inds] = np.take(med, inds[1])
        
        return M
        
    except Exception as e:
        raise ValueError(f"Error processing features: {str(e)}")


def weighted_average(A: np.ndarray, B: np.ndarray, fa: np.ndarray, fb: np.ndarray) -> np.ndarray:
    """Compute a weighted average of two arrays with weights fa and fb.
    
    Args:
        A: First array of shape (n_samples, n_features)
        B: Second array of shape (n_samples, n_features)
        fa: Weights for array A, shape (n_samples,)
        fb: Weights for array B, shape (n_samples,)
        
    Returns:
        Weighted average of A and B with shape (n_samples, n_features)
        
    Note:
        Adds a small epsilon (1e-12) to denominator for numerical stability.
    """
    # Input validation
    if not all(isinstance(x, np.ndarray) for x in [A, B, fa, fb]):
        raise ValueError("All inputs must be numpy arrays")
        
    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape")
        
    if fa.shape != fb.shape or fa.ndim != 1:
        raise ValueError("fa and fb must be 1D arrays with the same length")
        
    if len(fa) != A.shape[0]:
        raise ValueError("Length of weights must match first dimension of A and B")
    
    # Reshape for broadcasting
    fa = fa.reshape(-1, 1)
    fb = fb.reshape(-1, 1)
    
    # Compute weighted average with numerical stability
    denom = (fa + fb + 1e-12)
    fa_norm = fa / denom
    fb_norm = fb / denom
    
    return fa_norm * A + fb_norm * B

def canon_pair(a, b, wa, wb):
    # sort lexicographically; if swapped, flip fractions
    if b < a:
        return b, a, wb, wa
    return a, b, wa, wb


# ------------------------ Feature assembly ---------------------------




def build_features(
    df: pd.DataFrame,
    train_idx: List[int],
    descriptor_columns: List[str],
    kind: str,
    use_rdkit: bool,
    use_ab: bool = True,
    pool: str = 'mean',
    add_counts: bool = False,
    smiles_column: str = "smiles",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Assemble feature blocks for homo- and co-polymers.

    Returns:
      ab_block: np.ndarray or None
      descriptor_block: np.ndarray or None
      names: list[str] (AB_* first, then descriptor names)
    """
    use_descriptor = len(descriptor_columns) > 0
    ab_block = None
    descriptor_block = None
    names: List[str] = []

    # -------------------- HOMOPOLYMER --------------------
    if kind == "homo":
        if smiles_column not in df.columns:
            raise KeyError(f"homo mode expects a SMILES column '{smiles_column}' (pass via smiles_column).")
        smiles = df[smiles_column].astype(str).tolist()

        # AB pooled features
        if use_ab:
            ab_block = atom_bond_block_from_smiles(smiles, pool=pool, add_counts=add_counts)
            names += [f"AB_{i}" for i in range(ab_block.shape[1])]

        # RDKit + dataset descriptors
        desc_blocks, desc_names = [], []

        if use_rdkit:
            # ensure RD_DESC_NAMES is initialized
            _init_rdkit_descriptors()
            rdkit_block = rdkit_block_from_smiles(smiles)
            desc_blocks.append(rdkit_block)
            desc_names += [f"RD_{n}" for n in RD_DESC_NAMES]

        if use_descriptor:
            dataset_desc_block = df[descriptor_columns].values
            desc_blocks.append(dataset_desc_block)
            desc_names += descriptor_columns

        if desc_blocks:
            descriptor_block = np.concatenate(desc_blocks, axis=1)
            names += desc_names

        if ab_block is None and descriptor_block is None:
            raise ValueError("No features selected. Use --incl_ab and/or --incl_rdkit / --incl_desc.")

        return ab_block, descriptor_block, names

    # -------------------- COPOLYMER --------------------
    # accept either smilesA/smilesB or smiles_A/smiles_B
    smilesA_col = "smilesA" if "smilesA" in df.columns else "smiles_A"
    smilesB_col = "smilesB" if "smilesB" in df.columns else "smiles_B"
    if smilesA_col not in df.columns or smilesB_col not in df.columns:
        raise KeyError("copolymer mode expects 'smilesA'/'smilesB' or 'smiles_A'/'smiles_B' columns.")

    sA_raw = df[smilesA_col].astype(str).tolist()
    sB_raw = df[smilesB_col].astype(str).tolist()

    # accept either fracA/fracB or frac_A/frac_B; compute fracB if missing
    if "fracA" in df.columns or "fracB" in df.columns:
        fA_raw = pd.to_numeric(df.get("fracA", np.nan), errors="coerce")
        fB_raw = pd.to_numeric(df.get("fracB", np.nan), errors="coerce")
    elif "frac_A" in df.columns or "frac_B" in df.columns:
        fA_raw = pd.to_numeric(df.get("frac_A", np.nan), errors="coerce")
        fB_raw = pd.to_numeric(df.get("frac_B", np.nan), errors="coerce")
    else:
        raise KeyError("copolymer mode expects fraction columns: 'fracA'/'fracB' or 'frac_A'/'frac_B'.")

    # if only fA provided, infer fB
    if fB_raw.isna().any() and not fA_raw.isna().all():
        fB_raw = 1.0 - fA_raw

    if fA_raw.isna().any() or fB_raw.isna().any():
        raise ValueError("Found NaNs in fracA/fracB after coercion/inference.")

    # clip tiny numeric issues and normalize to sum 1.0 (safe normalization)
    fsum = (fA_raw.values.astype(float) + fB_raw.values.astype(float))
    bad = ~np.isfinite(fsum) | (fsum <= 0)
    if bad.any():
        raise ValueError("Invalid fractions: non-finite or non-positive totals in fracA+fracB.")
    fA_raw = fA_raw.values.astype(float) / fsum
    fB_raw = 1.0 - fA_raw

    # canonicalize pairs so A+B == B+A; swap fractions accordingly
    sA, sB, fA, fB = [], [], [], []
    for a, b, wa, wb in zip(sA_raw, sB_raw, fA_raw, fB_raw):
        a2, b2, wa2, wb2 = canon_pair(a, b, wa, wb)
        sA.append(a2); sB.append(b2); fA.append(wa2); fB.append(wb2)
    fA = np.asarray(fA, float); fB = np.asarray(fB, float)

    # AB pooled: per monomer then weighted blend
    if use_ab:
        abA = atom_bond_block_from_smiles(sA, pool=pool, add_counts=add_counts)
        abB = atom_bond_block_from_smiles(sB, pool=pool, add_counts=add_counts)
        ab_block = weighted_average(abA, abB, fA, fB)
        names += [f"AB_{i}" for i in range(ab_block.shape[1])]

    # RDKit + dataset descriptors
    desc_blocks, desc_names = [], []

    if use_rdkit:
        _init_rdkit_descriptors()
        rdA = rdkit_block_from_smiles(sA)
        rdB = rdkit_block_from_smiles(sB)

        # fit scaler on TRAIN rows only, using both monomer matrices (as you had)
        train_stack = np.vstack([rdA[train_idx], rdB[train_idx]])
        sc = StandardScaler().fit(train_stack)
        rdA_z = sc.transform(rdA)
        rdB_z = sc.transform(rdB)

        rd_blend = weighted_average(rdA_z, rdB_z, fA, fB)
        desc_blocks.append(rd_blend)
        desc_names += [f"RD_{n}" for n in RD_DESC_NAMES]

    if use_descriptor:
        dataset_desc_block = df[descriptor_columns].values
        desc_blocks.append(dataset_desc_block)
        desc_names += descriptor_columns

    if desc_blocks:
        descriptor_block = np.concatenate(desc_blocks, axis=1)
        names += desc_names

    if ab_block is None and descriptor_block is None:
        raise ValueError("No features selected. Use --incl_ab and/or --incl_rdkit / --incl_desc.")

    return ab_block, descriptor_block, names



def eval_regression(y_true, y_pred) -> Dict[str, float]:
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred)
    }

def eval_binary(y_true, y_pred, y_prob) -> Dict[str, float]:
    # y_prob is probability for positive class (1)
    out = {
        "acc": accuracy_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred, zero_division=0),
        "rec": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }
    if y_prob is not None:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
        out["logloss"] = log_loss(y_true, np.vstack([1-y_prob, y_prob]).T, labels=[0,1])
    return out

def eval_multi(y_true, y_pred, y_proba) -> Dict[str, float]:
    out = {
        "acc": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0)
    }
    if y_proba is not None:
        out["logloss"] = log_loss(y_true, y_proba)
        # Add ROC-AUC for multi-class using one-vs-rest approach
        try:
            from sklearn.preprocessing import label_binarize
            n_classes = len(np.unique(y_true))
            if n_classes == 2:
                # Binary classification
                out["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                # Multi-class classification
                y_bin = label_binarize(y_true, classes=list(range(n_classes)))
                out["roc_auc"] = roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr")
        except Exception as e:
            # Skip ROC-AUC if calculation fails
            pass
    return out

def summarize(results: List[Dict[str, float]]) -> Dict[str, float]:
    keys = results[0].keys()
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in results if k in r], dtype=float)
        out[f"{k}_mean"] = np.nanmean(vals)
        out[f"{k}_std"] = np.nanstd(vals)
    return out


# ------------------------ Data Preprocessing Functions ------------------------

def preprocess_descriptor_data(descriptor_block: np.ndarray, train_idx: List[int], val_idx: List[int], 
                             test_idx: List[int], orig_desc_names: List[str], 
                             logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                            List[str], Dict[str, Any], SimpleImputer, 
                                                            np.ndarray, np.ndarray]:
    """Preprocess descriptor data with constant removal, imputation, and correlation filtering.
    
    Args:
        descriptor_block: Raw descriptor data array
        train_idx: Training indices
        val_idx: Validation indices  
        test_idx: Test indices
        orig_desc_names: Original descriptor column names
        logger: Logger instance
        
    Returns:
        Tuple containing:
            - desc_tr_selected: Processed training descriptors
            - desc_val_selected: Processed validation descriptors
            - desc_te_selected: Processed test descriptors
            - selected_desc_names: Names of selected descriptors
            - preprocessing_metadata: Metadata dict
            - imputer: Fitted imputer object
            - constant_mask: Boolean mask for constant removal
            - corr_mask: Boolean mask for correlation removal
    """
    # Clean descriptor data before converting to float32
    desc_X = np.asarray(descriptor_block, dtype=np.float64)  # Use float64 first
    
    # Replace inf with NaN
    inf_mask = np.isinf(desc_X)
    if np.any(inf_mask):
        logger.warning(f"Found {np.sum(inf_mask)} infinite values in descriptors, replacing with NaN")
        desc_X[inf_mask] = np.nan
    
    # Clip extreme values to prevent float32 overflow
    float32_max = np.finfo(np.float32).max
    float32_min = np.finfo(np.float32).min
    desc_X = np.clip(desc_X, float32_min, float32_max)
    
    # Now safely convert to float32
    desc_X = desc_X.astype(np.float32)
    logger.debug(f"Descriptor data shape: {desc_X.shape}")
    logger.debug(f"Descriptor data - finite values: {np.isfinite(desc_X).all()}")
    
    # 1) Remove constants on FULL descriptor dataset BEFORE imputation
    desc_full_df = pd.DataFrame(desc_X, columns=orig_desc_names)
    non_na_uniques = desc_full_df.nunique(dropna=True)
    constant_mask = (non_na_uniques >= 2).values  # True = keep
    constant_features = [n for n, keep in zip(orig_desc_names, constant_mask) if not keep]
    const_kept_names = [n for n, keep in zip(orig_desc_names, constant_mask) if keep]

    # Remove constants from full dataset
    const_keep_idx = np.where(constant_mask)[0]
    desc_X_no_const = desc_X[:, const_keep_idx]
    
    # Split descriptor data AFTER constant removal but BEFORE imputation
    desc_tr, desc_val, desc_te = desc_X_no_const[train_idx], desc_X_no_const[val_idx], desc_X_no_const[test_idx]
    
    # Handle NaN values with median imputation fitted only on training data
    nan_mask = np.isnan(desc_tr)
    imputer = SimpleImputer(strategy='median')
    if np.any(nan_mask):
        logger.warning(f"Found {np.sum(nan_mask)} NaN values in training descriptors, using median imputation")
    # Fit on training either way
    desc_tr = imputer.fit_transform(desc_tr)
    # ALWAYS transform val/test
    desc_val = imputer.transform(desc_val)
    desc_te  = imputer.transform(desc_te)

    
    # Correlation filtering on TRAIN ONLY, using constant-removed training set
    desc_tr_df = pd.DataFrame(desc_tr, columns=const_kept_names)
    if len(const_kept_names) > 1:
        corr = desc_tr_df.corr(method="pearson").abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = {c for c in upper.columns if (upper[c] >= 0.90).any()}
        keep_names = [n for n in const_kept_names if n not in to_drop]
    else:
        to_drop = set()
        keep_names = const_kept_names[:]

    # Final selection by name
    desc_tr_selected = desc_tr_df[keep_names].values
    desc_val_selected = pd.DataFrame(desc_val, columns=const_kept_names)[keep_names].values
    desc_te_selected = pd.DataFrame(desc_te, columns=const_kept_names)[keep_names].values

    # 4) Remove exact-zero variance features that may have been created by imputation
    # This can happen when imputation makes a feature constant
    desc_tr_final_df = pd.DataFrame(desc_tr_selected, columns=keep_names)
    final_variances = desc_tr_final_df.var()
    
    # Keep features with variance > 0 (exact zero variance removal)
    # Note: We keep low-variance features as they might be meaningful
    final_keep_mask = final_variances > 0.0
    final_keep_names = [name for name, keep in zip(keep_names, final_keep_mask) if keep]
    zero_var_after_impute = [name for name, keep in zip(keep_names, final_keep_mask) if not keep]
    
    # Count low-variance features being kept
    low_variance_count = np.sum((final_variances > 0.0) & (final_variances < 1e-10))
    
    if zero_var_after_impute:
        logger.info(f"Removed {len(zero_var_after_impute)} exact-zero-variance features after imputation: {zero_var_after_impute}")
    
    if low_variance_count > 0:
        logger.info(f"Keeping {low_variance_count} low-variance features after imputation (variance between 0 and 1e-10)")
    
    # Apply final selection
    desc_tr_final = desc_tr_final_df[final_keep_names].values
    desc_val_final = pd.DataFrame(desc_val_selected, columns=keep_names)[final_keep_names].values
    desc_te_final = pd.DataFrame(desc_te_selected, columns=keep_names)[final_keep_names].values
    
    # Update correlation mask to reflect final selection
    final_corr_mask = np.isin(const_kept_names, final_keep_names)

    # Create preprocessing metadata
    preprocessing_metadata = {
        "n_desc_before_any_selection": len(orig_desc_names),
        "n_desc_after_constant_removal": len(const_kept_names),
        "n_desc_after_corr_removal": len(keep_names),
        "n_desc_after_final_zero_var_removal": len(final_keep_names),
        "constant_features_removed": constant_features,
        "correlated_features_removed": sorted(list(to_drop)),
        "zero_var_after_impute_removed": zero_var_after_impute,
        "selected_features": final_keep_names,
        "imputation_strategy": "median",
        "correlation_threshold": 0.90,
        "orig_desc_names": orig_desc_names,
        "const_kept_names": const_kept_names
    }
    
    return (desc_tr_final, desc_val_final, desc_te_final, final_keep_names, 
            preprocessing_metadata, imputer, constant_mask, final_corr_mask)


def save_preprocessing_objects(out_dir: Path, split_idx: int, preprocessing_metadata: Dict[str, Any],
                             imputer: SimpleImputer, constant_mask: np.ndarray, 
                             corr_mask: np.ndarray, selected_desc_names: List[str]) -> None:
    """Save preprocessing metadata and objects to disk.
    
    Args:
        out_dir: Output directory path
        split_idx: Split index for naming files
        preprocessing_metadata: Metadata dictionary
        imputer: Fitted imputer object
        constant_mask: Boolean mask for constant removal
        corr_mask: Boolean mask for correlation removal
        selected_desc_names: Names of selected descriptors
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    with open(out_dir / f"preprocessing_metadata_split_{split_idx}.json", "w") as f:
        json.dump(preprocessing_metadata, f, indent=2)

    # Save objects
    joblib.dump(imputer, out_dir / f"descriptor_imputer_{split_idx}.pkl")

    # Save boolean masks for reproducibility
    np.save(out_dir / f"constant_mask_{split_idx}.npy", constant_mask)
    np.save(out_dir / f"corr_mask_{split_idx}.npy", corr_mask)

    # Save selected features list
    with open(out_dir / f"split_{split_idx}.txt", "w") as f:
        f.write("\n".join(selected_desc_names))


def load_preprocessing_objects(checkpoint_dir: Path, split_idx: int) -> Optional[Dict[str, Any]]:
    """Load preprocessing metadata and objects from disk.
    
    Args:
        checkpoint_dir: Directory containing preprocessing objects
        split_idx: Split index for loading files
        
    Returns:
        Dictionary containing preprocessing metadata and objects, or None if not found
    """
    try:
        # Load metadata
        metadata_file = checkpoint_dir / f"preprocessing_metadata_split_{split_idx}.json"
        if not metadata_file.exists():
            return None
            
        with open(metadata_file, 'r') as f:
            preprocessing_metadata = json.load(f)
        
        # Load objects
        imputer = joblib.load(checkpoint_dir / f"descriptor_imputer_{split_idx}.pkl")
        constant_mask = np.load(checkpoint_dir / f"constant_mask_{split_idx}.npy")
        corr_mask = np.load(checkpoint_dir / f"corr_mask_{split_idx}.npy")
        
        # Load selected features list
        with open(checkpoint_dir / f"split_{split_idx}.txt", "r") as f:
            selected_desc_names = [line.strip() for line in f.readlines()]
        
        return {
            'preprocessing_metadata': preprocessing_metadata,
            'imputer': imputer,
            'constant_mask': constant_mask,
            'corr_mask': corr_mask,
            'selected_desc_names': selected_desc_names
        }
        
    except Exception as e:
        print(f"Error loading preprocessing objects: {e}")
        return None

