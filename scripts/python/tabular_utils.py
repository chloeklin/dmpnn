"""
Tabular Utilities for Molecular Property Prediction

This module provides utilities for processing tabular molecular data, including:
- RDKit descriptor computation
- Atom and bond feature extraction
- Feature engineering and preprocessing
- Model evaluation metrics
"""

# Standard library imports
from typing import List, Tuple, Optional, Dict, Any, Union, Callable

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, roc_auc_score,
    f1_score, log_loss
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
    if not RD_DESC or not RD_DESC_NAMES:
        RD_DESC, RD_DESC_NAMES = _get_rdkit_descriptors()

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
            # Create placeholder with NaN values (will be imputed later)
            placeholder_size = 10  # Arbitrary size for placeholder
            if add_counts:
                placeholder_size += 2  # For atom and bond counts
            feats.append(np.full(placeholder_size, np.nan, dtype=float))
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
            # If any error occurs, add NaN vector (will be imputed)
            feat_size = atom_mat.shape[1] + bond_mat.shape[1] + (2 if add_counts else 0)
            feats.append(np.full(feat_size, np.nan, dtype=float))
    
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

# ------------------------ Feature assembly ---------------------------

def build_features(df: pd.DataFrame, train_idx: List[int], descriptor_columns: List[str], kind: str, use_rdkit: bool, pool: str = 'mean', add_counts: bool = False) -> Tuple[np.ndarray, List[str]]:
    use_descriptor = len(descriptor_columns) > 0
    if not (use_descriptor or use_rdkit):
        use_descriptor = True
        use_rdkit = True

    blocks, names = [], []

    if kind == "homo":
        smiles = df["smiles"].tolist()
        ab = atom_bond_block_from_smiles(smiles, pool=pool, add_counts=add_counts)
        blocks.append(ab); names += [f"AB_{i}" for i in range(ab.shape[1])]
        if use_rdkit:
            rd = rdkit_block_from_smiles(smiles)
            blocks.append(rd); names += [f"RD_{n}" for n in RD_DESC_NAMES]
        if use_descriptor:
            desc = df[descriptor_columns].values
            blocks.append(desc); names += descriptor_columns
            
    else:
        sA, sB = df["smiles_A"].tolist(), df["smiles_B"].tolist()
        fA, fB = df["frac_A"].astype(float).values, df["frac_B"].astype(float).values

        abA = atom_bond_block_from_smiles(sA, pool=pool, add_counts=add_counts)
        abB = atom_bond_block_from_smiles(sB, pool=pool, add_counts=add_counts)
        ab = weighted_average(abA, abB, fA, fB)   # weighting is reasonable for AB too
        blocks.append(ab); names += [f"AB_{i}" for i in range(ab.shape[1])]
        if use_rdkit:
            rdA = rdkit_block_from_smiles(sA)
            rdB = rdkit_block_from_smiles(sB)

            # impute NaNs using train-only means, then fit scaler on train rows only
            train_stack = np.vstack([rdA[train_idx], rdB[train_idx]])
            sc = StandardScaler().fit(train_stack)

            rdA_z = sc.transform(rdA)
            rdB_z = sc.transform(rdB)

            rd = weighted_average(rdA_z, rdB_z, fA, fB)
            blocks.append(rd); names += [f"RD_{n}" for n in RD_DESC_NAMES]

        if use_descriptor:
            desc = df[descriptor_columns].values
            blocks.append(desc); names += descriptor_columns

    if not blocks:
        raise ValueError("No features selected. Use --descriptor and/or --incl_rdkit.")
    X = np.concatenate(blocks, axis=1)
    return X, names


def eval_regression(y_true, y_pred) -> Dict[str, float]:
    return {
        "rmse": mean_squared_error(y_true, y_pred),
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
    return out

def summarize(results: List[Dict[str, float]]) -> Dict[str, float]:
    keys = results[0].keys()
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in results if k in r], dtype=float)
        out[f"{k}_mean"] = np.nanmean(vals)
        out[f"{k}_std"] = np.nanstd(vals)
    return out