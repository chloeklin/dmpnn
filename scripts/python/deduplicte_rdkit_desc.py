# pip install rdkit-pypi pandas numpy scikit-learn
import re
import numpy as np
import pandas as pd
from rdkit import Chem

# ⬇️ Use your tabular utilities
# Make sure tabular_utils exposes: compute_rdkit_desc, RD_DESC, _init_rdkit_descriptors
from tabular_utils import compute_rdkit_desc, RD_DESC, _init_rdkit_descriptors

# =========================
# Name handling / aliases
# =========================
def _canon(s: str) -> str:
    """Canonicalize dataset/RD names for matching."""
    s = re.sub(r'(_x|_y|_left|_right)$', '', s, flags=re.I)  # strip merge suffix
    s = s.strip().lower()
    return s

def _squeeze(s: str) -> str:
    """Aggressive canonical form: keep only [a-z0-9]."""
    return re.sub(r'[^a-z0-9]+', '', _canon(s))

# dataset name -> RD name (as they appear in RD_DESC), mapped in 'squeezed' space
ALIASES = {
    # Common synonyms seen in your columns
    "slogp": "MolLogP",
    "topopsa(no)": "TPSA",
    "mw": "MolWt",
    "nhbdon": "NumHBDonors",
    "nhbacct": "NumHAcceptors",
    "nrot": "NumRotatableBonds",
    "naromatom": "NumAromaticAtoms",
    "kier2": "Kappa2",
    "kier3": "Kappa3",
    # Add more if you discover additional renamings in your data
}
ALIASES_SQZ = { _squeeze(k): _squeeze(v) for k, v in ALIASES.items() }

def _dataset_to_rd_key(name: str, rd_name_set_sqz: set) -> str | None:
    """
    Map a dataset colname to an RD descriptor key (squeezed).
    Try direct match, then alias.
    """
    dsq = _squeeze(name)
    if dsq in rd_name_set_sqz:
        return dsq
    if dsq in ALIASES_SQZ and ALIASES_SQZ[dsq] in rd_name_set_sqz:
        return ALIASES_SQZ[dsq]
    return None

# =========================
# Helpers
# =========================
def _mol_from_smiles(smi):
    if pd.isna(smi):
        return None
    try:
        return Chem.MolFromSmiles(str(smi))
    except Exception:
        return None

def _equal_numeric(a: pd.Series, b: pd.Series, tol=1e-6) -> bool:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    both_nan = a.isna() & b.isna()
    close = (a - b).abs() <= tol
    return bool((both_nan | close).all())

def drop_y_if_duplicate_of_x(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops all _y columns that are completely identical to their _x partner.
    Keeps _y if values differ from _x.
    """
    df_out = df.copy()
    x_cols = [c for c in df.columns if c.endswith('_x')]
    dropped = []
    for x in x_cols:
        base = x[:-2]
        y = base + '_y'
        if y in df_out.columns and df_out[x].equals(df_out[y]):
            df_out.drop(columns=[y], inplace=True)
            dropped.append(y)
    print(f"[merge] Dropped {len(dropped)} duplicate _y columns.")
    return df_out

# =========================
# RDKit duplicate dropper
# =========================
def drop_columns_duplicate_of_rdkit(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    tol: float = 1e-6,
    verbose: bool = True,
    protect_cols: set | None = None,
):
    """
    Drop columns that duplicate RDKit descriptors computed by YOUR compute_rdkit_desc().
    Only drops columns not in protect_cols.
    """
    protect_cols = protect_cols or set()

    if smiles_col not in df.columns:
        raise KeyError(f"'{smiles_col}' not in dataframe")

    # Prepare RD descriptor names
    _init_rdkit_descriptors()
    if not RD_DESC:
        print("Warning: RDKit descriptors not available or failed to initialize")
        return df.copy(), []
    
    rd_names = [name for name, _ in RD_DESC]
    rd_names_sqz = [_squeeze(n) for n in rd_names]
    rd_name_to_idx = { k: i for i, k in enumerate(rd_names_sqz) }
    rd_name_set_sqz = set(rd_names_sqz)

    # Build molecules once and compute RD matrix
    mols = df[smiles_col].map(_mol_from_smiles)
    rd_mat = np.vstack([compute_rdkit_desc(m) for m in mols])
    rd_df = pd.DataFrame(rd_mat, columns=rd_names, index=df.index)

    # Map dataset columns -> RD index (if any)
    col_to_rd_idx = {}
    for col in df.columns:
        if col in protect_cols:
            continue
        key = _dataset_to_rd_key(col, rd_name_set_sqz)
        if key is not None:
            col_to_rd_idx[col] = rd_name_to_idx[key]

    dropped = []
    for col, ridx in col_to_rd_idx.items():
        rd_col = rd_df.iloc[:, ridx]
        if _equal_numeric(df[col], rd_col, tol=tol):
            dropped.append(col)
            if verbose:
                print(f"[rdkit] drop {col}  (matches RD '{rd_df.columns[ridx]}')")

    if verbose:
        print(f"[rdkit] Dropping {len(dropped)} RDKit-equivalent column(s).")
    return df.drop(columns=dropped), dropped

# =========================
# Leak control (by target root)
# =========================
def _target_root(colname: str) -> str:
    """Return a root token used to block obvious leakage."""
    c = _canon(colname)
    for pref in [
        'k_bond','k_ang','k_dih','theta0','r0','mw','tc','vdw',
        'monomer_length','mw_ratio','mass','charge','epsilon','sigma'
    ]:
        if c.startswith(pref):
            return pref
    return c.split('_')[0]

def drop_same_root_leaks(X: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    roots = {_target_root(t) for t in targets}
    leak_cols = [c for c in X.columns if _target_root(c) in roots]
    if leak_cols:
        print(f"[leak] Dropping {len(leak_cols)} same-root columns: e.g., {leak_cols[:8]}")
        X = X.drop(columns=sorted(set(leak_cols)))
    return X

# =========================
# Orchestrator
# =========================
def prepare_features(
    df: pd.DataFrame,
    targets: list[str] | str = ("TC_x", "VDW"),
    smiles_col: str = "smiles",
    tol: float = 1e-6,
    drop_same_root=True,
):
    """Full pipeline: resolve _x/_y → build X,y → drop RDKit dupes from X → leak control."""
    if isinstance(targets, str):
        targets = [targets]
    for t in targets:
        if t not in df.columns:
            raise KeyError(f"Target '{t}' not found in dataframe")

    # --- 1) Resolve _x/_y duplicates first
    df1 = drop_y_if_duplicate_of_x(df)

    # --- 2) Split out y (protect targets from later drops)
    y = df1[targets].copy()

    # --- 3) Build candidate X (drop targets + obvious identifiers)
    ident_like = [c for c in ["smiles", "ID", "id"] if c in df1.columns]
    X = df1.drop(columns=list(set(targets) | set(ident_like)))

    # --- 4) RDKit duplicate removal (from features ONLY)
    # We need smiles_col available for the RDKit dropper; temporarily reattach it.
    X_tmp = X.copy()
    X_tmp[smiles_col] = df1[smiles_col]
    protect = set()  # (we already removed targets from X; this is here for completeness)
    X_tmp2, dropped = drop_columns_duplicate_of_rdkit(
        X_tmp, smiles_col=smiles_col, tol=tol, verbose=True, protect_cols=protect
    )
    if smiles_col in X_tmp2.columns:
        X_tmp2 = X_tmp2.drop(columns=[smiles_col])
    X = X_tmp2

    # --- 5) Optional: drop same-root leaks relative to chosen targets
    if drop_same_root:
        X = drop_same_root_leaks(X, targets)

    print(f"[done] X shape: {X.shape} | y shape: {y.shape}")
    return X, y

# =========================
# Example usage
# =========================
if __name__ == "__main__":
    df = pd.read_csv("data/tc.csv")

    # Choose your targets explicitly
    TARGETS = ["TC_x", "VDW"]

    X, y = prepare_features(
        df,
        targets=TARGETS,
        smiles_col="smiles",
        tol=1e-6,
        drop_same_root=True,
    )

    # Keep identifiers / admin columns in the final CSV
    admin_cols = [c for c in ["smiles", "WDMPNN_Input"] if c in df.columns]

    # Build one dataframe: [admin] + [targets] + [features]
    df_out = pd.concat([df[admin_cols], y, X], axis=1)

    out_path = "data/tc_preprocessed.csv"
    df_out.to_csv(out_path, index=False)
    print(f"[save] Wrote {out_path} with shape {df_out.shape}")
