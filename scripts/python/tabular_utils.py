import argparse, os, numpy as np, pandas as pd, torch, torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import models

from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType, BondStereo, BondType

# ---------- D-MPNN ATOM FEATURES ----------
# Ref feature set & sizes: atom: 100 + 6 + 5 + 4 + 5 + 5 + 1 + 1 ; bond: 4 + 1 + 1 + 6
# Atom: type(100), degree(6), formal charge(5), chirality(4), numHs(5), hybridization(5), aromatic(1), mass/100 (1)
Z_MAX = 100  # one-hot length for atomic number (0-99)

def one_hot(k, length, clamp=False):
    if clamp:
        k = max(0, min(length - 1, int(k)))
    v = [0]*length
    if 0 <= int(k) < length:
        v[int(k)] = 1
    return v

def atom_features_dmpnn(atom: Chem.Atom) -> np.ndarray:
    # 1) Atom type (by atomic number), 0..99
    z = atom.GetAtomicNum()
    f_type = one_hot(z, Z_MAX, clamp=True)

    # 2) # bonds (total degree), 0..5 (cap >=5 to 5)
    deg = min(atom.GetTotalDegree(), 5)
    f_deg = one_hot(deg, 6)

    # 3) Formal charge in {-2,-1,0,1,2} -> index 0..4
    charge = int(atom.GetFormalCharge())
    charge = -2 if charge < -2 else 2 if charge > 2 else charge
    f_charge = one_hot(charge + 2, 5)

    # 4) Chirality: Unspecified, Tetra_CW, Tetra_CCW, Other
    ch = atom.GetChiralTag()
    # Map RDKit tags
    # NONE/CHI_UNSPECIFIED -> Unspecified
    # CHI_TETRAHEDRAL_CW -> CW
    # CHI_TETRAHEDRAL_CCW -> CCW
    # others -> Other
    if ch in (Chem.rdchem.ChiralType.CHI_UNSPECIFIED, Chem.rdchem.ChiralType.CHI_OTHER, Chem.rdchem.ChiralType.CHI_UNSPECIFIED):
        idx = 0
    elif ch == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
        idx = 1
    elif ch == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
        idx = 2
    else:
        idx = 3
    f_ch = one_hot(idx, 4)

    # 5) # Hs (total Hs), 0..4 (cap >=4 to 4)
    nH = min(atom.GetTotalNumHs(includeNeighbors=True), 4)
    f_h = one_hot(nH, 5)

    # 6) Hybridization: sp, sp2, sp3, sp3d, sp3d2
    hyb_list = [HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP3,
                HybridizationType.SP3D, HybridizationType.SP3D2]
    hyb_onehot = [int(atom.GetHybridization() == h) for h in hyb_list]

    # 7) Aromaticity (bool)
    f_arom = [int(atom.GetIsAromatic())]

    # 8) Atomic mass scaled
    f_mass = [atom.GetMass() / 100.0]

    feats = np.asarray(f_type + f_deg + f_charge + f_ch + f_h + hyb_onehot + f_arom + f_mass, dtype=np.float32)
    return feats

# ---------- D-MPNN BOND FEATURES ----------
# Bond: type(4), conjugated(1), in_ring(1), stereo(6: none, any, E, Z, cis, trans)
def bond_features_dmpnn(bond: Chem.Bond) -> np.ndarray:
    bt = bond.GetBondType()
    f_type = [
        int(bt == BondType.SINGLE),
        int(bt == BondType.DOUBLE),
        int(bt == BondType.TRIPLE),
        int(bt == BondType.AROMATIC),
    ]
    f_conj = [int(bond.GetIsConjugated())]
    f_ring = [int(bond.IsInRing())]

    stereo = bond.GetStereo()
    stereo_cats = [BondStereo.STEREONONE, BondStereo.STEREOANY,
                   BondStereo.STEREOE, BondStereo.STEREOZ,
                   BondStereo.STEREOCIS, BondStereo.STEREOTRANS]
    f_stereo = [int(stereo == s) for s in stereo_cats]

    feats = np.asarray(f_type + f_conj + f_ring + f_stereo, dtype=np.float32)
    return feats

# ---------- SMILES -> PyG graph ----------
def mol_to_pyg_dmpnn(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Bad SMILES: {smiles}")
    mol = Chem.AddHs(mol, addCoords=False)

    # nodes
    x = [atom_features_dmpnn(a) for a in mol.GetAtoms()]
    x = torch.tensor(np.stack(x, 0), dtype=torch.float)

    # edges (both directions)
    ei, ea = [], []
    for bond in mol.GetBonds():
