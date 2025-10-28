# featurize.py
import torch as th
from rdkit import Chem

def _one_hot(idx: int, size: int):
    v = th.zeros(size, dtype=th.float32)
    if 0 <= idx < size:
        v[idx] = 1.0
    return v

def _one_hot_from_set(value, choices):
    size = len(choices)
    v = th.zeros(size, dtype=th.float32)
    try:
        i = choices.index(value)
        v[i] = 1.0
    except ValueError:
        pass
    return v

# ----- Atom features (127) -----
# spec:
# Z: one-hot 1..100 (100)
# degree: one-hot 0..5 (6, clamp>=5)
# formal charge: {-2,-1,0,+1,+2} (5, else zeros)
# chirality: {UNSPECIFIED, TETRA_CW, TETRA_CCW, OTHER} (4)
# num H: 0..4 (5, clamp>=4)
# hybridization: {SP, SP2, SP3, SP3D, SP3D2} (5)
# aromatic: boolean (1)
# mass/100: scalar (1)
def atom_vector(atom: Chem.Atom) -> th.Tensor:
    # atomic number bin (1..100)
    Z = atom.GetAtomicNum()
    z_idx = Z - 1  # shift to 0..99
    z = _one_hot(z_idx if 0 < Z <= 100 else -1, 100)

    # degree 0..5 (clamp)
    deg = min(atom.GetTotalDegree(), 5)
    deg_oh = _one_hot(deg, 6)

    # formal charge {-2,-1,0,+1,+2}
    charge_map = [-2, -1, 0, +1, +2]
    fc = _one_hot_from_set(atom.GetFormalCharge(), charge_map)

    # chirality
    chiral_map = [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        "OTHER",  # bucket for anything else
    ]
    chi = th.zeros(4, dtype=th.float32)
    ct = atom.GetChiralTag()
    if ct in (chiral_map[0], chiral_map[1], chiral_map[2]):
        chi = _one_hot([chiral_map[0], chiral_map[1], chiral_map[2]].index(ct), 4)
    else:
        chi[3] = 1.0

    # num hydrogens 0..4 (clamp)
    nH = min(atom.GetTotalNumHs(includeNeighbors=True), 4)
    h_oh = _one_hot(nH, 5)

    # hybridization
    hyb_choices = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]
    hyb = _one_hot_from_set(atom.GetHybridization(), hyb_choices)

    # aromatic
    arom = th.tensor([1.0 if atom.GetIsAromatic() else 0.0], dtype=th.float32)

    # mass scaled
    mass = th.tensor([atom.GetMass() / 100.0], dtype=th.float32)

    # concat (should be 127 dims)
    feats = th.cat([z, deg_oh, fc, chi, h_oh, hyb, arom, mass], dim=0)
    assert feats.shape[0] == 127, f"Atom feature dim != 127 (got {feats.shape[0]})"
    return feats

# ----- Bond features (12) -----
# One reasonable 12-D design (sum = 12):
# - bond type one-hot: {SINGLE, DOUBLE, TRIPLE, AROMATIC} → 4
# - conjugated: 1
# - in_ring: 1
# - stereo one-hot: {STEREONONE, STEREOANY, E, Z, CIS, TRANS} → 6
def bond_vector(b: Chem.Bond) -> th.Tensor:
    # bond type
    btype_map = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3,
    }
    bt = th.zeros(4, dtype=th.float32)
    bt[btype_map.get(b.GetBondType(), 0)] = 1.0  # default to SINGLE bin if unknown

    conj = th.tensor([1.0 if b.GetIsConjugated() else 0.0], dtype=th.float32)
    ring = th.tensor([1.0 if b.IsInRing() else 0.0], dtype=th.float32)

    stereo_choices = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS,
    ]
    stereo = _one_hot_from_set(b.GetStereo(), stereo_choices)  # 6-d

    feats = th.cat([bt, conj, ring, stereo], dim=0)
    assert feats.shape[0] == 12, f"Bond feature dim != 12 (got {feats.shape[0]})"
    return feats
