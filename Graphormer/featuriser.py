import dgl
import torch as th
from rdkit import Chem
from rdkit.Chem import rdchem

# ------------------------
# Atom feature ID mappers
# ------------------------

def atomic_number_id(atom, Z_cap=100):
    z = atom.GetAtomicNum()  # 0 for "*"
    if 1 <= z <= Z_cap:
        return z - 1              # 0..99
    return Z_cap                  # 100 = UNKNOWN bucket

def formal_charge_id(atom):
    q = int(atom.GetFormalCharge())
    if q < -2 or q > 2:
        return 5                  # UNKNOWN
    return q + 2                  # map -2..+2 -> 0..4

def chirality_id(atom):
    tag = int(atom.GetChiralTag())
    # RDKit: 0=UNSPEC, 1=CW, 2=CCW, others=OTHER(3)
    if tag == 0:  # UNSPECIFIED
        return 0
    if tag == 1:  # TETRA_CW
        return 1
    if tag == 2:  # TETRA_CCW
        return 2
    return 3      # OTHER

def num_hs_id(atom):
    return min(atom.GetTotalNumHs(includeNeighbors=True), 4)  # 0..4

HYB2ID = {
    rdchem.HybridizationType.SP: 0,
    rdchem.HybridizationType.SP2: 1,
    rdchem.HybridizationType.SP3: 2,
    rdchem.HybridizationType.SP3D: 3,
    rdchem.HybridizationType.SP3D2: 4,
}
def hybridization_id(atom):
    return HYB2ID.get(atom.GetHybridization(), 5)  # OTHER=5

def aromatic_id(atom):
    return int(atom.GetIsAromatic())  # 0/1

def atom_features(atom: rdchem.Atom) -> th.Tensor:
    """Return [Z, charge, chirality, numHs, hybridization, aromatic] as long IDs."""
    return th.tensor([
        atomic_number_id(atom),    # 0..100
        formal_charge_id(atom),    # 0..5
        chirality_id(atom),        # 0..3
        num_hs_id(atom),           # 0..4
        hybridization_id(atom),    # 0..5
        aromatic_id(atom),         # 0..1
    ], dtype=th.long)

# ------------------------
# Bond feature ID mappers
# (keep what we used earlier)
# ------------------------

# ---- bond type ----
BOND2ID = {
    rdchem.BondType.SINGLE: 0,   # single
    rdchem.BondType.DOUBLE: 1,   # double
    rdchem.BondType.TRIPLE: 2,   # triple
    rdchem.BondType.AROMATIC: 3, # aromatic
}
def bond_type_id(b: rdchem.Bond) -> int:
    # Map anything unexpected into the closest bucket; default to SINGLE
    return BOND2ID.get(b.GetBondType(), 0)

# ---- stereo ----
# Your categories: none, any, E, Z, cis, trans  -> 0..5
STEREO2ID = {
    rdchem.BondStereo.STEREONONE:  0,  # none
    rdchem.BondStereo.STEREOANY:   1,  # any
    rdchem.BondStereo.STEREOE:     2,  # E
    rdchem.BondStereo.STEREOZ:     3,  # Z
    rdchem.BondStereo.STEREOCIS:   4,  # cis
    rdchem.BondStereo.STEREOTRANS: 5,  # trans
}
def stereo_id(b: rdchem.Bond) -> int:
    return STEREO2ID.get(b.GetStereo(), 0)  # default to "none"

def conjugated_id(b: rdchem.Bond) -> int:
    return int(b.GetIsConjugated())

def in_ring_id(b: rdchem.Bond) -> int:
    return int(b.IsInRing())

def bond_features(b: rdchem.Bond) -> th.Tensor:
    """[bond_type, conjugated, in_ring, stereo] as long IDs."""
    return th.tensor(
        [bond_type_id(b), conjugated_id(b), in_ring_id(b), stereo_id(b)],
        dtype=th.long
    )

# ------------------------
# SMILES → DGL
# ------------------------

def smiles_to_dgl(smiles: str, add_explicit_h: bool = False) -> dgl.DGLGraph:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    if add_explicit_h:
        mol = Chem.AddHs(mol, addCoords=False)

    N = mol.GetNumAtoms()
    nfeats = [atom_features(mol.GetAtomWithIdx(i)) for i in range(N)]

    src, dst, efeats = [], [], []
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = bond_features(b)
        src += [u, v]
        dst += [v, u]
        efeats += [f, f]

    if len(src) == 0:
        src = list(range(N)); dst = list(range(N))
        # bond_type=single(0), conjugated=0, in_ring=0, stereo=none(0)
        efeats = [th.tensor([0, 0, 0, 0], dtype=th.long) for _ in range(N)]

    g = dgl.graph((th.tensor(src), th.tensor(dst)), num_nodes=N)
    g.ndata["feat"] = th.stack(nfeats, dim=0)    # [N, 6] long
    g.edata["feat"] = th.stack(efeats, dim=0)    # [E, 4] long
    return g
