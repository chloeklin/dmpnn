from __future__ import annotations
import pandas as pd
import torch as th
from typing import List, Sequence, Tuple, Optional, Union

from dgl import shortest_dist
import dgl
from rdkit import Chem

from .base import BaseGraphDataset

# --- simple RDKit featurizers compatible with the provided Graphormer ---
BOND_TYPE_TO_ID = {
    Chem.rdchem.BondType.SINGLE: 1,
    Chem.rdchem.BondType.DOUBLE: 2,
    Chem.rdchem.BondType.TRIPLE: 3,
    Chem.rdchem.BondType.AROMATIC: 4,
}

STEREO_TO_ID = {
    Chem.rdchem.BondStereo.STEREONONE: 0,
    Chem.rdchem.BondStereo.STEREOZ: 1,
    Chem.rdchem.BondStereo.STEREOE: 2,
    Chem.rdchem.BondStereo.STEREOCIS: 3,
    Chem.rdchem.BondStereo.STEREOTRANS: 4,
}

def atom_index(atom: Chem.Atom, offset: int = 0) -> int:
    """
    Map an atom to a single categorical index (int) for nn.Embedding.
    Here we combine atomic number with chirality into a compact space.

    You may expand to richer schemes as needed; keep < num_atoms in config.
    """
    Z = atom.GetAtomicNum()               # 1..118 typically
    ch = int(atom.GetChiralTag())         # small integer 0..3
    return offset + Z * 5 + ch            # simple hashing; fits into 4608 easily

def bond_features(bond: Chem.Bond) -> th.Tensor:
    """
    Produce a 3-dim edge feature to match edge_dim=3:
    [bond_type_id, stereo_id, is_conjugated]
    """
    t = BOND_TYPE_TO_ID.get(bond.GetBondType(), 0)
    s = STEREO_TO_ID.get(bond.GetStereo(), 0)
    c = 1 if bond.GetIsConjugated() else 0
    return th.tensor([t, s, c], dtype=th.long)

def mol_to_dgl_graph(smiles: str, max_spd: int = 511, multi_hop_max_dist: int = 5):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol = Chem.AddHs(mol, addCoords=False) if mol.GetNumConformers() == 0 else mol

    N = mol.GetNumAtoms()
    # edges (undirected -> both directions for consistency with in/out degree)
    src, dst, efeat = [], [], []
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = bond_features(b)
        src.extend([u, v])
        dst.extend([v, u])
        efeat.extend([bf, bf])

    g = dgl.graph((src, dst), num_nodes=N)
    if len(efeat) == 0:
        efeat = [th.tensor([0, 0, 0], dtype=th.long)]  # isolated nodes -> one dummy edge row
        g = dgl.add_self_loop(g)
    g.edata["feat"] = th.stack(efeat) if len(efeat) else th.stack(efeat)

    # node categorical "feat" as indices (shape [N, 1] so .sum(dim=-2) in model works)
    node_ids = [atom_index(mol.GetAtomWithIdx(i)) for i in range(N)]
    g.ndata["feat"] = th.tensor(node_ids, dtype=th.long).unsqueeze(-1)

    # precompute SPD + paths (DGL provides both)
    spd, path = shortest_dist(g, root=None, return_paths=True, cutoff=max_spd)
    g.ndata["spd"] = spd  # [N, N] long; -1 unreachable
    g.ndata["path"] = path  # [N, N, L] edge IDs; -1 padded

    return g


class CSVPolymerDataset(BaseGraphDataset):
    """
    CSV format:
      col 0: SMILES
      following k columns: optional global descriptors (float)
      following t columns: 1+ target columns (regression or classification)
    """
    def __init__(self,
                 path: str,
                 smiles_col: Union[int, str] = 0,
                 num_global_desc: int = 0,
                 target_cols: Sequence[Union[int, str]] = (-1,),
                 task_type: str = "regression",
                 normalize_targets: bool = True,
                 standardize_globals: bool = True,
                 split: Optional[Tuple[List[int], List[int], List[int]]] = None,
                 seed: int = 1):
        super().__init__()
        self.df = pd.read_csv(path)
        self.smiles_col = smiles_col
        self.num_global_desc = num_global_desc
        self.target_cols = list(target_cols)
        self.task_type = task_type
        self.normalize_targets = normalize_targets
        self.standardize_globals = standardize_globals
        self.seed = seed

        # resolve column indices
        def idx(c):
            return self.df.columns.get_loc(c) if isinstance(c, str) else c

        s_idx = idx(smiles_col)
        self.global_slice = slice(s_idx + 1, s_idx + 1 + num_global_desc) if num_global_desc > 0 else None

        self.target_indices = [idx(c) for c in self.target_cols]
        self.num_tasks = len(self.target_indices)

        # optionally compute normalization stats
        if self.global_slice and self.standardize_globals:
            g = self.df.iloc[:, self.global_slice].astype(float)
            self._g_mean = g.mean().values
            self._g_std = g.std().replace(0, 1.0).values
        else:
            self._g_mean, self._g_std = None, None

        if self.normalize_targets and self.task_type == "regression":
            y = self.df.iloc[:, self.target_indices].astype(float)
            self._y_mean = y.mean().values
            self._y_std = y.std().replace(0, 1.0).values
        else:
            self._y_mean, self._y_std = None, None

        # build graphs once
        self.graphs: List[dgl.DGLGraph] = []
        self.labels: List[th.Tensor] = []
        self.globals: List[th.Tensor | None] = []
        self._build_all()

        # optional split precomputed from outside
        self.split = split

    def _build_all(self):
        for _, row in self.df.iterrows():
            smiles = str(row[self.smiles_col])
            g = mol_to_dgl_graph(smiles)

            # targets
            y = th.tensor([float(row[i]) for i in self.target_indices], dtype=th.float32)
            if self._y_mean is not None:
                y = (y - th.tensor(self._y_mean)) / th.tensor(self._y_std)

            # globals
            if self.global_slice:
                gv = th.tensor(row.iloc[self.global_slice].astype(float).values, dtype=th.float32)
                if self._g_mean is not None:
                    gv = (gv - th.tensor(self._g_mean)) / th.tensor(self._g_std)
            else:
                gv = None

            self.graphs.append(g)
            self.labels.append(y)
            self.globals.append(gv)

    def __len__(self): return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx], self.globals[idx]
