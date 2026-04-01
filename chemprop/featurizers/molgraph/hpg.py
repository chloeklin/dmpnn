"""HPG (Hierarchical Polymer Graph) featurizer.

Converts a polymer (given as fragment SMILES + connections) into an
:class:`~chemprop.data.hpg.HPGMolGraph` — a flat graph mixing fragment-level
and atom-level nodes with three edge types, matching the original HPG-GAT paper.

Atom features exactly replicate the original HPG-GAT 49-dim encoding:
  20 symbol + 5 H-count + 7 degree + 1 aromatic + 6 hybridization
  + 1 ring + 9 formal-charge = **49**.
(E/Z stereo is NOT used in the hierarchical graph — see mol2dgl_single.)
Fragment nodes receive ``ones(d_v)`` features (as in the original).
Edge features are 1-D scalars: bond order for atom–atom, 1.0 for
atom→fragment, and ``degree`` for fragment–fragment edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import Atom, Bond

from chemprop.data.hpg import HPGMolGraph


# ---------------------------------------------------------------------------
#  Original HPG atom feature helpers  (from HPG/src/smiles_utils.py)
# ---------------------------------------------------------------------------

# Use atomic numbers 1-100 to match Chemprop v1, plus an "other" category
_HPG_ATOMIC_NUMS = list(range(1, 101))  # 100 elements
_HPG_H_NUMS = [0, 1, 2, 3, 4]                   # 5
_HPG_DEGREES = [0, 1, 2, 3, 4, 5, 6]            # 7
_HPG_HYBRIDIZATIONS = ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2"]  # 6
_HPG_FORMAL_CHARGES = [-4, -3, -2, -1, 0, 1, 2, 3, 4]            # 9

# 101 (100 + 1 for "other") + 5 + 7 + 1 + 6 + 1 + 9 = 130
# NOTE: E/Z stereo (2-dim) is NOT used in the hierarchical graph.
HPG_ATOM_FDIM = 130


def _one_of_k_with_unk(x, allowable: list) -> List[int]:
    """One-hot with unknown: last position is 1 if x not in allowable."""
    if x in allowable:
        return [int(x == v) for v in allowable] + [0]
    else:
        return [0] * len(allowable) + [1]


def _one_of_k(x, allowable: list) -> List[int]:
    """Strict one-hot: raises if *x* is not in *allowable*."""
    if x not in allowable:
        raise ValueError(f"HPG featurizer: {x!r} not in {allowable}")
    return [int(x == v) for v in allowable]


def _hpg_atom_features(atom: Atom) -> np.ndarray:
    """130-dim base features for a single atom (expanded to match Chemprop v1)."""
    feats = (
        _one_of_k_with_unk(atom.GetAtomicNum(), _HPG_ATOMIC_NUMS)  # 101 dims (100 + unk)
        + _one_of_k(atom.GetTotalNumHs(), _HPG_H_NUMS)             # 5 dims
        + _one_of_k(atom.GetDegree(), _HPG_DEGREES)                # 7 dims
        + [int(atom.GetIsAromatic())]                              # 1 dim
        + _one_of_k(str(atom.GetHybridization()), _HPG_HYBRIDIZATIONS)  # 6 dims
        + [int(atom.IsInRing())]                                   # 1 dim
        + _one_of_k(atom.GetFormalCharge(), _HPG_FORMAL_CHARGES)   # 9 dims
    )
    return np.array(feats, dtype=np.float32)


def _hpg_atom_features_for_mol(mol: Chem.Mol) -> np.ndarray:
    """49-dim features for every atom in *mol* (matching mol2dgl_single)."""
    return np.array([_hpg_atom_features(a) for a in mol.GetAtoms()], dtype=np.float32)


# Bond-order lookup matching the original HPG implementation
_BOND_ORDER = {
    Chem.rdchem.BondType.SINGLE: 1.0,
    Chem.rdchem.BondType.DOUBLE: 2.0,
    Chem.rdchem.BondType.TRIPLE: 3.0,
    Chem.rdchem.BondType.AROMATIC: 1.5,
}


@dataclass
class HPGMolGraphFeaturizer:
    """Build an :class:`HPGMolGraph` from fragment SMILES and connections.

    Wildcard atoms ([*], [R], [Q], etc.) are **removed** from the atom graph.
    They define fragment connectivity at the connection level only; they are
    not featurized as atom nodes.  Non-standard polymer wildcard notation
    ([R], [Q], [T], [U]) is first normalized to the standard [*] so RDKit
    parses it as atomic number 0, which is then filtered out.

    Parameters
    ----------
    wildcard_replacements : dict[str, str]
        Map non-standard wildcard notation to the RDKit-parseable ``[*]``.
    """

    wildcard_replacements: dict[str, str] = field(default_factory=lambda: {
        "[R]": "[*]", "[Q]": "[*]", "[T]": "[*]", "[U]": "[*]",
    })

    def __post_init__(self):
        self.d_v = HPG_ATOM_FDIM  # 49

    @property
    def shape(self) -> tuple[int, int]:
        """(node_feature_dim, edge_feature_dim)"""
        return self.d_v, 1

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def __call__(
        self,
        fragment_smiles: list[str],
        connections: list[tuple[int, int, float]] | None = None,
        frag_fracs: np.ndarray | None = None,
    ) -> HPGMolGraph:
        """Featurize a polymer into an HPGMolGraph.

        Parameters
        ----------
        fragment_smiles : list[str]
            SMILES for each fragment / repeat unit.
        connections : list[tuple[int, int, float]] | None
            Each entry is ``(src_frag_idx, dst_frag_idx, degree)``.
            If *None*, fragments are connected linearly with degree 1.0.
        frag_fracs : np.ndarray | None
            Monomer fractions ``[n_fragments]``, one per fragment.
            Used for fraction-weighted pooling in HPG_frac variants.
            If *None*, no fractions are stored in the graph.

        Returns
        -------
        HPGMolGraph
        """
        n_frags = len(fragment_smiles)

        # --- 1. Build fragment-fragment edges ---
        if connections is None:
            if n_frags == 1:
                # Homopolymer: self-loop with degree 1.0 (matching original HPG)
                connections = [(0, 0, 1.0)]
            else:
                # Linear chain
                connections = [(i, i + 1, 1.0) for i in range(n_frags - 1)]

        ff_src, ff_dst, ff_feat = [], [], []
        for src_f, dst_f, deg in connections:
            # Handle unknown degree (matching original HPG polyG.py:84-85)
            if deg == "?" or deg is None:
                deg = 1.0
            ff_src.append(src_f)
            ff_dst.append(dst_f)
            ff_feat.append(float(deg))

        # --- 2. Parse each fragment and collect atom features + bond edges ---
        frag_node_feats = np.ones((n_frags, self.d_v), dtype=np.float32)

        all_atom_feats: list[np.ndarray] = []
        aa_src, aa_dst, aa_feat = [], [], []  # atom-atom
        af_src, af_dst, af_feat = [], [], []  # atom→fragment
        atom_offset = n_frags  # atoms start after fragment nodes

        for frag_idx, smi in enumerate(fragment_smiles):
            clean_smi = self._clean_smiles(smi)
            mol = Chem.MolFromSmiles(clean_smi)
            if mol is None:
                raise ValueError(f"RDKit cannot parse fragment SMILES: {smi!r}")

            # Identify real (non-wildcard) atoms; wildcards have atomic num 0
            real_local_indices = [
                a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 0
            ]
            # Map old local index → new global index (only for real atoms)
            old_to_global = {
                old: atom_offset + new
                for new, old in enumerate(real_local_indices)
            }
            n_real = len(real_local_indices)

            # Atom features (49-dim, wildcards excluded)
            for old_idx in real_local_indices:
                all_atom_feats.append(_hpg_atom_features(mol.GetAtomWithIdx(old_idx)))

            # Atom-atom bond edges (bidirectional; skip any bond touching a wildcard)
            for bond in mol.GetBonds():
                u_old = bond.GetBeginAtomIdx()
                v_old = bond.GetEndAtomIdx()
                if u_old not in old_to_global or v_old not in old_to_global:
                    continue  # wildcard endpoint — skip
                u = old_to_global[u_old]
                v = old_to_global[v_old]
                bo = _BOND_ORDER.get(bond.GetBondType(), 1.0)
                aa_src.extend([u, v])
                aa_dst.extend([v, u])
                aa_feat.extend([bo, bo])

            # Atom→fragment edges (directed: real atom → owning fragment)
            for old_idx in real_local_indices:
                af_src.append(old_to_global[old_idx])
                af_dst.append(frag_idx)
                af_feat.append(1.0)

            atom_offset += n_real

        # --- 3. Assemble into HPGMolGraph ---
        n_atoms_total = atom_offset - n_frags

        if n_atoms_total > 0:
            atom_feats = np.array(all_atom_feats, dtype=np.float32)
        else:
            atom_feats = np.empty((0, self.d_v), dtype=np.float32)

        V = np.concatenate([frag_node_feats, atom_feats], axis=0)

        # Edges: fragment-fragment, then atom-atom, then atom→fragment
        all_src = ff_src + aa_src + af_src
        all_dst = ff_dst + aa_dst + af_dst
        all_efeat = ff_feat + aa_feat + af_feat

        n_edges = len(all_src)
        if n_edges > 0:
            edge_index = np.array([all_src, all_dst], dtype=np.int64)
            E = np.array(all_efeat, dtype=np.float32).reshape(-1, 1)
        else:
            edge_index = np.empty((2, 0), dtype=np.int64)
            E = np.empty((0, 1), dtype=np.float32)

        return HPGMolGraph(
            V=V,
            E=E,
            edge_index=edge_index,
            n_fragments=n_frags,
            n_atoms=n_atoms_total,
            frag_fracs=frag_fracs,
        )

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _clean_smiles(self, smi: str) -> str:
        """Replace wildcard placeholders with valid atoms for RDKit parsing."""
        for old, new in self.wildcard_replacements.items():
            smi = smi.replace(old, new)
        return smi
