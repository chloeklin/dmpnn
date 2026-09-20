"""Faithful graph construction for the published Han et al. HPG model.

This module accepts an explicit published-style polymer graph specification. It
contains no dataset- or architecture-label logic: callers must supply fragment
connectivity and connection/repetition degrees.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import Atom

from chemprop.data.hpg import HPGMolGraph


HPG_PUBLISHED_ELEMENTS = (
    "C", "N", "O", "S", "H", "F", "Cl", "Br", "I", "Se", "Te", "Si",
    "P", "B", "Ca", "Mg", "Al", "Sb", "Ge", "As",
)
HPG_PUBLISHED_H_COUNTS = (0, 1, 2, 3, 4)
HPG_PUBLISHED_DEGREES = (0, 1, 2, 3, 4, 5, 6)
HPG_PUBLISHED_HYBRIDIZATIONS = ("S", "SP", "SP2", "SP3", "SP3D", "SP3D2")
HPG_PUBLISHED_FORMAL_CHARGES = (-4, -3, -2, -1, 0, 1, 2, 3, 4)
HPG_PUBLISHED_ATOM_FDIM = 49

_MARKERS = ("R", "Q", "T", "U")
_MARKER_TO_MG = {
    "[R]": "[Mg:1]",
    "[Q]": "[Mg:2]",
    "[T]": "[Mg:3]",
    "[U]": "[Mg:4]",
}
_STAR_PATTERN = re.compile(r"\[\*:(\d+)\]|\[\*\]|\*")

_BOND_ORDER = {
    Chem.rdchem.BondType.SINGLE: 1.0,
    Chem.rdchem.BondType.DOUBLE: 2.0,
    Chem.rdchem.BondType.TRIPLE: 3.0,
    Chem.rdchem.BondType.AROMATIC: 1.5,
}


def _one_hot_strict(value, allowable: Iterable) -> list[int]:
    allowable = tuple(allowable)
    if value not in allowable:
        raise ValueError(f"Published HPG feature value {value!r} is not in {allowable}")
    return [int(value == candidate) for candidate in allowable]


def hpg_published_atom_features(atom: Atom) -> np.ndarray:
    """Return the exact 49-dimensional atom vector used by published HPG."""
    features = (
        _one_hot_strict(atom.GetSymbol(), HPG_PUBLISHED_ELEMENTS)
        + _one_hot_strict(atom.GetTotalNumHs(), HPG_PUBLISHED_H_COUNTS)
        + _one_hot_strict(atom.GetDegree(), HPG_PUBLISHED_DEGREES)
        + [int(atom.GetIsAromatic())]
        + _one_hot_strict(str(atom.GetHybridization()), HPG_PUBLISHED_HYBRIDIZATIONS)
        + [int(atom.IsInRing())]
        + _one_hot_strict(atom.GetFormalCharge(), HPG_PUBLISHED_FORMAL_CHARGES)
    )
    result = np.asarray(features, dtype=np.float32)
    if result.shape != (HPG_PUBLISHED_ATOM_FDIM,):
        raise AssertionError(f"Published HPG atom feature shape is {result.shape}, expected (49,)")
    return result


def adapt_psmiles_to_published_markers(smiles: str) -> str:
    """Translate pSMILES stars to published ``[R]/[Q]/[T]/[U]`` markers.

    Mapped stars ``[*:1]`` through ``[*:4]`` retain their map identity. Unmapped
    stars are assigned the first unused markers in SMILES order.
    """
    used = {index + 1 for index, marker in enumerate(_MARKERS) if f"[{marker}]" in smiles}

    def replace(match: re.Match) -> str:
        mapped = match.group(1)
        if mapped is not None:
            marker_index = int(mapped)
            if marker_index not in range(1, 5):
                raise ValueError("Published HPG supports attachment maps 1 through 4 only")
        else:
            marker_index = next((index for index in range(1, 5) if index not in used), 0)
            if marker_index == 0:
                raise ValueError("Published HPG supports at most four attachment markers per fragment")
        if marker_index in used and mapped is None:
            raise AssertionError("Internal attachment-marker assignment collision")
        used.add(marker_index)
        return f"[{_MARKERS[marker_index - 1]}]"

    return _STAR_PATTERN.sub(replace, smiles)


def published_markers_to_mg(smiles: str) -> str:
    """Apply the published marker-to-mapped-Mg substitution."""
    for marker, mg in _MARKER_TO_MG.items():
        smiles = smiles.replace(marker, mg)
    return smiles


@dataclass(frozen=True)
class HPGPublishedConnection:
    """One directed fragment connection in a published-style PolymerGraph."""

    source: int
    destination: int
    degree: float | str | None
    source_endpoint: str | None = None
    destination_endpoint: str | None = None


@dataclass(frozen=True)
class HPGPublishedPolymerGraph:
    """Framework-neutral equivalent of the published ``PolymerGraph`` input."""

    fragments: tuple[str, ...]
    connections: tuple[HPGPublishedConnection, ...]

    def __post_init__(self):
        if not self.fragments:
            raise ValueError("HPG-published requires at least one fragment")
        for connection in self.connections:
            if not 0 <= connection.source < len(self.fragments):
                raise IndexError(f"Invalid source fragment index {connection.source}")
            if not 0 <= connection.destination < len(self.fragments):
                raise IndexError(f"Invalid destination fragment index {connection.destination}")

    @classmethod
    def from_psmiles(
        cls,
        fragments: Iterable[str],
        connections: Iterable[HPGPublishedConnection],
    ) -> HPGPublishedPolymerGraph:
        """Adapt pSMILES fragments without deriving any connectivity or degree."""
        return cls(
            tuple(adapt_psmiles_to_published_markers(smiles) for smiles in fragments),
            tuple(connections),
        )


@dataclass
class HPGPublishedMolGraphFeaturizer:
    """Construct the exact flattened graph consumed by published HPG-GAT."""

    @property
    def d_v(self) -> int:
        return HPG_PUBLISHED_ATOM_FDIM

    @property
    def shape(self) -> tuple[int, int]:
        return self.d_v, 1

    def __call__(self, polymer: HPGPublishedPolymerGraph) -> HPGMolGraph:
        n_fragments = len(polymer.fragments)
        ff_src: list[int] = []
        ff_dst: list[int] = []
        ff_features: list[float] = []
        for connection in polymer.connections:
            degree = 1.0 if connection.degree == "?" or connection.degree is None else float(connection.degree)
            ff_src.append(connection.source)
            ff_dst.append(connection.destination)
            ff_features.append(degree)

        node_features: list[np.ndarray] = [
            np.ones((n_fragments, self.d_v), dtype=np.float32)
        ]
        aa_src: list[int] = []
        aa_dst: list[int] = []
        aa_features: list[float] = []
        af_src: list[int] = []
        af_dst: list[int] = []
        af_features: list[float] = []
        atom_offset = n_fragments

        for fragment_index, fragment_smiles in enumerate(polymer.fragments):
            mol = Chem.MolFromSmiles(published_markers_to_mg(fragment_smiles))
            if mol is None:
                raise ValueError(f"RDKit cannot parse published HPG fragment {fragment_smiles!r}")

            fragment_features = np.asarray(
                [hpg_published_atom_features(atom) for atom in mol.GetAtoms()],
                dtype=np.float32,
            )
            node_features.append(fragment_features)

            for bond in mol.GetBonds():
                begin = atom_offset + bond.GetBeginAtomIdx()
                end = atom_offset + bond.GetEndAtomIdx()
                bond_order = _BOND_ORDER[bond.GetBondType()]
                aa_src.extend((begin, end))
                aa_dst.extend((end, begin))
                aa_features.extend((bond_order, bond_order))

            for local_atom_index in range(mol.GetNumAtoms()):
                af_src.append(atom_offset + local_atom_index)
                af_dst.append(fragment_index)
                af_features.append(1.0)

            atom_offset += mol.GetNumAtoms()

        V = np.concatenate(node_features, axis=0)
        sources = ff_src + aa_src + af_src
        destinations = ff_dst + aa_dst + af_dst
        edge_values = ff_features + aa_features + af_features
        edge_index = np.asarray([sources, destinations], dtype=np.int64).reshape(2, -1)
        E = np.asarray(edge_values, dtype=np.float32).reshape(-1, 1)

        return HPGMolGraph(
            V=V,
            E=E,
            edge_index=edge_index,
            n_fragments=n_fragments,
            n_atoms=atom_offset - n_fragments,
        )
