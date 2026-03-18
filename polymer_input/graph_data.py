"""Intermediate graph data structures for the HPG (Hierarchical Polymer Graph).

These dataclasses represent the *intermediate* graph construction output that
can later be converted into Chemprop-compatible tensors for message passing.

Design notes
------------
* **No DGL / PyG dependency** — plain Python dataclasses + numpy arrays.
* Three node levels: atom, fragment, (future: component).
* Three edge categories: atom–atom chemical bonds, atom–fragment attachment,
  fragment–fragment polymer topology.
* Edge types are HPG-specific (see :class:`HPGEdgeType`).  Other featurizers
  define their own vocabularies.
* Polymers are NOT unrolled — one fragment node per ``FragmentSpec`` entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
#  HPG edge vocabulary (local to HPG)
# ---------------------------------------------------------------------------

class HPGEdgeType(IntEnum):
    """Edge type vocabulary for the Hierarchical Polymer Graph.

    Chemical bond types mirror Chemprop / RDKit conventions.  The two
    additional types encode the hierarchical structure:

    * ``ATTACHMENT`` — atom ↔ fragment (hierarchical membership)
    * ``POLYMER_LINK`` — fragment ↔ fragment (polymer topology)

    Other featurizers (PPG, wD-MPNN, …) should define their own edge
    vocabularies rather than sharing this one.
    """

    SINGLE = 0
    DOUBLE = 1
    TRIPLE = 2
    AROMATIC = 3
    ATTACHMENT = 4
    POLYMER_LINK = 5


# ---------------------------------------------------------------------------
#  Node dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AtomNode:
    """One atom inside a fragment.

    Parameters
    ----------
    global_idx : int
        Unique atom index across the whole polymer graph.
    fragment_idx : int
        Index of the owning fragment in ``PolymerSpec.fragments``.
    local_idx : int
        Atom index within the RDKit molecule for this fragment.
    atomic_num : int
        Atomic number (e.g. 6 for carbon).
    symbol : str
        Element symbol (e.g. ``"C"``).
    is_attachment : bool
        Whether this atom is an attachment point (bonded to a wildcard ``[*]``).
    features : np.ndarray | None
        Placeholder for Chemprop-compatible atom feature vector.
        Populated downstream by a featurizer.
    """

    global_idx: int
    fragment_idx: int
    local_idx: int
    atomic_num: int
    symbol: str
    is_attachment: bool = False
    features: np.ndarray | None = None


@dataclass
class FragmentNode:
    """One abstract fragment (repeat unit) in the polymer.

    Parameters
    ----------
    fragment_idx : int
        Index into ``PolymerSpec.fragments``.
    name : str | None
        Human-readable label from ``FragmentSpec.name``.
    smiles : str
        Fragment SMILES.
    n_atoms : int
        Number of (non-wildcard) atoms in this fragment.
    atom_global_indices : list[int]
        Global atom indices belonging to this fragment.
    features : np.ndarray | None
        Placeholder for fragment-level feature vector (e.g. aggregated atom
        features, learned embedding, …).  Populated downstream.
    """

    fragment_idx: int
    name: str | None
    smiles: str
    n_atoms: int = 0
    atom_global_indices: list[int] = field(default_factory=list)
    features: np.ndarray | None = None


# ---------------------------------------------------------------------------
#  Edge dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AtomBondEdge:
    """A chemical bond between two atoms inside or across fragments.

    Parameters
    ----------
    src : int
        Global atom index of the source atom.
    dst : int
        Global atom index of the destination atom.
    edge_type : HPGEdgeType
        One of SINGLE / DOUBLE / TRIPLE / AROMATIC.
    features : np.ndarray | None
        Placeholder for Chemprop-compatible bond feature vector.
    """

    src: int
    dst: int
    edge_type: HPGEdgeType
    features: np.ndarray | None = None


@dataclass
class AtomFragmentEdge:
    """A hierarchical attachment edge: atom ↔ fragment.

    Connects an attachment-point atom to its owning fragment node.

    Parameters
    ----------
    atom_idx : int
        Global atom index.
    fragment_idx : int
        Fragment index.
    edge_type : HPGEdgeType
        Always ``HPGEdgeType.ATTACHMENT``.
    """

    atom_idx: int
    fragment_idx: int
    edge_type: HPGEdgeType = HPGEdgeType.ATTACHMENT


@dataclass
class FragmentFragmentEdge:
    """A polymer-topology edge between two fragment nodes.

    Parameters
    ----------
    src : int
        Source fragment index.
    dst : int
        Destination fragment index.
    edge_type : HPGEdgeType
        Always ``HPGEdgeType.POLYMER_LINK``.
    connection_label : str
        Original ``PolymerConnection.edge_type`` label for provenance.
    """

    src: int
    dst: int
    edge_type: HPGEdgeType = HPGEdgeType.POLYMER_LINK
    connection_label: str = "polymer_link"


# ---------------------------------------------------------------------------
#  Top-level HPG graph container
# ---------------------------------------------------------------------------

@dataclass
class HPGGraphData:
    """Intermediate HPG representation for one polymer sample.

    This is the output of :class:`HPGFeaturizer` and is designed as an
    intermediate format that can later be converted to Chemprop-compatible
    tensors (V, E, edge_index, …).

    Parameters
    ----------
    polymer_id : str
        Sample identifier from ``PolymerSpec.polymer_id``.
    atom_nodes : list[AtomNode]
        All atom-level nodes across fragments.
    fragment_nodes : list[FragmentNode]
        Fragment-level nodes (one per ``FragmentSpec``).
    atom_bond_edges : list[AtomBondEdge]
        Chemical bonds between atoms.
    atom_fragment_edges : list[AtomFragmentEdge]
        Hierarchical edges (atom → fragment).
    fragment_fragment_edges : list[FragmentFragmentEdge]
        Polymer topology edges (fragment ↔ fragment).
    scalar_features : dict[str, float] | None
        Pass-through of ``PolymerSpec.scalars``.
    target : float | None
        Pass-through of ``PolymerSpec.target``.

    Examples
    --------
    >>> graph = HPGGraphData(
    ...     polymer_id="peo_001",
    ...     atom_nodes=[...],
    ...     fragment_nodes=[...],
    ...     atom_bond_edges=[...],
    ...     atom_fragment_edges=[...],
    ...     fragment_fragment_edges=[],
    ...     scalar_features={"mw": 50000.0},
    ... )
    >>> graph.n_atoms
    4
    """

    polymer_id: str
    atom_nodes: list[AtomNode] = field(default_factory=list)
    fragment_nodes: list[FragmentNode] = field(default_factory=list)
    atom_bond_edges: list[AtomBondEdge] = field(default_factory=list)
    atom_fragment_edges: list[AtomFragmentEdge] = field(default_factory=list)
    fragment_fragment_edges: list[FragmentFragmentEdge] = field(default_factory=list)
    scalar_features: dict[str, float] | None = None
    target: float | None = None

    # -- convenience properties --

    @property
    def n_atoms(self) -> int:
        """Total number of atom nodes."""
        return len(self.atom_nodes)

    @property
    def n_fragments(self) -> int:
        """Number of fragment nodes."""
        return len(self.fragment_nodes)

    @property
    def n_atom_bonds(self) -> int:
        """Number of atom–atom chemical bond edges."""
        return len(self.atom_bond_edges)

    @property
    def n_attachment_edges(self) -> int:
        """Number of atom–fragment attachment edges."""
        return len(self.atom_fragment_edges)

    @property
    def n_polymer_edges(self) -> int:
        """Number of fragment–fragment polymer edges."""
        return len(self.fragment_fragment_edges)

    def summary(self) -> str:
        """Human-readable one-line summary."""
        return (
            f"HPGGraphData(id={self.polymer_id!r}, "
            f"atoms={self.n_atoms}, fragments={self.n_fragments}, "
            f"bonds={self.n_atom_bonds}, attachments={self.n_attachment_edges}, "
            f"polymer_edges={self.n_polymer_edges})"
        )
