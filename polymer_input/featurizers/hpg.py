"""HPG (Hierarchical Polymer Graph) featurizer scaffold.

Converts a :class:`~polymer_input.schema.PolymerSpec` into an
:class:`~polymer_input.graph_data.HPGGraphData` intermediate representation
with three edge categories:

1. **atom–atom** chemical bonds (SINGLE / DOUBLE / TRIPLE / AROMATIC)
2. **atom–fragment** attachment edges (ATTACHMENT)
3. **fragment–fragment** polymer topology edges (POLYMER_LINK)

Design notes
------------
* Uses RDKit for fragment parsing and atom/bond enumeration.
* Wildcards (``[*]``, ``[*:1]``, atomic number 0) are detected as
  attachment points but are **not** included as atom nodes.
* Atom feature vectors are left as extension points — this scaffold
  populates ``AtomNode.features = None`` by default.  A future step
  can plug in Chemprop's ``MultiHotAtomFeaturizer``.
* Polymers are NOT unrolled.  One ``FragmentNode`` per ``FragmentSpec``.
* Edge vocabulary is HPG-specific (see :class:`HPGEdgeType`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rdkit import Chem
from rdkit.Chem import BondType

from polymer_input.featurizers.base import BasePolymerFeaturizer
from polymer_input.graph_data import (
    AtomBondEdge,
    AtomFragmentEdge,
    AtomNode,
    FragmentFragmentEdge,
    FragmentNode,
    HPGEdgeType,
    HPGGraphData,
)

if TYPE_CHECKING:
    from polymer_input.schema import PolymerSpec


# ---------------------------------------------------------------------------
#  RDKit BondType -> HPGEdgeType mapping
# ---------------------------------------------------------------------------

_BOND_TYPE_MAP: dict[BondType, HPGEdgeType] = {
    BondType.SINGLE: HPGEdgeType.SINGLE,
    BondType.DOUBLE: HPGEdgeType.DOUBLE,
    BondType.TRIPLE: HPGEdgeType.TRIPLE,
    BondType.AROMATIC: HPGEdgeType.AROMATIC,
}


def _rdkit_bond_to_hpg(bond_type: BondType) -> HPGEdgeType:
    """Map an RDKit BondType to the HPG edge vocabulary.

    Falls back to SINGLE for exotic bond types.
    """
    return _BOND_TYPE_MAP.get(bond_type, HPGEdgeType.SINGLE)


# ---------------------------------------------------------------------------
#  HPG Featurizer
# ---------------------------------------------------------------------------

class HPGFeaturizer(BasePolymerFeaturizer):
    """Build an :class:`HPGGraphData` from a :class:`PolymerSpec`.

    This is a **scaffold** — it constructs the full graph topology but
    leaves atom/bond feature vectors as ``None`` placeholders.  To populate
    features, extend this class or post-process the returned graph.

    Parameters
    ----------
    include_hydrogens : bool
        If True, add explicit hydrogens to fragments before graph
        construction (default False).

    Examples
    --------
    >>> from polymer_input.schema import PolymerSpec, FragmentSpec
    >>> spec = PolymerSpec(
    ...     polymer_id="peo",
    ...     fragments=[FragmentSpec(smiles="[*]OCC[*]", name="A")],
    ... )
    >>> feat = HPGFeaturizer()
    >>> graph = feat.featurize(spec)
    >>> graph.n_atoms  # O, C, C  (wildcards excluded)
    3
    """

    def __init__(self, include_hydrogens: bool = False) -> None:
        self.include_hydrogens = include_hydrogens

    # ------------------------------------------------------------------
    #  Main entry point
    # ------------------------------------------------------------------

    def featurize(self, spec: PolymerSpec) -> HPGGraphData:
        """Convert a PolymerSpec into an HPGGraphData.

        Steps:
        1. Parse each fragment SMILES with RDKit.
        2. Build atom nodes (excluding wildcards).
        3. Build atom–atom bond edges (intra-fragment chemical bonds).
        4. Identify attachment-point atoms and build atom–fragment edges.
        5. Build fragment–fragment edges from ``PolymerSpec.connections``.

        Parameters
        ----------
        spec : PolymerSpec
            A validated polymer specification.

        Returns
        -------
        HPGGraphData
        """
        atom_nodes: list[AtomNode] = []
        fragment_nodes: list[FragmentNode] = []
        atom_bond_edges: list[AtomBondEdge] = []
        atom_fragment_edges: list[AtomFragmentEdge] = []

        global_atom_offset = 0  # running counter for global atom indices

        for frag_idx, frag_spec in enumerate(spec.fragments):
            mol = self._parse_fragment(frag_spec.smiles)
            if mol is None:
                raise ValueError(
                    f"RDKit failed to parse fragment [{frag_idx}] "
                    f"SMILES: {frag_spec.smiles!r}"
                )

            # Identify wildcard vs core atoms
            wildcard_indices: set[int] = set()
            attachment_core_indices: set[int] = set()
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    wildcard_indices.add(atom.GetIdx())
                    # Mark neighbours as attachment points
                    for nbr in atom.GetNeighbors():
                        attachment_core_indices.add(nbr.GetIdx())

            # Build mapping: local RDKit idx -> global idx (wildcards excluded)
            local_to_global: dict[int, int] = {}
            frag_atom_globals: list[int] = []

            for atom in mol.GetAtoms():
                if atom.GetIdx() in wildcard_indices:
                    continue  # skip wildcard pseudo-atoms

                g_idx = global_atom_offset
                local_to_global[atom.GetIdx()] = g_idx

                atom_nodes.append(
                    AtomNode(
                        global_idx=g_idx,
                        fragment_idx=frag_idx,
                        local_idx=atom.GetIdx(),
                        atomic_num=atom.GetAtomicNum(),
                        symbol=atom.GetSymbol(),
                        is_attachment=(atom.GetIdx() in attachment_core_indices),
                        features=None,  # TODO: plug in Chemprop atom featurizer
                    )
                )
                frag_atom_globals.append(g_idx)
                global_atom_offset += 1

            # Fragment node
            fragment_nodes.append(
                FragmentNode(
                    fragment_idx=frag_idx,
                    name=frag_spec.name,
                    smiles=frag_spec.smiles,
                    n_atoms=len(frag_atom_globals),
                    atom_global_indices=frag_atom_globals,
                    features=None,  # TODO: fragment-level features
                )
            )

            # Atom–atom bond edges (intra-fragment chemical bonds)
            for bond in mol.GetBonds():
                u_local = bond.GetBeginAtomIdx()
                v_local = bond.GetEndAtomIdx()

                # Skip bonds involving wildcards
                if u_local in wildcard_indices or v_local in wildcard_indices:
                    continue

                u_global = local_to_global[u_local]
                v_global = local_to_global[v_local]
                etype = _rdkit_bond_to_hpg(bond.GetBondType())

                atom_bond_edges.append(
                    AtomBondEdge(
                        src=u_global,
                        dst=v_global,
                        edge_type=etype,
                        features=None,  # TODO: plug in Chemprop bond featurizer
                    )
                )

            # Atom–fragment attachment edges
            for atom_node in atom_nodes[global_atom_offset - len(frag_atom_globals):]:
                if atom_node.is_attachment:
                    atom_fragment_edges.append(
                        AtomFragmentEdge(
                            atom_idx=atom_node.global_idx,
                            fragment_idx=frag_idx,
                        )
                    )

        # Fragment–fragment polymer topology edges
        fragment_fragment_edges = [
            FragmentFragmentEdge(
                src=conn.src,
                dst=conn.dst,
                edge_type=HPGEdgeType.POLYMER_LINK,
                connection_label=conn.edge_type,
            )
            for conn in spec.connections
        ]

        return HPGGraphData(
            polymer_id=spec.polymer_id,
            atom_nodes=atom_nodes,
            fragment_nodes=fragment_nodes,
            atom_bond_edges=atom_bond_edges,
            atom_fragment_edges=atom_fragment_edges,
            fragment_fragment_edges=fragment_fragment_edges,
            scalar_features=spec.scalars,
            target=spec.target,
        )

    # ------------------------------------------------------------------
    #  Featurizer-specific validation
    # ------------------------------------------------------------------

    def validate_spec(self, spec: PolymerSpec) -> list[str]:
        """HPG-specific checks on top of generic validation."""
        errors: list[str] = []
        if not spec.fragments:
            errors.append("HPGFeaturizer requires at least one fragment.")
        return errors

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _parse_fragment(self, smiles: str) -> Chem.Mol | None:
        """Parse a fragment SMILES, optionally adding explicit Hs.

        Parameters
        ----------
        smiles : str
            Fragment SMILES (may contain wildcards like ``[*]``).

        Returns
        -------
        Chem.Mol | None
            Parsed molecule, or None on failure.
        """
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None

        # Partial sanitization (skip kekulization for wildcard SMILES)
        try:
            Chem.SanitizeMol(
                mol,
                sanitizeOps=(
                    Chem.SanitizeFlags.SANITIZE_ALL
                    ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                ),
            )
        except Exception:
            return None

        if self.include_hydrogens:
            mol = Chem.AddHs(mol)

        return mol
