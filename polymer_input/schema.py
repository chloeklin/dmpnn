"""Canonical polymer specification schema.

Defines the model-agnostic representation for polymer samples:
- ``FragmentSpec``: one monomer / repeat-unit fragment
- ``PolymerConnection``: a polymer-level topology edge between fragments
- ``PolymerSpec``: the top-level sample container

Design notes
------------
* One ``FragmentSpec`` per abstract fragment — polymers are NOT unrolled into
  repeated nodes.  DP / MW belong in ``scalars``.
* ``connections`` encode polymer-level topology (block links, branching, etc.),
  NOT atom-level bonds inside fragments.
* ``scalars`` hold arbitrary numerical metadata (temperature, MW, fractions, …).
* ``target`` is optional — supports both labelled and unlabelled data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FragmentSpec:
    """A single monomer or repeat-unit fragment.

    Parameters
    ----------
    smiles : str
        SMILES string for the fragment, typically with dummy-atom attachment
        points such as ``[*]CC[*]``.
    name : str | None
        Optional human-readable label (e.g. ``"A"``, ``"backbone"``).

    Examples
    --------
    >>> FragmentSpec(smiles="[*]OCC[*]", name="PEO")
    """

    smiles: str
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.smiles, str) or len(self.smiles.strip()) == 0:
            raise ValueError(
                f"FragmentSpec.smiles must be a non-empty string, got {self.smiles!r}"
            )


@dataclass
class PolymerConnection:
    """A polymer-level topology edge between two fragments.

    ``src`` and ``dst`` are indices into ``PolymerSpec.fragments``.

    Parameters
    ----------
    src : int
        Index of the source fragment.
    dst : int
        Index of the destination fragment.
    edge_type : str
        Semantic label for the connection (default ``"polymer_link"``).
        Featurizer-specific meaning is determined downstream.

    Examples
    --------
    >>> PolymerConnection(src=0, dst=1, edge_type="polymer_link")
    """

    src: int
    dst: int
    edge_type: str = "polymer_link"


@dataclass
class PolymerSpec:
    """Canonical representation of a single polymer sample.

    This is the **model-agnostic** input format consumed by all downstream
    featurizers.  It separates raw chemistry from model-specific graph
    construction.

    Parameters
    ----------
    polymer_id : str
        Unique identifier for this sample.
    fragments : list[FragmentSpec]
        Monomer / repeat-unit fragments.  One entry per *abstract* fragment —
        do NOT duplicate for repeat count.
    connections : list[PolymerConnection]
        Polymer-level topology edges between fragments.
    topology_type : str | None
        Optional label: ``"homopolymer"``, ``"block"``, ``"alternating"``,
        ``"random"``, ``"branched"``, etc.
    scalars : dict[str, float] | None
        Optional numerical metadata (MW, temperature, DP, composition
        fractions, …).
    target : float | None
        Optional regression / classification label.

    Examples
    --------
    Homopolymer (PEO):

    >>> PolymerSpec(
    ...     polymer_id="peo_001",
    ...     fragments=[FragmentSpec(smiles="[*]OCC[*]", name="A")],
    ...     connections=[],
    ...     topology_type="homopolymer",
    ...     scalars={"mw": 50000.0, "temperature": 298.15},
    ...     target=1.23,
    ... )

    Block copolymer:

    >>> PolymerSpec(
    ...     polymer_id="block_001",
    ...     fragments=[
    ...         FragmentSpec(smiles="[*]CC[*]", name="A"),
    ...         FragmentSpec(smiles="[*]OCC[*]", name="B"),
    ...     ],
    ...     connections=[PolymerConnection(src=0, dst=1)],
    ...     topology_type="block",
    ...     scalars={"mw": 120000.0, "ratio_A": 0.7, "ratio_B": 0.3},
    ...     target=0.85,
    ... )
    """

    polymer_id: str
    fragments: list[FragmentSpec]
    connections: list[PolymerConnection] = field(default_factory=list)
    topology_type: str | None = None
    scalars: dict[str, float] | None = None
    target: float | None = None

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def n_fragments(self) -> int:
        """Number of fragments in this polymer."""
        return len(self.fragments)

    @property
    def fragment_smiles(self) -> list[str]:
        """List of SMILES strings for all fragments."""
        return [f.smiles for f in self.fragments]

    def get_scalar(self, key: str, default: float | None = None) -> float | None:
        """Safely retrieve a scalar value by key."""
        if self.scalars is None:
            return default
        return self.scalars.get(key, default)

    # ------------------------------------------------------------------
    # Validation entry-point (delegates to validation module)
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Validate this spec and return a list of error messages (empty = OK).

        Delegates to :func:`polymer_input.validation.validate_polymer_spec`.
        """
        from polymer_input.validation import validate_polymer_spec

        return validate_polymer_spec(self)

    # ------------------------------------------------------------------
    # Serialization entry-points (delegate to serialization module)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for JSON."""
        from polymer_input.serialization import polymer_spec_to_dict

        return polymer_spec_to_dict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PolymerSpec:
        """Deserialize from a plain dict."""
        from polymer_input.serialization import polymer_spec_from_dict

        return polymer_spec_from_dict(d)
