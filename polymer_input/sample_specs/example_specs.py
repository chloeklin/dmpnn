"""Concrete PolymerSpec examples for testing and documentation.

Each example demonstrates a different polymer topology.  All fragment SMILES
use ``[*]`` dummy-atom markers for attachment points.
"""

from polymer_input.schema import FragmentSpec, PolymerConnection, PolymerSpec


# ---------------------------------------------------------------------------
#  Example A: Homopolymer (PEO)
# ---------------------------------------------------------------------------

HOMOPOLYMER_PEO = PolymerSpec(
    polymer_id="peo_001",
    fragments=[
        FragmentSpec(name="A", smiles="[*]OCC[*]"),
    ],
    connections=[],
    topology_type="homopolymer",
    scalars={"mw": 50000.0, "temperature": 298.15},
    target=1.23,
)
"""Poly(ethylene oxide) — simplest case: single repeat unit, no connections."""


# ---------------------------------------------------------------------------
#  Example B: Block copolymer
# ---------------------------------------------------------------------------

BLOCK_COPOLYMER = PolymerSpec(
    polymer_id="block_001",
    fragments=[
        FragmentSpec(name="A", smiles="[*]CC[*]"),
        FragmentSpec(name="B", smiles="[*]OCC[*]"),
    ],
    connections=[
        PolymerConnection(src=0, dst=1, edge_type="polymer_link"),
    ],
    topology_type="block",
    scalars={"mw": 120000.0, "ratio_A": 0.7, "ratio_B": 0.3},
    target=0.85,
)
"""A-B block copolymer with composition ratios stored as scalars."""


# ---------------------------------------------------------------------------
#  Example C: Alternating copolymer
# ---------------------------------------------------------------------------

ALTERNATING_COPOLYMER = PolymerSpec(
    polymer_id="alt_001",
    fragments=[
        FragmentSpec(name="A", smiles="[*]CC[*]"),
        FragmentSpec(name="B", smiles="[*]C(F)C(F)[*]"),
    ],
    connections=[
        PolymerConnection(src=0, dst=1, edge_type="polymer_link"),
    ],
    topology_type="alternating",
    scalars={"temperature": 350.0},
    target=None,
)
"""Alternating copolymer with no target (unlabelled data)."""


# ---------------------------------------------------------------------------
#  Example D: Branched polymer
# ---------------------------------------------------------------------------

BRANCHED_POLYMER = PolymerSpec(
    polymer_id="branch_001",
    fragments=[
        FragmentSpec(name="backbone", smiles="[*]CC[*]"),
        FragmentSpec(name="sidechain", smiles="[*]OCC[*]"),
    ],
    connections=[
        PolymerConnection(src=0, dst=1, edge_type="polymer_link"),
    ],
    topology_type="branched",
    scalars={"mw": 80000.0, "branch_density": 0.15},
    target=2.5,
)
"""Branched polymer with backbone + sidechain fragments."""


# ---------------------------------------------------------------------------
#  Example E: Polymer with rich scalar features
# ---------------------------------------------------------------------------

POLYMER_WITH_SCALARS = PolymerSpec(
    polymer_id="rich_001",
    fragments=[
        FragmentSpec(name="repeat", smiles="[*]c1ccc(C(C)(C)c2ccc([*])cc2)cc1"),
    ],
    connections=[],
    topology_type="homopolymer",
    scalars={
        "mw": 35000.0,
        "mn": 30000.0,
        "pdi": 1.17,
        "temperature": 300.0,
        "dp": 120.0,
    },
    target=418.0,
)
"""Homopolymer (bisphenol-A like) with rich scalar metadata including PDI, DP."""


# ---------------------------------------------------------------------------
#  Convenience list
# ---------------------------------------------------------------------------

ALL_EXAMPLES: list[PolymerSpec] = [
    HOMOPOLYMER_PEO,
    BLOCK_COPOLYMER,
    ALTERNATING_COPOLYMER,
    BRANCHED_POLYMER,
    POLYMER_WITH_SCALARS,
]
"""All example specs in a single list for iteration."""
