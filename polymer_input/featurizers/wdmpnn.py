"""wD-MPNN (weighted D-MPNN) polymer featurizer.

Converts a :class:`~polymer_input.schema.PolymerSpec` into Chemprop's
:class:`~chemprop.data.molgraph.PolymerMolGraph` using the weighted D-MPNN
representation, which models stochastic polymer connectivity via edge and
atom weights.

Edge vocabulary (wD-MPNN-specific):
- Standard chemical bond types (SINGLE / DOUBLE / TRIPLE / AROMATIC)
- Inter-fragment bonds with stochastic weights (w_uv in (0,1])
  representing connectivity probabilities in a polymer ensemble

The existing ``chemprop.featurizers.molgraph.molecule.PolymerMolGraphFeaturizer``
implements the core wD-MPNN logic using a combined RDKit Mol + edge strings.
This module bridges from ``PolymerSpec`` to that featurizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

from polymer_input.featurizers.base import BasePolymerFeaturizer
from polymer_input.mol_utils import build_wdmpnn_mol, extract_scalar_features

from chemprop.data.molgraph import PolymerMolGraph
from chemprop.featurizers.molgraph.molecule import PolymerMolGraphFeaturizer

if TYPE_CHECKING:
    from polymer_input.schema import PolymerSpec


# ---------------------------------------------------------------------------
#  wD-MPNN edge vocabulary
# ---------------------------------------------------------------------------

class WDMPNNEdgeType(IntEnum):
    """Edge types for wD-MPNN.

    Standard chemical bonds plus a weighted polymer edge type.
    """

    SINGLE = 0
    DOUBLE = 1
    TRIPLE = 2
    AROMATIC = 3
    WEIGHTED_POLYMER = 4


# ---------------------------------------------------------------------------
#  Featurizer
# ---------------------------------------------------------------------------

@dataclass
class WDMPNNFeaturizer(BasePolymerFeaturizer):
    """Weighted D-MPNN polymer featurizer.

    Bridges ``PolymerSpec`` → Chemprop ``PolymerMolGraph`` by:

    1. Building a combined RDKit Mol from fragments with ``w_frag``
       atom properties (composition weights).
    2. Generating wDMPNN edge strings (``"R1-R2:w12:w21"``) from
       ``PolymerSpec.connections`` and scalar composition ratios.
    3. Delegating to ``PolymerMolGraphFeaturizer`` which handles wildcard
       tagging, removal, inter-fragment bond creation with stochastic
       weights, and atom/bond featurization.

    Fragment weights are derived from ``spec.scalars`` if composition
    ratios are present (keys like ``ratio_A``, ``ratio_B``), otherwise
    default to ``1 / n_fragments``.

    Parameters
    ----------
    chemprop_featurizer : PolymerMolGraphFeaturizer
        The underlying Chemprop polymer featurizer.

    See Also
    --------
    chemprop.featurizers.molgraph.molecule.PolymerMolGraphFeaturizer
    chemprop.data.molgraph.PolymerMolGraph

    Examples
    --------
    >>> from polymer_input.schema import PolymerSpec, FragmentSpec
    >>> spec = PolymerSpec(
    ...     "peo", [FragmentSpec("[*:1]OCC[*:2]", "A")],
    ...     scalars={"mw": 50000.0},
    ... )
    >>> feat = WDMPNNFeaturizer()
    >>> graph = feat.featurize(spec)
    >>> type(graph).__name__
    'PolymerMolGraph'
    """

    chemprop_featurizer: PolymerMolGraphFeaturizer = field(
        default_factory=PolymerMolGraphFeaturizer
    )

    def featurize(self, spec: PolymerSpec) -> PolymerMolGraph:
        """Convert a PolymerSpec into a Chemprop PolymerMolGraph.

        Parameters
        ----------
        spec : PolymerSpec
            A validated polymer specification.  Fragment SMILES should
            contain numbered wildcards (``[*:1]``, ``[*:2]``, …) for
            proper edge mapping.  Plain ``[*]`` wildcards are auto-numbered.

        Returns
        -------
        PolymerMolGraph
            Chemprop-compatible weighted polymer molecular graph with
            atom weights, edge weights, and degree of polymerization.
        """
        # 1. Build combined mol + edge strings
        combined_mol, edge_strings = build_wdmpnn_mol(spec)

        # 2. Delegate to PolymerMolGraphFeaturizer
        return self.chemprop_featurizer(combined_mol, edge_strings)

    def validate_spec(self, spec: PolymerSpec) -> list[str]:
        """wD-MPNN requires at least one fragment."""
        errors: list[str] = []
        if not spec.fragments:
            errors.append("WDMPNNFeaturizer requires at least one fragment.")
        return errors
