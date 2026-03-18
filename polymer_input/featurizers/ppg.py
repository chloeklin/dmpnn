"""PPG (Periodic Polymer Graph) featurizer.

Converts a :class:`~polymer_input.schema.PolymerSpec` into Chemprop's
:class:`~chemprop.data.molgraph.MolGraph` with periodic bonds added between
polymer connection-point atoms.

The existing ``chemprop.featurizers.molgraph.ppg.PPGMolGraphFeaturizer``
already implements the core PPG logic for a single RDKit Mol.  This module
bridges from the canonical ``PolymerSpec`` schema to that featurizer.

Edge vocabulary (PPG-specific):
- Chemical bond types (SINGLE / DOUBLE / TRIPLE / AROMATIC) — standard RDKit
- Periodic bond — a synthetic bond between attachment-point atoms
  (same bond type set, but added between fragments)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

from polymer_input.featurizers.base import BasePolymerFeaturizer
from polymer_input.mol_utils import combine_fragments, extract_scalar_features

from chemprop.data.molgraph import MolGraph
from chemprop.featurizers.molgraph.ppg import PPGMolGraphFeaturizer

if TYPE_CHECKING:
    from polymer_input.schema import PolymerSpec


# ---------------------------------------------------------------------------
#  PPG edge vocabulary (local to PPG)
# ---------------------------------------------------------------------------

class PPGEdgeType(IntEnum):
    """Edge types for the Periodic Polymer Graph.

    Chemical bond types plus one periodic bond type.
    """

    SINGLE = 0
    DOUBLE = 1
    TRIPLE = 2
    AROMATIC = 3
    PERIODIC = 4


# ---------------------------------------------------------------------------
#  Featurizer
# ---------------------------------------------------------------------------

@dataclass
class PPGFeaturizer(BasePolymerFeaturizer):
    """Periodic Polymer Graph featurizer.

    Bridges ``PolymerSpec`` → Chemprop ``MolGraph`` with periodic edges by:

    1. Combining all fragment SMILES into a single RDKit Mol (keeping
       wildcard ``[*]`` atoms for periodic bond detection).
    2. Delegating to ``PPGMolGraphFeaturizer`` which:
       - Adds explicit hydrogens
       - Generates 3D coordinates
       - Identifies nearest-neighbor atoms (bonded to wildcards)
       - Creates periodic bonds between them
       - Featurizes atoms and bonds (including bond-length bins)

    The returned :class:`MolGraph` is directly compatible with Chemprop's
    PPG model.

    Parameters
    ----------
    chemprop_featurizer : PPGMolGraphFeaturizer
        The underlying Chemprop PPG featurizer.

    See Also
    --------
    chemprop.featurizers.molgraph.ppg.PPGMolGraphFeaturizer

    Examples
    --------
    >>> from polymer_input.schema import PolymerSpec, FragmentSpec
    >>> spec = PolymerSpec("peo", [FragmentSpec("[*]OCC[*]", "A")])
    >>> feat = PPGFeaturizer()
    >>> graph = feat.featurize(spec)
    >>> type(graph).__name__
    'MolGraph'
    """

    chemprop_featurizer: PPGMolGraphFeaturizer = field(
        default_factory=PPGMolGraphFeaturizer
    )

    def featurize(self, spec: PolymerSpec) -> MolGraph:
        """Convert a PolymerSpec into a Chemprop MolGraph with periodic bonds.

        Parameters
        ----------
        spec : PolymerSpec
            A validated polymer specification.

        Returns
        -------
        MolGraph
            Chemprop-compatible molecular graph with periodic bonds.
        """
        # 1. Combine fragments (keep wildcards — PPG needs them)
        combined_mol = combine_fragments(spec)

        # 2. Delegate to PPGMolGraphFeaturizer
        #    PPG handles: add Hs, 3D coords, periodic bond detection
        return self.chemprop_featurizer(combined_mol)

    def validate_spec(self, spec: PolymerSpec) -> list[str]:
        """PPG requires at least one fragment with attachment points."""
        errors: list[str] = []
        if not spec.fragments:
            errors.append("PPGFeaturizer requires at least one fragment.")
        return errors
