"""D-MPNN polymer featurizer.

Converts a :class:`~polymer_input.schema.PolymerSpec` into Chemprop's
:class:`~chemprop.data.molgraph.MolGraph` using the standard D-MPNN
molecular graph representation (atom-level message passing on directed
bond graph).

For homopolymers, this is equivalent to featurizing a single repeat-unit
fragment as a small molecule.  For copolymers, fragments are combined into
a single molecular graph with wildcards removed.

Edge vocabulary (DMPNN-specific):
- Standard chemical bond types only (SINGLE / DOUBLE / TRIPLE / AROMATIC)
- No hierarchical or periodic edges — purely atom-level message passing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

from polymer_input.featurizers.base import BasePolymerFeaturizer
from polymer_input.mol_utils import (
    combine_fragments,
    extract_scalar_features,
    remove_wildcards_and_cap,
)

from chemprop.data.molgraph import MolGraph
from chemprop.featurizers.molgraph.molecule import SimpleMoleculeMolGraphFeaturizer

if TYPE_CHECKING:
    from polymer_input.schema import PolymerSpec


# ---------------------------------------------------------------------------
#  DMPNN edge vocabulary (standard molecular bonds only)
# ---------------------------------------------------------------------------

class DMPNNEdgeType(IntEnum):
    """Edge types for D-MPNN: standard chemical bonds only."""

    SINGLE = 0
    DOUBLE = 1
    TRIPLE = 2
    AROMATIC = 3


# ---------------------------------------------------------------------------
#  Featurizer
# ---------------------------------------------------------------------------

@dataclass
class DMPNNFeaturizer(BasePolymerFeaturizer):
    """D-MPNN polymer featurizer.

    Bridges ``PolymerSpec`` → Chemprop ``MolGraph`` by:

    1. Combining all fragment SMILES into a single RDKit Mol.
    2. Removing wildcard attachment-point atoms.
    3. Delegating to ``SimpleMoleculeMolGraphFeaturizer`` for atom/bond
       featurization and graph construction.

    The returned :class:`MolGraph` is directly compatible with Chemprop's
    D-MPNN message-passing model.

    Parameters
    ----------
    chemprop_featurizer : SimpleMoleculeMolGraphFeaturizer
        The underlying Chemprop featurizer.  Defaults to the standard
        multi-hot atom/bond featurizer.

    See Also
    --------
    chemprop.featurizers.molgraph.molecule.SimpleMoleculeMolGraphFeaturizer

    Examples
    --------
    >>> from polymer_input.schema import PolymerSpec, FragmentSpec
    >>> spec = PolymerSpec("peo", [FragmentSpec("[*]OCC[*]", "A")])
    >>> feat = DMPNNFeaturizer()
    >>> graph = feat.featurize(spec)
    >>> type(graph).__name__
    'MolGraph'
    """

    chemprop_featurizer: SimpleMoleculeMolGraphFeaturizer = field(
        default_factory=SimpleMoleculeMolGraphFeaturizer
    )

    def featurize(self, spec: PolymerSpec) -> MolGraph:
        """Convert a PolymerSpec into a standard Chemprop MolGraph.

        Parameters
        ----------
        spec : PolymerSpec
            A validated polymer specification.

        Returns
        -------
        MolGraph
            Chemprop-compatible molecular graph.
        """
        # 1. Combine fragments into a single RDKit Mol
        combined_mol = combine_fragments(spec)

        # 2. Remove wildcard atoms
        clean_mol = remove_wildcards_and_cap(combined_mol)

        # 3. Delegate to Chemprop's featurizer
        return self.chemprop_featurizer(clean_mol)

    def validate_spec(self, spec: PolymerSpec) -> list[str]:
        """DMPNN requires at least one fragment."""
        errors: list[str] = []
        if not spec.fragments:
            errors.append("DMPNNFeaturizer requires at least one fragment.")
        return errors
