"""Polymer featurizer interfaces and implementations.

Each featurizer converts a :class:`~polymer_input.schema.PolymerSpec` into a
model-specific intermediate representation that can later be converted to
Chemprop-compatible tensors.

Available featurizers:
- :class:`BasePolymerFeaturizer` — abstract interface
- :class:`HPGFeaturizer` — Hierarchical Polymer Graph scaffold
- :class:`PPGFeaturizer` — Periodic Polymer Graph (stub)
- :class:`DMPNNFeaturizer` — D-MPNN polymer featurizer (stub)
- :class:`WDMPNNFeaturizer` — weighted D-MPNN featurizer (stub)
"""

from polymer_input.featurizers.base import BasePolymerFeaturizer
from polymer_input.featurizers.hpg import HPGFeaturizer
from polymer_input.featurizers.ppg import PPGFeaturizer
from polymer_input.featurizers.dmpnn import DMPNNFeaturizer
from polymer_input.featurizers.wdmpnn import WDMPNNFeaturizer

__all__ = [
    "BasePolymerFeaturizer",
    "HPGFeaturizer",
    "PPGFeaturizer",
    "DMPNNFeaturizer",
    "WDMPNNFeaturizer",
]
