"""Abstract base class for polymer featurizers.

All concrete featurizers inherit from :class:`BasePolymerFeaturizer` and
implement ``featurize()`` to convert a :class:`~polymer_input.schema.PolymerSpec`
into a model-specific intermediate representation.

Design notes
------------
* The return type is intentionally ``Any`` at the base level so that each
  featurizer can return its own typed output (``HPGGraphData``,
  ``MolGraph``, ``PolymerMolGraph``, etc.).
* Edge vocabularies are featurizer-specific — the base class does NOT
  prescribe a global edge type set.
* Designed to integrate with Chemprop's existing featurization style.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from polymer_input.schema import PolymerSpec


class BasePolymerFeaturizer(ABC):
    """Abstract interface for converting a PolymerSpec into model inputs.

    Subclasses must implement :meth:`featurize`.  They may also override
    :meth:`validate_spec` to add featurizer-specific checks beyond the
    generic validation in :mod:`polymer_input.validation`.

    Examples
    --------
    >>> class MyFeaturizer(BasePolymerFeaturizer):
    ...     def featurize(self, spec):
    ...         ...  # return model-specific graph data
    """

    @abstractmethod
    def featurize(self, spec: PolymerSpec) -> Any:
        """Convert a PolymerSpec into a model-specific representation.

        Parameters
        ----------
        spec : PolymerSpec
            Validated canonical polymer specification.

        Returns
        -------
        Any
            Model-specific output (e.g. ``HPGGraphData``, ``MolGraph``, …).
        """

    def featurize_batch(self, specs: list[PolymerSpec]) -> list[Any]:
        """Featurize a list of specs.  Default: sequential map.

        Parameters
        ----------
        specs : list[PolymerSpec]
            Batch of specs.

        Returns
        -------
        list[Any]
            List of featurized outputs.
        """
        return [self.featurize(s) for s in specs]

    def validate_spec(self, spec: PolymerSpec) -> list[str]:
        """Optional featurizer-specific validation (on top of generic checks).

        Override in subclasses to add model-specific requirements, e.g.
        ``HPGFeaturizer`` might require at least one fragment.

        Parameters
        ----------
        spec : PolymerSpec
            The spec to validate.

        Returns
        -------
        list[str]
            Error messages (empty = valid).
        """
        return []
