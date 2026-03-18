"""Validation utilities for PolymerSpec objects.

Checks performed:
- Fragment SMILES parseable by RDKit
- Connection indices within bounds
- No empty fragment list
- No malformed topology (self-loops, duplicate edges)
- Scalar values are finite floats
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polymer_input.schema import PolymerSpec


def validate_polymer_spec(spec: PolymerSpec) -> list[str]:
    """Validate a :class:`PolymerSpec` and return a list of error messages.

    Returns an empty list when the spec is valid.

    Parameters
    ----------
    spec : PolymerSpec
        The polymer specification to validate.

    Returns
    -------
    list[str]
        Error messages.  Empty means valid.

    Examples
    --------
    >>> from polymer_input.schema import PolymerSpec, FragmentSpec
    >>> spec = PolymerSpec(polymer_id="ok", fragments=[FragmentSpec(smiles="[*]CC[*]")])
    >>> validate_polymer_spec(spec)
    []
    """
    errors: list[str] = []

    # -- polymer_id --
    if not spec.polymer_id or not isinstance(spec.polymer_id, str):
        errors.append("polymer_id must be a non-empty string.")

    # -- fragments --
    errors.extend(_validate_fragments(spec))

    # -- connections --
    errors.extend(_validate_connections(spec))

    # -- scalars --
    errors.extend(_validate_scalars(spec))

    # -- target --
    if spec.target is not None:
        if not isinstance(spec.target, (int, float)):
            errors.append(f"target must be a number or None, got {type(spec.target).__name__}.")

    return errors


# ---------------------------------------------------------------------------
#  Private helpers
# ---------------------------------------------------------------------------

def _validate_fragments(spec: PolymerSpec) -> list[str]:
    """Check fragment list is non-empty and each SMILES is RDKit-parseable.

    Delegates actual SMILES parsing to
    :func:`polymer_input.mol_utils.parse_fragment_mol` to avoid
    duplicating the partial-sanitization logic.
    """
    from polymer_input.mol_utils import parse_fragment_mol

    errors: list[str] = []
    if not spec.fragments:
        errors.append("fragments list must not be empty.")
        return errors

    for i, frag in enumerate(spec.fragments):
        smi = frag.smiles
        if not smi or not isinstance(smi, str):
            errors.append(f"Fragment [{i}]: smiles must be a non-empty string.")
            continue

        try:
            parse_fragment_mol(smi)
        except ValueError as exc:
            errors.append(
                f"Fragment [{i}] (name={frag.name!r}): {exc}"
            )

    return errors


def _validate_connections(spec: PolymerSpec) -> list[str]:
    """Check connection indices and topology invariants."""
    errors: list[str] = []
    n = len(spec.fragments) if spec.fragments else 0

    seen_edges: set[tuple[int, int]] = set()

    for i, conn in enumerate(spec.connections):
        # Index bounds
        if conn.src < 0 or conn.src >= n:
            errors.append(
                f"Connection [{i}]: src={conn.src} out of range [0, {n})."
            )
        if conn.dst < 0 or conn.dst >= n:
            errors.append(
                f"Connection [{i}]: dst={conn.dst} out of range [0, {n})."
            )

        # Self-loop
        if conn.src == conn.dst:
            errors.append(
                f"Connection [{i}]: self-loop (src==dst=={conn.src})."
            )

        # Duplicate (unordered)
        edge_key = (min(conn.src, conn.dst), max(conn.src, conn.dst))
        if edge_key in seen_edges:
            errors.append(
                f"Connection [{i}]: duplicate edge {conn.src} <-> {conn.dst}."
            )
        seen_edges.add(edge_key)

        # Edge type
        if not conn.edge_type or not isinstance(conn.edge_type, str):
            errors.append(
                f"Connection [{i}]: edge_type must be a non-empty string."
            )

    return errors


def _validate_scalars(spec: PolymerSpec) -> list[str]:
    """Check that scalar values are finite floats."""
    errors: list[str] = []
    if spec.scalars is None:
        return errors

    if not isinstance(spec.scalars, dict):
        errors.append(f"scalars must be a dict or None, got {type(spec.scalars).__name__}.")
        return errors

    import math

    for key, val in spec.scalars.items():
        if not isinstance(key, str):
            errors.append(f"Scalar key {key!r} must be a string.")
        if not isinstance(val, (int, float)):
            errors.append(f"Scalar '{key}': value must be numeric, got {type(val).__name__}.")
        elif math.isnan(val) or math.isinf(val):
            errors.append(f"Scalar '{key}': value must be finite, got {val}.")

    return errors


def validate_or_raise(spec: PolymerSpec) -> None:
    """Validate a PolymerSpec and raise ``ValueError`` on any error.

    Parameters
    ----------
    spec : PolymerSpec
        The polymer specification to validate.

    Raises
    ------
    ValueError
        If validation finds any problems.
    """
    errors = validate_polymer_spec(spec)
    if errors:
        msg = "PolymerSpec validation failed:\n  - " + "\n  - ".join(errors)
        raise ValueError(msg)
