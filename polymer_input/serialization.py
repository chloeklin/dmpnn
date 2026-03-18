"""JSON / JSONL-friendly serialization for PolymerSpec.

Provides ``to_dict`` and ``from_dict`` converters that produce plain Python
dicts suitable for ``json.dumps`` / ``json.loads``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from polymer_input.schema import FragmentSpec, PolymerConnection, PolymerSpec


# ---------------------------------------------------------------------------
#  to_dict / from_dict
# ---------------------------------------------------------------------------

def polymer_spec_to_dict(spec: PolymerSpec) -> dict[str, Any]:
    """Convert a :class:`PolymerSpec` to a JSON-serializable dict.

    Parameters
    ----------
    spec : PolymerSpec
        The polymer specification to serialize.

    Returns
    -------
    dict[str, Any]
        A plain dict with all fields serialized.

    Examples
    --------
    >>> from polymer_input.schema import PolymerSpec, FragmentSpec
    >>> spec = PolymerSpec("id1", [FragmentSpec("[*]CC[*]", "A")])
    >>> polymer_spec_to_dict(spec)
    {'polymer_id': 'id1', 'fragments': [{'smiles': '[*]CC[*]', 'name': 'A'}], ...}
    """
    return {
        "polymer_id": spec.polymer_id,
        "fragments": [
            {"smiles": f.smiles, "name": f.name} for f in spec.fragments
        ],
        "connections": [
            {"src": c.src, "dst": c.dst, "edge_type": c.edge_type}
            for c in spec.connections
        ],
        "topology_type": spec.topology_type,
        "scalars": spec.scalars,
        "target": spec.target,
    }


def polymer_spec_from_dict(d: dict[str, Any]) -> PolymerSpec:
    """Reconstruct a :class:`PolymerSpec` from a plain dict.

    Handles both already-parsed Python objects and JSON-encoded strings
    for ``fragments``, ``connections``, and ``scalars`` fields.

    Parameters
    ----------
    d : dict[str, Any]
        The serialized representation (e.g. from ``polymer_spec_to_dict``
        or loaded from JSON / JSONL).

    Returns
    -------
    PolymerSpec

    Raises
    ------
    KeyError
        If required fields are missing.
    """
    fragments_raw = _maybe_parse_json(d.get("fragments", []))
    connections_raw = _maybe_parse_json(d.get("connections", []))
    scalars_raw = _maybe_parse_json(d.get("scalars"))

    fragments = [
        FragmentSpec(
            smiles=f["smiles"],
            name=f.get("name"),
        )
        for f in fragments_raw
    ]

    connections = [
        PolymerConnection(
            src=int(c["src"]),
            dst=int(c["dst"]),
            edge_type=c.get("edge_type", "polymer_link"),
        )
        for c in connections_raw
    ]

    scalars: dict[str, float] | None = None
    if scalars_raw is not None:
        scalars = {str(k): float(v) for k, v in scalars_raw.items()}

    target_raw = d.get("target")
    target = float(target_raw) if target_raw is not None else None

    return PolymerSpec(
        polymer_id=str(d["polymer_id"]),
        fragments=fragments,
        connections=connections,
        topology_type=d.get("topology_type"),
        scalars=scalars,
        target=target,
    )


# ---------------------------------------------------------------------------
#  Batch I/O helpers
# ---------------------------------------------------------------------------

def save_jsonl(specs: list[PolymerSpec], path: str | Path) -> None:
    """Write a list of PolymerSpec objects to a JSONL file.

    Parameters
    ----------
    specs : list[PolymerSpec]
        Specs to serialize.
    path : str | Path
        Output file path.
    """
    path = Path(path)
    with path.open("w") as fh:
        for spec in specs:
            fh.write(json.dumps(polymer_spec_to_dict(spec)) + "\n")


def load_jsonl(path: str | Path) -> list[PolymerSpec]:
    """Load PolymerSpec objects from a JSONL file.

    Parameters
    ----------
    path : str | Path
        Input file path.

    Returns
    -------
    list[PolymerSpec]
    """
    path = Path(path)
    specs: list[PolymerSpec] = []
    with path.open() as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed to parse JSON on line {line_no} of {path}: {exc}"
                ) from exc
            specs.append(polymer_spec_from_dict(d))
    return specs


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _maybe_parse_json(value: Any) -> Any:
    """If *value* is a JSON string, parse it; otherwise return as-is."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value
