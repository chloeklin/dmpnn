"""Dataset parsing layer: build PolymerSpec objects from dicts, CSV rows, JSONL.

Supports configurable schema mappings so that heterogeneous column naming
conventions across different datasets can all map to the canonical
:class:`PolymerSpec` fields.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from polymer_input.schema import PolymerSpec
from polymer_input.serialization import polymer_spec_from_dict


# ---------------------------------------------------------------------------
#  Schema mapping
# ---------------------------------------------------------------------------

@dataclass
class SchemaMapping:
    """Maps external field names to canonical :class:`PolymerSpec` fields.

    Each attribute is the name of the column / key in the source data that
    corresponds to the matching ``PolymerSpec`` field.  ``None`` means the
    field is absent from the source.

    Parameters
    ----------
    polymer_id : str
        Source key for ``PolymerSpec.polymer_id``.
    fragments : str
        Source key for the fragments list (a list of dicts or a JSON string).
    connections : str
        Source key for the connections list.
    topology_type : str | None
        Source key for topology type.
    scalars : str | None
        Source key for the scalars dict.
    target : str | None
        Source key for the target value.

    Examples
    --------
    >>> mapping = SchemaMapping(
    ...     polymer_id="sample_id",
    ...     fragments="fragment_smiles",
    ...     connections="connections",
    ...     topology_type="topology",
    ...     scalars="scalars",
    ...     target="y",
    ... )
    """

    polymer_id: str = "polymer_id"
    fragments: str = "fragments"
    connections: str = "connections"
    topology_type: str | None = "topology_type"
    scalars: str | None = "scalars"
    target: str | None = "target"


# ---------------------------------------------------------------------------
#  Default mapping
# ---------------------------------------------------------------------------

DEFAULT_MAPPING = SchemaMapping()


# ---------------------------------------------------------------------------
#  Parser
# ---------------------------------------------------------------------------

class PolymerParser:
    """Parses raw data rows into :class:`PolymerSpec` objects.

    Supports:
    - Python dicts (already in memory)
    - CSV rows (as dicts from ``csv.DictReader``)
    - JSONL rows (as dicts from ``json.loads``)

    Fields like ``fragments``, ``connections``, and ``scalars`` can arrive as
    either already-parsed Python objects **or** JSON strings stored in CSV
    columns — the parser handles both transparently.

    Parameters
    ----------
    mapping : SchemaMapping
        Column / key name mapping.  Defaults to identity mapping.

    Examples
    --------
    >>> parser = PolymerParser()
    >>> spec = parser.parse_row({
    ...     "polymer_id": "peo",
    ...     "fragments": [{"smiles": "[*]OCC[*]", "name": "A"}],
    ...     "connections": [],
    ...     "target": 1.23,
    ... })
    >>> spec.polymer_id
    'peo'
    """

    def __init__(self, mapping: SchemaMapping | None = None) -> None:
        self.mapping = mapping or DEFAULT_MAPPING

    # ---------------------------------------------------------------
    #  Core: parse one row
    # ---------------------------------------------------------------

    def parse_row(self, row: dict[str, Any]) -> PolymerSpec:
        """Parse a single dict-like row into a :class:`PolymerSpec`.

        Parameters
        ----------
        row : dict[str, Any]
            A dict whose keys match the configured :class:`SchemaMapping`.

        Returns
        -------
        PolymerSpec

        Raises
        ------
        KeyError
            If a required field is missing.
        ValueError
            If a field cannot be parsed.
        """
        m = self.mapping

        canonical: dict[str, Any] = {
            "polymer_id": row[m.polymer_id],
            "fragments": _get_field(row, m.fragments, required=True),
            "connections": _get_field(row, m.connections, required=False) or [],
        }

        if m.topology_type and m.topology_type in row:
            canonical["topology_type"] = row[m.topology_type]

        if m.scalars and m.scalars in row:
            canonical["scalars"] = row[m.scalars]

        if m.target and m.target in row:
            val = row[m.target]
            canonical["target"] = val if val is None or val == "" else val

        # Empty-string target -> None
        if canonical.get("target") == "":
            canonical["target"] = None

        return polymer_spec_from_dict(canonical)

    # ---------------------------------------------------------------
    #  Batch helpers
    # ---------------------------------------------------------------

    def parse_dicts(self, rows: list[dict[str, Any]]) -> list[PolymerSpec]:
        """Parse a list of dicts into PolymerSpec objects.

        Parameters
        ----------
        rows : list[dict]
            Rows of data.

        Returns
        -------
        list[PolymerSpec]
        """
        return [self.parse_row(r) for r in rows]

    def parse_csv(self, path: str | Path) -> list[PolymerSpec]:
        """Parse a CSV file into PolymerSpec objects.

        Parameters
        ----------
        path : str | Path
            Path to a CSV file.

        Returns
        -------
        list[PolymerSpec]
        """
        path = Path(path)
        specs: list[PolymerSpec] = []
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for line_no, row in enumerate(reader, 2):  # header = line 1
                try:
                    specs.append(self.parse_row(row))
                except (KeyError, ValueError) as exc:
                    raise ValueError(
                        f"Failed to parse row on line {line_no} of {path}: {exc}"
                    ) from exc
        return specs

    def parse_jsonl(self, path: str | Path) -> list[PolymerSpec]:
        """Parse a JSONL file into PolymerSpec objects.

        Parameters
        ----------
        path : str | Path
            Path to a JSONL file.

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
                    specs.append(self.parse_row(d))
                except (json.JSONDecodeError, KeyError, ValueError) as exc:
                    raise ValueError(
                        f"Failed to parse line {line_no} of {path}: {exc}"
                    ) from exc
        return specs

    def iter_jsonl(self, path: str | Path) -> Iterator[PolymerSpec]:
        """Lazily iterate over a JSONL file yielding PolymerSpec objects.

        Parameters
        ----------
        path : str | Path
            Path to a JSONL file.

        Yields
        ------
        PolymerSpec
        """
        path = Path(path)
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                yield self.parse_row(d)


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _get_field(
    row: dict[str, Any], key: str, *, required: bool = False
) -> Any:
    """Retrieve a field from a row, auto-parsing JSON strings.

    Parameters
    ----------
    row : dict
        The source row.
    key : str
        The key to look up.
    required : bool
        If True, raise KeyError when the key is missing.

    Returns
    -------
    Any
        The parsed value, or None if not found and not required.
    """
    if key not in row:
        if required:
            raise KeyError(f"Required field '{key}' not found in row.")
        return None

    value = row[key]

    # Auto-parse JSON strings (common in CSV columns)
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    return value
