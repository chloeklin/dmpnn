"""Polymer input abstraction layer for Chemprop-based polymer property prediction.

This package provides a canonical polymer specification schema, validation,
serialization, parsing, and featurizer interfaces designed to integrate with
the existing Chemprop codebase.

Key components:
- ``schema``: Canonical ``PolymerSpec``, ``FragmentSpec``, ``PolymerConnection``
- ``validation``: RDKit-based SMILES and topology validation
- ``serialization``: JSON/JSONL-friendly ``to_dict`` / ``from_dict``
- ``parsing``: Configurable CSV/dict/JSONL parsers with schema mapping
- ``graph_data``: HPG intermediate graph data structures
- ``featurizers``: Abstract and concrete polymer featurizer interfaces
"""

from polymer_input.schema import FragmentSpec, PolymerConnection, PolymerSpec
from polymer_input.validation import validate_polymer_spec
from polymer_input.serialization import polymer_spec_to_dict, polymer_spec_from_dict
from polymer_input.parsing import PolymerParser, SchemaMapping
from polymer_input.mol_utils import (
    combine_fragments,
    remove_wildcards_and_cap,
    build_wdmpnn_mol,
    extract_scalar_features,
    collect_scalar_keys,
)

__all__ = [
    "FragmentSpec",
    "PolymerConnection",
    "PolymerSpec",
    "validate_polymer_spec",
    "polymer_spec_to_dict",
    "polymer_spec_from_dict",
    "PolymerParser",
    "SchemaMapping",
    "combine_fragments",
    "remove_wildcards_and_cap",
    "build_wdmpnn_mol",
    "extract_scalar_features",
    "collect_scalar_keys",
]
