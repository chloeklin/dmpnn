"""Provenance tests for run_hpg_generalization variant resolution.

These tests do not train; they construct the resolved variant dict directly and
verify that CLI overrides are reflected in the sidecar-facing record.
"""

import argparse

import pytest

from scripts.python.run_hpg_generalization import _resolve_variant, _VARIANT_FLAGS


def _args(**overrides) -> argparse.Namespace:
    defaults = {
        "stage1_pool": "sum",
        "stage2_depth": 2,
        "stage2_edge": "full",
        "stage2_readout": None,
        "octamer_len": 8,
        "n_random_samples": 16,
        "n_coupling_steps": 2,
        "octamer_position_embeddings": "on",
    }
    return argparse.Namespace(**{**defaults, **overrides})


def test_resolve_variant_default_readout():
    args = _args()
    for token, preset in _VARIANT_FLAGS.items():
        resolved = _resolve_variant(token, args)
        assert resolved["stage2_readout"] == preset["stage2_readout"]


def test_resolve_variant_stage2_readout_override():
    """Arm D runs as hpg_hier_octamer with --stage2_readout stoich_weighted."""
    args = _args(stage2_readout="stoich_weighted")
    resolved = _resolve_variant("hpg_hier_octamer", args)
    assert resolved["stage2_readout"] == "stoich_weighted"


def test_resolve_variant_stage2_readout_override_attention():
    args = _args(stage2_readout="attention")
    resolved = _resolve_variant("hpg_hier", args)
    assert resolved["stage2_readout"] == "attention"


def test_resolve_variant_octamer_edge_features_default_false():
    """Variants without octamer_edge_features in their preset record False."""
    for token in ("hpg_hier", "hpg_hier_octamer"):
        resolved = _resolve_variant(token, _args())
        assert resolved["octamer_edge_features"] is False


def test_resolve_variant_octamer_edge_features_m2_true():
    """The M2 preset explicitly enables junction edge features."""
    resolved = _resolve_variant("hpg_hier_octamer_edges", _args())
    assert resolved["octamer_edge_features"] is True


def test_resolve_variant_does_not_mutate_preset():
    original = dict(_VARIANT_FLAGS["hpg_hier_octamer"])
    _resolve_variant("hpg_hier_octamer", _args(stage2_readout="stoich_weighted"))
    assert _VARIANT_FLAGS["hpg_hier_octamer"] == original
