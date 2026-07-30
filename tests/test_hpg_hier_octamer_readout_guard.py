"""Fail-fast guard for the unimplemented octamer_sequence + stoich_weighted combination.

The OctamerEncoder in chemprop/models/hpg_hier.py is only constructed when
stage2_mode == "octamer_sequence" and stage2_readout == "attention".  Using
"octamer_sequence" with any other readout silently falls back to the 2-node
baseline transition-graph path.  These tests ensure both entry points raise a
ValueError instead.
"""
from pathlib import Path

import pytest

from chemprop.featurizers.molgraph.hpg_hier import TwoStageHPGFeaturizer
from chemprop.models.hpg_hier import HPGHierMPNN


ROOT_DIR = Path(__file__).resolve().parents[1]


def _dummy_model(stage2_readout: str) -> HPGHierMPNN:
    featurizer = TwoStageHPGFeaturizer()
    return HPGHierMPNN(
        atom_fdim=featurizer.atom_fdim,
        bond_fdim=featurizer.bond_fdim,
        d_h=32,
        stage1_depth=2,
        stage2_depth=2,
        stage2_mode="octamer_sequence",
        stage2_readout=stage2_readout,
        octamer_len=8,
        n_random_samples=4,
    )


def test_octamer_sequence_with_stoich_weighted_raises():
    with pytest.raises(ValueError, match="octamer_sequence with readout 'stoich_weighted' is not implemented"):
        _dummy_model("stoich_weighted")


def test_octamer_sequence_with_attention_is_allowed():
    model = _dummy_model("attention")
    assert model.octamer_encoder is not None


def test_octamer_sequence_guard_mentions_handoff_arm_d():
    with pytest.raises(ValueError, match="See HANDOFF §7 \\(arm D\\)"):
        _dummy_model("stoich_weighted")
