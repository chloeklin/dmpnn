"""Fail-fast guard for the octamer_sequence readout combination.

The OctamerEncoder in chemprop/models/hpg_hier.py is constructed for
stage2_mode == "octamer_sequence" with either "attention" (arm D baseline)
or "stoich_weighted" (arm D mean-pooling).  Unknown readouts still raise.
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


def test_octamer_sequence_with_stoich_weighted_uses_mean_readout():
    model = _dummy_model("stoich_weighted")
    assert model.octamer_encoder is not None
    assert model.octamer_encoder.readout == "mean"
    assert model.octamer_encoder.attention_readout is None


def test_octamer_sequence_with_attention_is_allowed():
    model = _dummy_model("attention")
    assert model.octamer_encoder is not None
    assert model.octamer_encoder.readout == "attention"
    assert model.octamer_encoder.attention_readout is not None


def test_octamer_sequence_guard_rejects_unknown_readout():
    with pytest.raises(ValueError, match="Unknown stage2_readout"):
        _dummy_model("max")
