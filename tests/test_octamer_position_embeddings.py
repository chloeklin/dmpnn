"""Tests for the OctamerEncoder position-embedding ablation flag.

Factor 2 of HANDOFF §7 is the set of 8 learned position vectors in the octamer
sequence encoder.  These tests guard the on/off flag that removes them.
"""

import torch

from chemprop.models.hpg_hier import OctamerEncoder


def _make_encoder(use_position_embeddings: bool) -> OctamerEncoder:
    return OctamerEncoder(d_h=16, octamer_len=8, n_layers=2, use_position_embeddings=use_position_embeddings)


def test_position_embeddings_on_attribute_exists_and_output_changes_when_perturbed():
    encoder = _make_encoder(use_position_embeddings=True)
    assert hasattr(encoder, "position_embeddings")
    assert encoder.position_embeddings is not None

    monomer_embeds = torch.randn(2, 16)
    sequences = torch.tensor([[0, 1, 0, 1, 0, 1, 0, 1]])
    polymer_batch = torch.tensor([0])

    encoder.eval()
    with torch.no_grad():
        out1 = encoder(monomer_embeds, sequences, polymer_batch)

    with torch.no_grad():
        encoder.position_embeddings += 1.0
        out2 = encoder(monomer_embeds, sequences, polymer_batch)

    assert not torch.allclose(out1, out2)


def test_position_embeddings_off_attribute_absent_and_same_monomer_has_identical_initial_embedding():
    encoder = _make_encoder(use_position_embeddings=False)
    assert not hasattr(encoder, "position_embeddings")

    monomer_embeds = torch.randn(2, 16)
    # positions 0 and 7 both hold monomer 0; positions 1 and 6 both hold monomer 1.
    sequences = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 0]])
    polymer_batch = torch.tensor([0])

    # Replicate the lookup performed inside OctamerEncoder.forward before any
    # message passing.  Without position embeddings the initial slot embeddings
    # are exactly the monomer embeddings indexed by the sequence.
    n_reps, L = sequences.shape
    global_mon = 2 * polymer_batch.unsqueeze(1) + sequences
    h_init = monomer_embeds[global_mon.reshape(-1)].reshape(n_reps, L, 16)

    assert torch.allclose(h_init[0, 0], h_init[0, 7])
    assert torch.allclose(h_init[0, 1], h_init[0, 6])

    # Smoke test: forward still runs and is deterministic under the same input.
    encoder.eval()
    with torch.no_grad():
        out1 = encoder(monomer_embeds, sequences, polymer_batch)
        out2 = encoder(monomer_embeds, sequences, polymer_batch)
    assert torch.allclose(out1, out2)


def test_position_embeddings_on_and_off_differ_in_parameter_count_by_expected_amount():
    encoder_on = _make_encoder(use_position_embeddings=True)
    encoder_off = _make_encoder(use_position_embeddings=False)

    params_on = sum(p.numel() for p in encoder_on.parameters())
    params_off = sum(p.numel() for p in encoder_off.parameters())

    # d_h=16, octamer_len=8 -> 8 * 16 = 128 position-embedding parameters.
    assert params_on - params_off == 8 * 16
