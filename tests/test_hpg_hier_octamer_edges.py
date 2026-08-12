"""M2 (hpg_hier_octamer_edges) verification tests.

M2 extends the octamer chain (arm D's 8-slot topology, mean/stoich_weighted readout) with the
17-d monomer-pair junction feature that Stage2Layer already consumes in transition_graph mode.
Rung 4 of ARCHITECTURE_LADDER_2026-08-12.md, see hpg_hier_design/EXPERIMENT_PLAN.md.

Tests (matching the four required verifications in the task spec, §3):
  1. Parameter count differs from arm D by exactly the widened message-MLP input.
  2. Edge features actually reach the layer: non-zero gradient w.r.t. the edge-feature tensor.
  3. Indexing is correct: a slot pair (A->B) receives the same 17-d vector _stage2_edges would
     give Stage2Layer for the A->B monomer edge.
  4. M2 differs from arm D on a real forward pass (features are not silently ignored).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from chemprop.data.hpg_hier import TwoStageHPGDatapoint, two_stage_hpg_collate_fn
from chemprop.featurizers.molgraph.hpg_hier import TwoStageHPGFeaturizer
from chemprop.models.hpg_hier import HPGHierMPNN, OctamerEncoder, OctamerPathLayer

ROOT_DIR = Path(__file__).resolve().parents[1]
_WDMPNN_INPUT = pd.read_csv(ROOT_DIR / "data" / "ea_ip.csv", usecols=["WDMPNN_Input"]).iloc[0, 0]

D_H = 128
D_E = 17


def _make_batch(graph1, graph2=None):
    g2 = graph2 if graph2 is not None else graph1
    return two_stage_hpg_collate_fn([
        TwoStageHPGDatapoint(graph1, np.asarray([0.0], dtype=np.float32)),
        TwoStageHPGDatapoint(g2, np.asarray([1.0], dtype=np.float32)),
    ])


def _arm_d_model(**extra):
    return HPGHierMPNN(
        atom_fdim=75, bond_fdim=14, d_h=D_H, stage1_depth=2, stage2_depth=2,
        stage2_mode="octamer_sequence", stage2_readout="stoich_weighted",
        octamer_len=8, n_random_samples=4, **extra,
    )


def _m2_model(**extra):
    return _arm_d_model(octamer_edge_features=True, **extra)


# ─── 1. Parameter count ─────────────────────────────────────────────────────

def test_param_count_diff_matches_widened_message_mlp_by_hand():
    """arm D's OctamerPathLayer.msg is Linear(d_h, d_h); M2's is Linear(d_h + d_e, d_h).

    Per layer: extra params = (d_h + d_e) * d_h + d_h  -  (d_h * d_h + d_h) = d_e * d_h.
    With stage2_depth=2 layers: total extra = 2 * d_e * d_h.
    """
    arm_d = _arm_d_model()
    m2 = _m2_model()

    n_arm_d = sum(p.numel() for p in arm_d.octamer_encoder.parameters())
    n_m2 = sum(p.numel() for p in m2.octamer_encoder.parameters())

    per_layer_extra = D_E * D_H  # new weight columns only; bias count is unchanged
    n_layers = 2
    expected_diff = per_layer_extra * n_layers
    assert n_m2 - n_arm_d == expected_diff, (
        f"n_m2={n_m2} n_arm_d={n_arm_d} diff={n_m2 - n_arm_d} expected={expected_diff}"
    )


def test_octamer_path_layer_param_count_by_hand():
    """Directly verify the per-layer arithmetic on OctamerPathLayer in isolation."""
    layer_no_edge = OctamerPathLayer(d_h=D_H, edge_dim=None)
    layer_edge = OctamerPathLayer(d_h=D_H, edge_dim=D_E)

    n_no_edge = sum(p.numel() for p in layer_no_edge.parameters())
    n_edge = sum(p.numel() for p in layer_edge.parameters())

    # msg: Linear(d_h, d_h) -> d_h*d_h + d_h weights+bias.
    # msg (edge): Linear(d_h+d_e, d_h) -> (d_h+d_e)*d_h + d_h.
    # update/norm are identical between the two (same d_h), so the whole diff is in msg.
    expected_msg_no_edge = D_H * D_H + D_H
    expected_msg_edge = (D_H + D_E) * D_H + D_H
    assert n_edge - n_no_edge == expected_msg_edge - expected_msg_no_edge == D_E * D_H


# ─── 2. Gradient actually flows through the edge features ──────────────────

def test_edge_features_receive_nonzero_gradient():
    featurizer = TwoStageHPGFeaturizer()
    graph = featurizer(_WDMPNN_INPUT, stage2_mode="octamer_sequence", octamer_len=8, n_random_samples=4)
    batch = _make_batch(graph)

    model = _m2_model()
    atom_embeddings, _ = model.stage1(batch[0].atom_graph)
    monomers = model._pool_monomers(atom_embeddings, batch[0].atom_graph.batch, len(batch[0].monomer_batch))
    h = model.stage2_input(torch.cat([monomers, batch[0].monomer_fracs.unsqueeze(-1)], dim=-1))

    n_polymers = len(batch[0])
    monomer_pair_features = batch[0].stage2_edge_features.reshape(n_polymers, 2, 2, -1).clone()
    monomer_pair_features.requires_grad_(True)

    oct_embeds = model.octamer_encoder(
        h, batch[0].octamer_sequences, batch[0].octamer_polymer_batch,
        monomer_pair_features=monomer_pair_features,
    )
    loss = oct_embeds.sum()
    loss.backward()

    assert monomer_pair_features.grad is not None, "no gradient reached the edge-feature tensor"
    assert torch.any(monomer_pair_features.grad != 0), (
        "edge-feature gradient is all-zero: the features are being silently ignored"
    )


def test_full_model_backward_is_finite_and_octamer_path_layer_weights_get_gradient():
    featurizer = TwoStageHPGFeaturizer()
    graph = featurizer(_WDMPNN_INPUT, stage2_mode="octamer_sequence", octamer_len=8, n_random_samples=4)
    batch = _make_batch(graph)
    model = _m2_model()

    prediction = model(batch[0])
    loss = torch.mean((prediction - batch[2]) ** 2)
    loss.backward()

    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)

    # The edge-aware msg Linear layer inside each OctamerPathLayer must have a non-zero gradient
    # on the columns that consume the edge feature (the last d_e input columns).
    for layer in model.octamer_encoder.layers:
        assert layer.msg.weight.grad is not None
        edge_cols_grad = layer.msg.weight.grad[:, D_H:]
        assert torch.any(edge_cols_grad != 0), "edge-feature weight columns received zero gradient"


# ─── 3. Indexing is correct — matches _stage2_edges exactly ────────────────

def test_monomer_pair_block_matches_stage2_edges_output():
    featurizer = TwoStageHPGFeaturizer()
    graph = featurizer(_WDMPNN_INPUT, stage2_mode="octamer_sequence", octamer_len=8, n_random_samples=4)
    batch = _make_batch(graph)

    n_polymers = len(batch[0])
    block = batch[0].stage2_edge_features.reshape(n_polymers, 2, 2, -1)

    # stage2_edge_index is always [[0,0,1,1],[0,1,0,1]] (+ 2*polymer_idx); stage2_edge_features
    # rows are in that same fixed order, so block[p, m_i, m_j] must equal the row of
    # stage2_edge_features for local pair (m_i, m_j) of polymer p.
    for polymer_idx in range(n_polymers):
        local_edge_index = batch[0].stage2_edge_index[:, 4 * polymer_idx: 4 * polymer_idx + 4] - 2 * polymer_idx
        local_features = batch[0].stage2_edge_features[4 * polymer_idx: 4 * polymer_idx + 4]
        for col in range(4):
            m_i, m_j = int(local_edge_index[0, col]), int(local_edge_index[1, col])
            expected = local_features[col]
            actual = block[polymer_idx, m_i, m_j]
            assert torch.allclose(actual, expected), (
                f"polymer {polymer_idx} pair ({m_i},{m_j}): block={actual} != stage2_edge_features row={expected}"
            )

    # Also check directly against the featurizer's own _stage2_edges for this WDMPNN_Input.
    smiles_a, smiles_b, fracs, rules = featurizer._parse_input(_WDMPNN_INPUT)
    _, ports_a, _ = featurizer._monomer_graph(smiles_a, {1, 2})
    _, ports_b, _ = featurizer._monomer_graph(smiles_b, {3, 4})
    owners = {**{port: 0 for port in ports_a}, **{port: 1 for port in ports_b}}
    edge_index, edge_features = featurizer._stage2_edges(rules, owners, "full")
    for col in range(4):
        m_i, m_j = int(edge_index[0, col]), int(edge_index[1, col])
        np.testing.assert_allclose(
            block[0, m_i, m_j].detach().numpy(), edge_features[col], atol=1e-6,
            err_msg=f"pair ({m_i},{m_j}) mismatch vs direct _stage2_edges call",
        )


def test_gather_path_edge_features_indexes_by_source_target_monomer():
    """A slot pair (A->B) must receive the same vector Stage2Layer would use for A->B."""
    n_polymers = 1
    monomer_pair_features = torch.arange(n_polymers * 2 * 2 * D_E, dtype=torch.float32).reshape(n_polymers, 2, 2, D_E)
    # sequence A,B,A,B (0,1,0,1) -> path edges are (0,1),(1,0),(1,2),(2,1),(2,3),(3,2) in flat node ids
    octamer_sequences = torch.tensor([[0, 1, 0, 1]])
    octamer_polymer_batch = torch.tensor([0])
    L = 4
    edge_index = OctamerEncoder._make_path_edge_index(n_reps=1, L=L, device=torch.device("cpu"))
    gathered = OctamerEncoder._gather_path_edge_features(
        edge_index, octamer_sequences, octamer_polymer_batch, monomer_pair_features, L
    )
    src, dst = edge_index
    for k in range(edge_index.shape[1]):
        pos_src, pos_dst = int(src[k]), int(dst[k])
        m_src = int(octamer_sequences[0, pos_src])
        m_dst = int(octamer_sequences[0, pos_dst])
        expected = monomer_pair_features[0, m_src, m_dst]
        assert torch.equal(gathered[k], expected), (
            f"edge {pos_src}->{pos_dst} (monomers {m_src}->{m_dst}): "
            f"got {gathered[k]} expected {expected}"
        )


# ─── 4. M2 differs from arm D on a real forward pass ───────────────────────

def test_m2_differs_from_arm_d_on_forward_pass():
    featurizer = TwoStageHPGFeaturizer()
    graph = featurizer(_WDMPNN_INPUT, stage2_mode="octamer_sequence", octamer_len=8, n_random_samples=4)
    batch = _make_batch(graph)

    torch.manual_seed(0)
    arm_d = _arm_d_model()
    torch.manual_seed(0)
    m2 = _m2_model()

    # Copy over every parameter arm D and M2 share (everything except the widened msg Linear
    # inside each OctamerPathLayer); leave M2's extra edge-feature weight columns at their
    # freshly initialized (non-zero) values so this is a real, not a vacuous, comparison.
    arm_d_state = arm_d.state_dict()
    m2_state = m2.state_dict()
    for key, value in arm_d_state.items():
        if key.endswith("msg.weight") and "octamer_encoder" in key:
            m2_state[key][:, :D_H] = value
        else:
            m2_state[key] = value
    m2.load_state_dict(m2_state)

    arm_d.eval(); m2.eval()
    with torch.no_grad():
        out_arm_d = arm_d(batch[0])
        out_m2 = m2(batch[0])

    assert not torch.allclose(out_arm_d, out_m2), (
        "M2 output is identical to arm D's — the edge features are not being used"
    )


def test_arm_d_octamer_encoder_has_no_edge_dim_and_is_unaffected():
    """Sanity: arm D's own OctamerEncoder is untouched by the new edge_dim machinery."""
    arm_d = _arm_d_model()
    assert arm_d.octamer_encoder.edge_dim is None
    for layer in arm_d.octamer_encoder.layers:
        assert layer.edge_dim is None
        assert layer.msg.in_features == D_H
