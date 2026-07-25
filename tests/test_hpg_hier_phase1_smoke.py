"""Smoke tests for the three Phase-1 HPG-hier variants.

Tests per variant:
  - featurize + forward + backward: finite loss and finite gradients
  - default flags reproduce baseline HPGHierMPNN output exactly (atol=1e-6)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from chemprop.data.hpg_hier import TwoStageHPGDatapoint, two_stage_hpg_collate_fn
from chemprop.featurizers.molgraph.hpg_hier import TwoStageHPGFeaturizer
from chemprop.models.hpg_hier import HPGHierMPNN


ROOT_DIR = Path(__file__).resolve().parents[1]
_WDMPNN_INPUT = pd.read_csv(ROOT_DIR / "data" / "ea_ip.csv", usecols=["WDMPNN_Input"]).iloc[0, 0]


def _make_batch(graph1, graph2=None):
    g2 = graph2 if graph2 is not None else graph1
    return two_stage_hpg_collate_fn([
        TwoStageHPGDatapoint(graph1, np.asarray([0.0], dtype=np.float32)),
        TwoStageHPGDatapoint(g2, np.asarray([1.0], dtype=np.float32)),
    ])


def _baseline_model(featurizer, **extra):
    return HPGHierMPNN(
        atom_fdim=featurizer.atom_fdim,
        bond_fdim=featurizer.bond_fdim,
        d_h=32, stage1_depth=2, stage2_depth=2,
        **extra,
    )


# ─── Variant 1: hpg_hier_wedge ─────────────────────────────────────────────

def test_wedge_featurize_forward_backward():
    featurizer = TwoStageHPGFeaturizer()
    graph = featurizer(_WDMPNN_INPUT, stage2_edge="full")
    batch = _make_batch(graph)
    model = HPGHierMPNN(
        atom_fdim=featurizer.atom_fdim, bond_fdim=featurizer.bond_fdim,
        d_h=32, stage1_depth=2, stage2_depth=2,
        stage2_edge_weight="multiplier",
    )
    prediction = model(batch[0])
    loss = torch.mean((prediction - batch[2]) ** 2)
    loss.backward()
    assert torch.isfinite(loss), f"Loss not finite: {loss}"
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_wedge_default_reproduces_baseline():
    featurizer = TwoStageHPGFeaturizer()
    graph = featurizer(_WDMPNN_INPUT)
    batch = _make_batch(graph)
    torch.manual_seed(0)
    baseline = _baseline_model(featurizer)
    variant = _baseline_model(featurizer, stage2_edge_weight="feature")
    variant.load_state_dict(baseline.state_dict())
    baseline.eval(); variant.eval()
    with torch.no_grad():
        out_base = baseline(batch[0])
        out_var = variant(batch[0])
    assert torch.allclose(out_base, out_var, atol=1e-6), \
        f"Default wedge differs from baseline:\n{out_base}\n{out_var}"


# ─── Variant 2: hpg_hier_octamer ───────────────────────────────────────────

def test_octamer_featurize_forward_backward():
    featurizer = TwoStageHPGFeaturizer()
    graph = featurizer(_WDMPNN_INPUT, stage2_mode="octamer_sequence", octamer_len=8, n_random_samples=4)
    assert graph.octamer_sequences is not None
    assert graph.octamer_sequences.shape == (4, 8)
    batch = _make_batch(graph)
    assert batch[0].octamer_sequences is not None
    assert batch[0].octamer_sequences.shape == (8, 8)  # 2 polymers * 4 samples
    model = HPGHierMPNN(
        atom_fdim=featurizer.atom_fdim, bond_fdim=featurizer.bond_fdim,
        d_h=32, stage1_depth=2, stage2_depth=2,
        stage2_mode="octamer_sequence", octamer_len=8, n_random_samples=4,
    )
    prediction = model(batch[0])
    loss = torch.mean((prediction - batch[2]) ** 2)
    loss.backward()
    assert torch.isfinite(loss), f"Loss not finite: {loss}"
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_octamer_default_reproduces_baseline():
    featurizer = TwoStageHPGFeaturizer()
    graph = featurizer(_WDMPNN_INPUT)  # stage2_mode="transition_graph"
    assert graph.octamer_sequences is None
    batch = _make_batch(graph)
    assert batch[0].octamer_sequences is None
    torch.manual_seed(0)
    baseline = _baseline_model(featurizer)
    variant = _baseline_model(featurizer, stage2_mode="transition_graph")
    variant.load_state_dict(baseline.state_dict())
    baseline.eval(); variant.eval()
    with torch.no_grad():
        out_base = baseline(batch[0])
        out_var = variant(batch[0])
    assert torch.allclose(out_base, out_var, atol=1e-6), \
        f"Default octamer differs from baseline:\n{out_base}\n{out_var}"


# ─── Variant 3: hpg_hier_junction ──────────────────────────────────────────

def test_junction_featurize_forward_backward():
    featurizer = TwoStageHPGFeaturizer()
    graph = featurizer(_WDMPNN_INPUT, junction_coupling="on")
    assert graph.junction_edge_index is not None
    assert graph.junction_edge_weights is not None
    assert graph.junction_edge_index.shape[0] == 2
    assert graph.junction_edge_weights.shape[0] == graph.junction_edge_index.shape[1]
    batch = _make_batch(graph)
    assert batch[0].junction_edge_index is not None
    model = HPGHierMPNN(
        atom_fdim=featurizer.atom_fdim, bond_fdim=featurizer.bond_fdim,
        d_h=32, stage1_depth=2, stage2_depth=2,
        junction_coupling="on", n_coupling_steps=2,
    )
    prediction = model(batch[0])
    loss = torch.mean((prediction - batch[2]) ** 2)
    loss.backward()
    assert torch.isfinite(loss), f"Loss not finite: {loss}"
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_junction_default_reproduces_baseline():
    featurizer = TwoStageHPGFeaturizer()
    graph = featurizer(_WDMPNN_INPUT)  # junction_coupling="off"
    assert graph.junction_edge_index is None
    batch = _make_batch(graph)
    assert batch[0].junction_edge_index is None
    torch.manual_seed(0)
    baseline = _baseline_model(featurizer)
    variant = _baseline_model(featurizer, junction_coupling="off")
    variant.load_state_dict(baseline.state_dict())
    baseline.eval(); variant.eval()
    with torch.no_grad():
        out_base = baseline(batch[0])
        out_var = variant(batch[0])
    assert torch.allclose(out_base, out_var, atol=1e-6), \
        f"Default junction differs from baseline:\n{out_base}\n{out_var}"
