from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from chemprop.data.hpg_hier import TwoStageHPGDatapoint, two_stage_hpg_collate_fn
from chemprop.featurizers.molgraph.hpg_hier import TwoStageHPGFeaturizer
from chemprop.models.hpg_hier import HPGHierMPNN


ROOT_DIR = Path(__file__).resolve().parents[1]


def test_two_stage_hpg_featurizes_and_backpropagates():
    value = pd.read_csv(ROOT_DIR / "data" / "ea_ip.csv", usecols=["WDMPNN_Input"]).iloc[0, 0]
    featurizer = TwoStageHPGFeaturizer()
    graph = featurizer(value)

    assert len(graph.monomer_graphs) == 2
    assert graph.stage2_edge_index.shape == (2, 4)
    assert set(map(tuple, graph.stage2_edge_index.T)) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert graph.stage2_edge_features.shape == (4, 17)
    assert np.isclose(graph.monomer_fracs.sum(), 1.0)
    for monomer_graph in graph.monomer_graphs:
        extras = monomer_graph.V[:, -3:]
        assert extras[:, 0].sum() >= 1
        assert np.all(extras[:, 1:].sum(axis=1) <= 2)

    batch = two_stage_hpg_collate_fn([
        TwoStageHPGDatapoint(graph, np.asarray([0.0], dtype=np.float32)),
        TwoStageHPGDatapoint(graph, np.asarray([1.0], dtype=np.float32)),
    ])
    model = HPGHierMPNN(
        atom_fdim=featurizer.atom_fdim,
        bond_fdim=featurizer.bond_fdim,
        d_h=32,
        stage1_depth=3,
        stage2_depth=2,
    )
    prediction = model(batch[0])
    loss = torch.mean((prediction - batch[2]) ** 2)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)


def test_attachment_features_match_pre_deletion_chemprop_features():
    value = pd.read_csv(ROOT_DIR / "data" / "ea_ip.csv", usecols=["WDMPNN_Input"]).iloc[0, 0]
    featurizer = TwoStageHPGFeaturizer()
    smiles_a, _, _, _ = featurizer._parse_input(value)
    original = Chem.MolFromSmiles(smiles_a)
    ports = featurizer._ports(original, {1, 2})
    graph, _ = featurizer._monomer_graph(smiles_a, {1, 2})
    original_base = featurizer.atom_graph_featurizer.atom_featurizer(original.GetAtomWithIdx(ports[1]))
    assert any(np.allclose(row[: len(original_base)], original_base) for row in graph.V)
