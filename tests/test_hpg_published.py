"""Fidelity tests for the isolated HPG-published implementation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from rdkit import Chem

from chemprop.data.hpg import BatchHPGMolGraph
from chemprop.featurizers.molgraph.hpg_published import (
    HPG_PUBLISHED_ATOM_FDIM,
    HPGPublishedConnection,
    HPGPublishedMolGraphFeaturizer,
    HPGPublishedPolymerGraph,
    adapt_psmiles_to_published_markers,
    hpg_published_atom_features,
    published_markers_to_mg,
)
from chemprop.models.hpg_published import HPGPublishedGATNet


def _load_reference_smiles_utils():
    path = Path(__file__).parents[1] / "HPG" / "src" / "smiles_utils.py"
    spec = importlib.util.spec_from_file_location("hpg_reference_smiles_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reference_atom_features(atom, reference):
    return np.concatenate([
        reference.atom_symbol_HNums(atom),
        reference.atom_degree(atom),
        reference.atom_Aroma(atom),
        reference.atom_Hybrid(atom),
        reference.atom_ring(atom),
        reference.atom_FC(atom),
    ]).astype(np.float32)


def _fragment_edges(graph):
    mask = (
        (graph.edge_index[0] < graph.n_fragments)
        & (graph.edge_index[1] < graph.n_fragments)
    )
    return graph.edge_index[:, mask], graph.E[mask]


def test_published_atom_features_match_reference():
    reference = _load_reference_smiles_utils()
    molecules = (
        "CCO",
        "c1cc(F)ccc1Cl",
        "N#CC(=O)O",
        "[Mg:1]CCO[Mg:2]",
        "CS(=O)(=O)N",
        "CBr",
        "B(O)O",
        "CP(=O)(O)O",
        "[SiH4]",
    )
    for smiles in molecules:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        for atom in mol.GetAtoms():
            actual = hpg_published_atom_features(atom)
            expected = _reference_atom_features(atom, reference)
            assert actual.shape == (HPG_PUBLISHED_ATOM_FDIM,)
            np.testing.assert_array_equal(actual, expected)


def test_psmiles_adapter_preserves_attachment_identity_and_bonds():
    input_smiles = "[*:1]CCO[*:2]"
    markers = adapt_psmiles_to_published_markers(input_smiles)
    assert markers == "[R]CCO[Q]"
    mg_smiles = published_markers_to_mg(markers)
    assert mg_smiles == "[Mg:1]CCO[Mg:2]"

    polymer = HPGPublishedPolymerGraph.from_psmiles(
        [input_smiles],
        [HPGPublishedConnection(0, 0, 7.0)],
    )
    graph = HPGPublishedMolGraphFeaturizer()(polymer)

    assert graph.n_fragments == 1
    assert graph.n_atoms == 5
    assert graph.V.shape == (6, 49)
    np.testing.assert_array_equal(graph.V[0], np.ones(49, dtype=np.float32))

    mol = Chem.MolFromSmiles(mg_smiles)
    mg_local_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == "Mg"]
    assert mg_local_indices == [0, 4]
    for local_index in mg_local_indices:
        expected = hpg_published_atom_features(mol.GetAtomWithIdx(local_index))
        np.testing.assert_array_equal(graph.V[1 + local_index], expected)

    atom_to_fragment = {
        (int(source), int(destination), float(value))
        for (source, destination), value in zip(graph.edge_index.T, graph.E[:, 0])
        if source >= graph.n_fragments and destination < graph.n_fragments
    }
    assert atom_to_fragment == {(index, 0, 1.0) for index in range(1, 6)}

    edges = {
        (int(source), int(destination), float(value))
        for (source, destination), value in zip(graph.edge_index.T, graph.E[:, 0])
    }
    assert (1, 2, 1.0) in edges
    assert (2, 1, 1.0) in edges
    assert (4, 5, 1.0) in edges
    assert (5, 4, 1.0) in edges


def test_unmapped_psmiles_stars_are_assigned_in_order():
    assert adapt_psmiles_to_published_markers("*CCO*") == "[R]CCO[Q]"
    assert adapt_psmiles_to_published_markers("[*]CC([*])C[*]") == "[R]CC([Q])C[T]"


def test_homopolymer_self_loop_and_degree():
    polymer = HPGPublishedPolymerGraph.from_psmiles(
        ["[*:1]CC[*:2]"],
        [HPGPublishedConnection(0, 0, 12.0)],
    )
    graph = HPGPublishedMolGraphFeaturizer()(polymer)
    edge_index, edge_features = _fragment_edges(graph)
    np.testing.assert_array_equal(edge_index, [[0], [0]])
    np.testing.assert_array_equal(edge_features, [[12.0]])


def test_two_fragment_connections_are_directed_and_not_invented():
    polymer = HPGPublishedPolymerGraph.from_psmiles(
        ["[*:1]CC[*:2]", "[*:3]CO[*:4]"],
        [
            HPGPublishedConnection(0, 0, 4.0),
            HPGPublishedConnection(0, 1, 1.0),
            HPGPublishedConnection(1, 1, 3.0),
        ],
    )
    graph = HPGPublishedMolGraphFeaturizer()(polymer)
    edge_index, edge_features = _fragment_edges(graph)
    np.testing.assert_array_equal(edge_index, [[0, 0, 1], [0, 1, 1]])
    np.testing.assert_array_equal(edge_features, [[4.0], [1.0], [3.0]])


def test_multifragment_graph_matches_supplied_connections():
    polymer = HPGPublishedPolymerGraph.from_psmiles(
        ["[*:1]CC[*:2]", "[*:1]CO[*:2]", "[*:1]CN[*:2]"],
        [
            HPGPublishedConnection(0, 0, 5.0),
            HPGPublishedConnection(0, 1, 1.0),
            HPGPublishedConnection(1, 1, 3.0),
            HPGPublishedConnection(1, 2, 1.0),
            HPGPublishedConnection(2, 2, 2.0),
        ],
    )
    graph = HPGPublishedMolGraphFeaturizer()(polymer)
    edge_index, edge_features = _fragment_edges(graph)
    np.testing.assert_array_equal(
        edge_index,
        [[0, 0, 1, 1, 2], [0, 1, 1, 2, 2]],
    )
    np.testing.assert_array_equal(edge_features[:, 0], [5.0, 1.0, 3.0, 1.0, 2.0])


def test_changing_degree_changes_only_fragment_edge_feature():
    make = lambda degree: HPGPublishedPolymerGraph.from_psmiles(
        ["[*:1]CC[*:2]"],
        [HPGPublishedConnection(0, 0, degree)],
    )
    featurizer = HPGPublishedMolGraphFeaturizer()
    graph_2 = featurizer(make(2.0))
    graph_9 = featurizer(make(9.0))

    np.testing.assert_array_equal(graph_2.V, graph_9.V)
    np.testing.assert_array_equal(graph_2.edge_index, graph_9.edge_index)
    np.testing.assert_array_equal(graph_2.E[1:], graph_9.E[1:])
    assert graph_2.E[0, 0] == 2.0
    assert graph_9.E[0, 0] == 9.0


def test_published_model_forward_and_parameter_interface():
    featurizer = HPGPublishedMolGraphFeaturizer()
    polymer = HPGPublishedPolymerGraph.from_psmiles(
        ["[*:1]CCO[*:2]"],
        [HPGPublishedConnection(0, 0, 5.0)],
    )
    component = BatchHPGMolGraph([featurizer(polymer)])
    model = HPGPublishedGATNet()
    model.eval()
    ratio = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    temperature = HPGPublishedGATNet.normalize_temperature(torch.tensor([[25.0]]))
    with torch.no_grad():
        prediction = model([component] * 6, ratio, temperature)
    assert prediction.shape == (1, 1)

    state = model.state_dict()
    assert "GAT_list_1.0.W_node.weight" in state
    assert "linear_g1.weight" in state
    assert "linear_ratio.weight" in state
    assert "linear_temp.weight" in state
    assert "linear1.weight" in state
    assert "linear2.weight" in state


def test_released_checkpoint_matches_reference_prediction():
    checkpoint = Path(__file__).parents[1] / "HPG" / "ckpt" / "HPG_GAT.bin"
    if not checkpoint.exists():
        pytest.skip("Published HPG checkpoint is not present")

    featurizer = HPGPublishedMolGraphFeaturizer()
    j6 = featurizer(HPGPublishedPolymerGraph(
        ("[R]CCO[Q]",),
        (HPGPublishedConnection(0, 0, 13636.0),),
    ))
    padding = featurizer(HPGPublishedPolymerGraph(("C",), ()))
    components = [BatchHPGMolGraph([j6])] + [BatchHPGMolGraph([padding]) for _ in range(5)]

    model = HPGPublishedGATNet()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    ratio = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    temperature = model.normalize_temperature(torch.tensor([[25.0]]))
    with torch.no_grad():
        prediction = model(components, ratio, temperature)

    reference_prediction = torch.tensor([[-1.7074987888336182]])
    torch.testing.assert_close(prediction, reference_prediction, rtol=0, atol=1e-7)
