from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from rdkit import Chem

from chemprop.data.collate import BatchMolGraph, BatchPolymerMolGraph
from chemprop.data.hpg import BatchHPGMolGraph
from evaluation.htpmd_benchmark import (
    PROVENANCE_FIELDS,
    HTPMDRepresentationModel,
    SharedPredictor,
    TargetStandardizer,
    TrainStandardizer,
    canonical_chemistry,
    external_columns,
    featurize_htpmd,
    fit_one_run,
    generate_chemistry_disjoint_splits,
    htpmd_generic_molecule,
    htpmd_hpg_spec,
    htpmd_wd_input,
    load_split_artifact,
    make_collate,
    reload_best_checkpoint,
    save_best_checkpoint,
    set_initialization_seed,
    validate_htpmd_frame,
    validate_split_artifact,
)


ROOT = Path(__file__).parents[1]
DATA = ROOT / "data" / "htpmd.csv"
SPLITS = ROOT / "metadata" / "splits" / "htpmd_chemistry_disjoint.json"


@pytest.fixture(scope="module")
def frame():
    return pd.read_csv(DATA)


def test_persisted_splits_are_chemistry_disjoint(frame):
    artifact = load_split_artifact(frame, SPLITS)
    assert len(artifact["splits"]) == 5
    assert [split["split_seed"] for split in artifact["splits"]] == [42, 43, 44, 45, 46]
    chemistry = frame.psmiles.map(canonical_chemistry).to_numpy()
    for split in artifact["splits"]:
        assert (len(split["train"]), len(split["validation"]), len(split["test"])) == (4760, 595, 595)
        sets = {name: set(chemistry[split[name]]) for name in ("train", "validation", "test")}
        assert not sets["train"] & sets["validation"]
        assert not sets["train"] & sets["test"]
        assert not sets["validation"] & sets["test"]


def test_split_artifact_is_representation_independent(frame):
    artifact = load_split_artifact(frame, SPLITS)
    assert "representation" not in artifact
    validate_split_artifact(frame, artifact)


def test_corrected_and_historical_wd_rules(frame):
    row = frame.iloc[0]
    corrected = htpmd_wd_input(row.psmiles, row.DoP, "corrected")
    historical = htpmd_wd_input(row.psmiles, row.DoP, "historical")
    assert corrected.endswith("<1-2:1.0:1.0~19")
    assert historical.endswith("<1-2:0.5:0.5~19")


def test_hpg_adapter_uses_dop_as_self_loop_degree(frame):
    row = frame.iloc[0]
    spec = htpmd_hpg_spec(row.psmiles, row.DoP)
    assert len(spec.fragments) == 1
    assert len(spec.connections) == 1
    connection = spec.connections[0]
    assert (connection.source, connection.destination, connection.degree) == (0, 0, 19.0)


def test_conditions_do_not_duplicate_native_dop():
    for condition in ("dop", "state"):
        assert "DoP" in external_columns("dmpnn", condition)
        assert "DoP" in external_columns("gin", condition)
        assert "DoP" in external_columns("gat", condition)
        assert "DoP" not in external_columns("wdmpnn-published", condition)
        assert "DoP" not in external_columns("hpg-published", condition)
    assert external_columns("wdmpnn-published", "state") == ("Molality",)
    assert external_columns("hpg-published", "state") == ("Molality",)


def test_external_normalization_uses_training_rows_only(frame):
    train = [0, 1, 2]
    scaler = TrainStandardizer.fit(frame, train, ("DoP", "Molality"))
    expected = frame.iloc[train][["DoP", "Molality"]].to_numpy(dtype=float)
    np.testing.assert_allclose(scaler.means, expected.mean(axis=0))
    np.testing.assert_allclose(scaler.scales, expected.std(axis=0))
    transformed_train = scaler.transform(frame, train)
    np.testing.assert_allclose(transformed_train.mean(axis=0), 0, atol=1e-6)
    changed = frame.copy()
    changed.loc[3:, "DoP"] = 1e9
    changed.loc[3:, "Molality"] = 1e9
    assert TrainStandardizer.fit(changed, train, ("DoP", "Molality")) == scaler


def test_split_and_initialization_seeds_are_independent(frame):
    first = generate_chemistry_disjoint_splits(frame, seeds=(42,))
    set_initialization_seed(100)
    a = torch.nn.Linear(3, 2).weight.detach().clone()
    set_initialization_seed(101)
    b = torch.nn.Linear(3, 2).weight.detach().clone()
    second = generate_chemistry_disjoint_splits(frame, seeds=(42,))
    assert first == second
    assert not torch.equal(a, b)
    set_initialization_seed(100)
    c = torch.nn.Linear(3, 2).weight.detach().clone()
    torch.testing.assert_close(a, c)


def test_modelling_frame_schema_and_size(frame):
    validate_htpmd_frame(frame)
    assert len(frame) == 5950
    assert {"smiles", "psmiles", "DoP", "Molality"} <= set(frame.columns)


@pytest.mark.parametrize("psmiles", ["[*]CCO[*]", "C([*])(C[*])C"])
def test_generic_molecule_removes_markers_and_preserves_attachment_sites(psmiles):
    molecule, attachment = htpmd_generic_molecule(psmiles)
    assert all(atom.GetAtomicNum() not in {0, 29, 79} for atom in molecule.GetAtoms())
    assert attachment.shape == (molecule.GetNumAtoms(), 1)
    assert attachment[:, 0].sum() == 2
    assert set(np.unique(attachment)) <= {0.0, 1.0}


def test_generic_molecule_marks_shared_attachment_atom():
    molecule, attachment = htpmd_generic_molecule("[*]C([*])C")
    assert all(atom.GetAtomicNum() not in {0, 29, 79} for atom in molecule.GetAtoms())
    assert attachment[:, 0].sum() == 1
    assert attachment[0, 0] == 1
    assert attachment[1, 0] == 0


def test_every_htpmd_generic_source_graph_has_no_placeholder_atoms(frame):
    for psmiles in frame.psmiles:
        molecule, attachment = htpmd_generic_molecule(psmiles)
        assert all(atom.GetAtomicNum() not in {0, 29, 79} for atom in molecule.GetAtoms())
        assert attachment[:, 0].sum() in {1, 2}


def test_generic_representations_receive_identical_graphs(frame):
    subset = frame.iloc[:3].reset_index(drop=True)
    graphs = {name: featurize_htpmd(subset, name) for name in ("dmpnn", "gin", "gat")}
    for index in range(len(subset)):
        reference = graphs["dmpnn"][index]
        for name in ("gin", "gat"):
            np.testing.assert_array_equal(graphs[name][index].V, reference.V)
            np.testing.assert_array_equal(graphs[name][index].E, reference.E)
            np.testing.assert_array_equal(graphs[name][index].edge_index, reference.edge_index)
            np.testing.assert_array_equal(graphs[name][index].rev_edge_index, reference.rev_edge_index)
        assert reference.V.shape[1] == 73
        assert reference.V[:, -1].sum() in {1, 2}


def test_generic_graph_is_invariant_to_nonstructural_columns(frame):
    row = frame.iloc[[0]].copy()
    changed = row.copy()
    changed.loc[:, "DoP"] = row.DoP.iloc[0] + 1000
    changed.loc[:, "Molality"] = row.Molality.iloc[0] + 1000
    for target in ("Conductivity", "TFSI Diffusivity", "Li Diffusivity", "Poly Diffusivity", "Transference Number"):
        changed.loc[:, target] = row[target].iloc[0] + 1000
    original = featurize_htpmd(row, "dmpnn")[0]
    modified = featurize_htpmd(changed, "dmpnn")[0]
    for field in original._fields:
        np.testing.assert_array_equal(getattr(original, field), getattr(modified, field))


@pytest.mark.parametrize(
    "condition,generic_external,polymer_external",
    [
        ("native", (), ()),
        ("dop", ("DoP",), ()),
        ("state", ("DoP", "Molality"), ("Molality",)),
    ],
)
def test_condition_routing(condition, generic_external, polymer_external):
    for representation in ("dmpnn", "gin", "gat"):
        assert external_columns(representation, condition) == generic_external
    for representation in ("wdmpnn-published", "hpg-published"):
        assert external_columns(representation, condition) == polymer_external
    assert not ({"Monomer_Molecular_Weight", "Density"} & set(generic_external + polymer_external))


def test_all_conditions_and_models_share_persisted_row_ids(frame):
    artifact = load_split_artifact(frame, SPLITS)
    for split in artifact["splits"]:
        expected = {partition: tuple(split[partition]) for partition in ("train", "validation", "test")}
        for representation in ("dmpnn", "gin", "gat", "wdmpnn-published", "hpg-published"):
            for condition in ("native", "dop", "state"):
                assert {partition: tuple(split[partition]) for partition in expected} == expected
                assert set(external_columns(representation, condition)) <= {"DoP", "Molality"}


@pytest.mark.parametrize(
    "representation,expected_dim",
    [("dmpnn", 300), ("gin", 300), ("gat", 300), ("wdmpnn-published", 300), ("hpg-published", 64)],
)
def test_native_embedding_dimensions(frame, representation, expected_dim):
    subset = frame.iloc[:2].reset_index(drop=True)
    graphs = featurize_htpmd(subset, representation)
    model = HTPMDRepresentationModel(representation, graphs[0], external_dim=0)
    model.eval()
    batch, external, _, _ = make_collate(representation)([
        (graphs[index], np.empty(0, dtype=np.float32), np.float32(0), index)
        for index in range(2)
    ])
    with torch.no_grad():
        _, embedding = model(batch, external)
    assert embedding.shape == (2, expected_dim)
    assert model.embedding_dimension == expected_dim


def test_wd_embedding_includes_xn_scaling(frame):
    subset = frame.iloc[:2].reset_index(drop=True)
    graphs = featurize_htpmd(subset, "wdmpnn-published")
    model = HTPMDRepresentationModel("wdmpnn-published", graphs[0], 0)
    model.eval()
    batch = BatchPolymerMolGraph(graphs)
    with torch.no_grad():
        nodes = model.encoder.message_passing(batch)
        pooled = model.encoder.aggregation(nodes, batch)
        encoded = model.encoder(batch)
    torch.testing.assert_close(encoded, pooled * batch.degree_of_polym.unsqueeze(1))
    assert not torch.allclose(encoded, pooled)


def test_hpg_embedding_is_native_encoder_output(frame):
    subset = frame.iloc[:2].reset_index(drop=True)
    graphs = featurize_htpmd(subset, "hpg-published")
    model = HTPMDRepresentationModel("hpg-published", graphs[0], 0)
    model.eval()
    batch = BatchHPGMolGraph(graphs)
    with torch.no_grad():
        encoded = model.encoder(batch)
        direct = model.encoder.message_passing(batch)
    assert encoded.shape == (2, 64)
    torch.testing.assert_close(encoded, direct)


def test_shared_predictor_design_differs_only_at_input_width():
    head_300 = SharedPredictor(300)
    head_64 = SharedPredictor(64)
    assert head_300.network[0].in_features == 300
    assert head_64.network[0].in_features == 64
    for head in (head_300, head_64):
        assert head.network[0].out_features == 300
        assert isinstance(head.network[1], torch.nn.ReLU)
        assert head.network[2].p == 0
        assert head.network[3].in_features == 300
        assert head.network[3].out_features == 1


def test_best_checkpoint_reload_restores_selected_predictions(tmp_path):
    set_initialization_seed(7)
    model = torch.nn.Linear(2, 1)
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    selected_predictions = model(inputs).detach().clone()
    checkpoint = tmp_path / "best.pt"
    save_best_checkpoint(checkpoint, model, epoch=3, validation_loss=0.25)
    with torch.no_grad():
        model.weight.add_(10)
        model.bias.add_(10)
    assert not torch.allclose(model(inputs), selected_predictions)
    metadata = reload_best_checkpoint(model, checkpoint, torch.device("cpu"))
    torch.testing.assert_close(model(inputs), selected_predictions)
    assert metadata["epoch"] == 3
    assert metadata["validation_loss"] == 0.25


def test_fresh_fit_reloads_best_checkpoint_and_writes_provenance(frame, tmp_path, monkeypatch):
    import evaluation.htpmd_benchmark as benchmark

    subset = frame.iloc[:20].reset_index(drop=True)
    graphs = featurize_htpmd(subset, "dmpnn")
    split = {
        "split_id": 0,
        "split_seed": 42,
        "train": list(range(16)),
        "validation": [16, 17],
        "test": [18, 19],
        "digests": {"train": "test", "validation": "test", "test": "test"},
    }
    original_reload = benchmark.reload_best_checkpoint
    calls = []

    def tracked_reload(model, path, device):
        calls.append(path)
        return original_reload(model, path, device)

    monkeypatch.setattr(benchmark, "reload_best_checkpoint", tracked_reload)
    provenance = fit_one_run(
        frame=subset,
        graphs=graphs,
        representation="dmpnn",
        condition="dop",
        target="Conductivity",
        split=split,
        split_reference=tmp_path / "splits.json",
        init_seed=101,
        output_dir=tmp_path,
        batch_size=4,
        max_epochs=1,
        patience=1,
        max_batches=1,
    )
    assert len(calls) == 1
    assert Path(provenance["checkpoint_selected"]) == calls[0]
    assert provenance["epoch_selected"] == 0
    assert PROVENANCE_FIELDS <= provenance.keys()
    run_dir = calls[0].parent
    assert (run_dir / "predictions.npz").exists()
    stored = json.loads((run_dir / "provenance.json").read_text())
    assert stored["checkpoint_selected"] == provenance["checkpoint_selected"]
    assert stored["init_seed"] == 101
    assert stored["split_seed"] == 42
    assert stored["trainable_parameter_count"] > 0


def test_provenance_schema_contains_required_fields():
    example = {field: None for field in PROVENANCE_FIELDS}
    assert PROVENANCE_FIELDS <= example.keys()
