from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from chemprop.data.collate import BatchPolymerMolGraph
from chemprop.data.datapoints import PolymerDatapoint
from chemprop.featurizers.molgraph.molecule import wDMPNNPublishedPolymerMolGraphFeaturizer
from evaluation.ea_ip_benchmark import (
    EA_IP_MODELS,
    EA_IP_TARGETS,
    EAIPRepresentationModel,
    add_xn8,
    certify_published_parity,
    chemistry_pair_id,
    chemistry_pair_ids,
    featurize_ea_ip,
    fit_one_run,
    load_split_artifact,
    make_collate,
    mutation_safe_published_graph,
    remove_xn8,
    validate_ea_ip_frame,
)

ROOT = Path(__file__).parents[1]
DATA = ROOT / "data" / "ea_ip.csv"
PUBLISHED = ROOT / "polymer-chemprop-data" / "datasets" / "vipea" / "chemprop_inputs" / "dataset-poly_chemprop.csv"
SPLITS = ROOT / "metadata" / "splits" / "ea_ip_chemistry_disjoint.json"


@pytest.fixture(scope="module")
def frame():
    return pd.read_csv(DATA)


@pytest.fixture(scope="module")
def published():
    return pd.read_csv(PUBLISHED)


def test_actual_dataset_and_published_input_parity(frame, published):
    validate_ea_ip_frame(frame)
    result = certify_published_parity(frame, published)
    assert result["rows_compared"] == 42966
    assert result["input_matches"] == 42966
    assert result["input_mismatches"] == 0
    assert result["row_order_aligned"]
    assert all(result["targets_within_tolerance"].values())
    assert max(result["target_max_abs_difference"].values()) < 5.1e-10


def test_xn8_is_the_only_input_change_for_every_row(frame):
    assert not frame.WDMPNN_Input.str.contains("~", regex=False).any()
    transformed = frame.WDMPNN_Input.map(add_xn8)
    assert transformed.str.endswith("~8").all()
    assert transformed.str.count("~").eq(1).all()
    recovered = transformed.map(remove_xn8)
    assert recovered.equals(frame.WDMPNN_Input)
    for original, xn8 in zip(frame.WDMPNN_Input.iloc[:100], transformed.iloc[:100]):
        assert original.split("|")[:3] == xn8.split("|")[:3]
        assert remove_xn8(xn8).split("<") == original.split("<")


def test_mutation_safe_xn8_repeatability(frame):
    source = add_xn8(frame.WDMPNN_Input.iloc[1])
    first = mutation_safe_published_graph(source)
    second = mutation_safe_published_graph(source)
    assert source.endswith("~8")
    assert first.degree_of_polym == pytest.approx(1 + np.log10(8))
    assert second.degree_of_polym == pytest.approx(first.degree_of_polym)
    for field in first._fields:
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field))


@pytest.mark.parametrize("poly_type", ["block", "random"])
def test_mutation_safe_path_preserves_published_permissive_semantics(frame, poly_type):
    source = frame.loc[frame.poly_type.eq(poly_type), "WDMPNN_Input"].iloc[0]
    datapoint = PolymerDatapoint.from_smi(source)
    historical = wDMPNNPublishedPolymerMolGraphFeaturizer()(datapoint.mol, list(datapoint.edges))
    safe = mutation_safe_published_graph(source)
    for field in historical._fields:
        np.testing.assert_array_equal(getattr(historical, field), getattr(safe, field))
    assert source == frame.loc[frame.poly_type.eq(poly_type), "WDMPNN_Input"].iloc[0]


def test_xn_degree_and_embedding_scaling(frame):
    base = mutation_safe_published_graph(frame.WDMPNN_Input.iloc[0])
    xn8 = mutation_safe_published_graph(add_xn8(frame.WDMPNN_Input.iloc[0]))
    assert base.degree_of_polym == pytest.approx(1.0)
    assert xn8.degree_of_polym == pytest.approx(1 + np.log10(8))
    np.testing.assert_array_equal(base.V, xn8.V)
    np.testing.assert_array_equal(base.E, xn8.E)
    np.testing.assert_array_equal(base.edge_weights, xn8.edge_weights)
    model = EAIPRepresentationModel("wd-published", base)
    model.eval()
    with torch.no_grad():
        _, base_embedding = model(BatchPolymerMolGraph([base]))
        _, xn8_embedding = model(BatchPolymerMolGraph([xn8]))
    torch.testing.assert_close(xn8_embedding, base_embedding * (1 + np.log10(8)))


def test_generic_mix_equals_fraction_weighted_shared_embeddings(frame):
    subset = frame.iloc[:2].reset_index(drop=True)
    inputs = featurize_ea_ip(subset, "dmpnn")
    items = [((inputs[0][i], inputs[1][i], inputs[2][i]), np.float32(0), i) for i in range(2)]
    batch, _, _ = make_collate("dmpnn")(items)
    model = EAIPRepresentationModel("dmpnn", inputs[0][0])
    model.eval()
    with torch.no_grad():
        _, embedding = model(batch)
        embedding_a = model.encoder(batch[0])
        embedding_b = model.encoder(batch[1])
        expected = batch[2][:, :1] * embedding_a + batch[2][:, 1:] * embedding_b
    torch.testing.assert_close(embedding, expected)


def test_generic_representation_is_architecture_invariant(frame):
    grouped = frame.groupby(["smiles_A", "smiles_B", "fracA"], sort=False)
    indices = next(
        group.index[:2].tolist()
        for _, group in grouped
        if group.poly_type.nunique() > 1
    )
    subset = frame.loc[indices].reset_index(drop=True)
    assert subset.poly_type.nunique() > 1
    inputs = featurize_ea_ip(subset, "gin")
    for left, right in ((inputs[0][0], inputs[0][1]), (inputs[1][0], inputs[1][1])):
        for field in left._fields:
            np.testing.assert_array_equal(getattr(left, field), getattr(right, field))
    np.testing.assert_array_equal(inputs[2][0], inputs[2][1])
    items = [((inputs[0][i], inputs[1][i], inputs[2][i]), np.float32(0), i) for i in range(2)]
    batch, _, _ = make_collate("gin")(items)
    model = EAIPRepresentationModel("gin", inputs[0][0])
    model.eval()
    with torch.no_grad():
        _, embedding = model(batch)
    torch.testing.assert_close(embedding[0], embedding[1])


def test_unordered_canonical_chemistry_grouping(frame):
    row = frame.iloc[0]
    forward = chemistry_pair_id(row.smiles_A, row.smiles_B)
    reverse = chemistry_pair_id(row.smiles_B, row.smiles_A)
    assert forward == reverse
    groups = chemistry_pair_ids(frame)
    assert len(set(groups)) == 6138
    counts = pd.Series(groups).value_counts()
    assert counts.eq(7).all()


def test_persisted_splits_are_pair_disjoint_and_common(frame):
    artifact = load_split_artifact(frame, SPLITS)
    assert artifact["n_chemistry_groups"] == 6138
    assert [split["split_seed"] for split in artifact["splits"]] == [42, 43, 44, 45, 46]
    groups = chemistry_pair_ids(frame)
    for split in artifact["splits"]:
        assert (len(split["train"]), len(split["validation"]), len(split["test"])) == (34370, 4298, 4298)
        group_sets = {
            partition: set(groups[split[partition]])
            for partition in ("train", "validation", "test")
        }
        assert not group_sets["train"] & group_sets["validation"]
        assert not group_sets["train"] & group_sets["test"]
        assert not group_sets["validation"] & group_sets["test"]
        expected = {partition: tuple(split[partition]) for partition in ("train", "validation", "test")}
        for _model in EA_IP_MODELS:
            assert {partition: tuple(split[partition]) for partition in expected} == expected


def test_only_physical_targets_are_exposed():
    assert EA_IP_TARGETS == ("EA vs SHE (eV)", "IP vs SHE (eV)")
    assert all("x5" not in target for target in EA_IP_TARGETS)


def test_fresh_fit_reloads_best_checkpoint(frame, tmp_path, monkeypatch):
    import evaluation.ea_ip_benchmark as benchmark

    subset = frame.iloc[:20].reset_index(drop=True)
    inputs = featurize_ea_ip(subset, "dmpnn")
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
        inputs=inputs,
        model_name="dmpnn",
        target=EA_IP_TARGETS[0],
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
    assert provenance["checkpoint_selected"] == str(calls[0])
    assert provenance["epoch_selected"] == 0
    stored = json.loads((calls[0].parent / "provenance.json").read_text())
    assert stored["split_seed"] == 42
    assert stored["init_seed"] == 101
