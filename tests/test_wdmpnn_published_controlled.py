"""Regression tests for wDMPNN-published and wDMPNN-controlled.

These tests establish that:

* ``wDMPNNPublishedPolymerMolGraphFeaturizer`` reproduces the reference
  polymer-chemprop v1 133D atom features and polymer graph tensors for
  representative inputs.
* ``wDMPNNControlledPolymerMolGraphFeaturizer`` fixes the known parser
  mutation and normalization-guard defects while preserving the core wD
  representation.
* The two implementations intentionally differ exactly where documented.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from chemprop.featurizers.atom import MultiHotAtomFeaturizer
from chemprop.featurizers.molgraph.molecule import (
    wDMPNNControlledPolymerMolGraphFeaturizer,
    wDMPNNPublishedPolymerMolGraphFeaturizer,
)
from chemprop.utils.utils import make_polymer_mol


# ---------------------------------------------------------------------------
# Reference polymer-chemprop loader (isolated from the top-level package,
# which depends on ``tap`` which is not installed here).
# ---------------------------------------------------------------------------
def _load_reference_featurization():
    """Load the original polymer-chemprop featurization module by path."""
    ref_dir = Path.home() / "Desktop" / "polymer-chemprop"
    featurization_path = ref_dir / "chemprop" / "features" / "featurization.py"
    rdkit_path = ref_dir / "chemprop" / "rdkit.py"

    if not featurization_path.exists():
        pytest.skip(f"Reference polymer-chemprop not found at {ref_dir}")

    # Ensure a chemprop namespace exists for submodule imports.
    if "chemprop" not in sys.modules:
        sys.modules["chemprop"] = types.ModuleType("chemprop")

    spec = importlib.util.spec_from_file_location(
        "chemprop.rdkit_ref", str(rdkit_path)
    )
    rdkit_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rdkit_mod)
    sys.modules["chemprop.rdkit"] = rdkit_mod

    spec = importlib.util.spec_from_file_location(
        "chemprop.features.featurization_ref", str(featurization_path)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ref = _load_reference_featurization()


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
def _reference_mol_graph(smi: str):
    """Build a reference MolGraph in polymer mode."""
    _ref.set_polymer(True)
    return _ref.MolGraph(smi)


def _ours(feat, smi: str, weights: list[float], rules: list[str]):
    """Build our PolymerMolGraph using a featurizer and the same inputs."""
    mol = make_polymer_mol(smi, weights)
    return feat(mol, rules)


@pytest.fixture
def published():
    return wDMPNNPublishedPolymerMolGraphFeaturizer()


@pytest.fixture
def controlled():
    return wDMPNNControlledPolymerMolGraphFeaturizer(
        atom_featurizer=MultiHotAtomFeaturizer.v1()
    )


# ---------------------------------------------------------------------------
# 1. Published atom features == reference v1 133D features
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "smiles",
    [
        "CCO",  # ethanol
        "c1ccccc1",  # benzene
        "C1CCOC1",  # THF
        "CC(=O)O",  # acetic acid
        "C1CCCCC1N",  # cyclohexylamine
        "Fc1ccccc1",  # fluorobenzene
        "O=C1OC(=O)C2=CC=CC=C12",  # phthalic anhydride
    ],
)
def test_published_atom_features_match_reference(smiles):
    """Our v1 atom features must equal polymer-chemprop's atom_features()."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None

    our_featurizer = MultiHotAtomFeaturizer.v1()
    ref_features = np.array([_ref.atom_features(a) for a in mol.GetAtoms()])
    our_features = np.array([our_featurizer(a) for a in mol.GetAtoms()])

    assert our_features.shape == ref_features.shape
    assert our_features.shape[1] == 133
    assert np.allclose(our_features, ref_features, atol=1e-9)


# ---------------------------------------------------------------------------
# 2. Published polymer graph construction == reference
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "smiles,weights,rules",
    [
        # Homopolymer with Xn=10
        (
            "[*:1]OCC[*:2]",
            [1.0],
            ["1-2:1:1~10"],
        ),
        # Block copolymer, equal fractions, Xn=100
        (
            "CC([*:1])C.[*:2]OCCO[*:3]",  # two distinct fragments each with two wildcards
            [0.5, 0.5],
            ["1-2:0.5:0.5~100"],
        ),
        # Alternating copolymer with two directed rules
        (
            "[*:1]CC[*:2].[*:3]OCC[*:4]",
            [0.5, 0.5],
            ["2-3:0.5:0.5", "4-1:0.5:0.5~50"],
        ),
    ],
)
def test_published_polymer_graph_matches_reference(published, smiles, weights, rules):
    """Full graph tensors from wDMPNN-published must match polymer-chemprop."""
    # Build reference input string:  SMILES|w1|w2|<rules
    edge_str = "".join(f"<{r}" for r in rules)
    ref_smi = f"{smiles}|" + "|".join(str(w) for w in weights) + "|" + edge_str
    ref_mg = _reference_mol_graph(ref_smi)

    # Build ours using the same SMILES/weights/rules.
    our_mg = _ours(published, smiles, weights, rules)

    # Atom features
    assert our_mg.V.shape == (ref_mg.n_atoms, 133)
    assert np.allclose(our_mg.V, ref_mg.f_atoms, atol=1e-6)

    # Atom weights
    assert np.allclose(our_mg.atom_weights, ref_mg.w_atoms, atol=1e-7)

    # Bond features: reference concatenates atom+bond, ours stores bond-only.
    # Reference ``n_bonds`` already counts directed edges, matching our E rows.
    ref_bond_feats = np.array(ref_mg.f_bonds)[:, -14:]
    assert our_mg.E.shape == (ref_mg.n_bonds, 14)
    assert np.allclose(our_mg.E, ref_bond_feats, atol=1e-7)

    # Edge weights
    assert our_mg.edge_weights.shape == (ref_mg.n_bonds,)
    assert np.allclose(our_mg.edge_weights, ref_mg.w_bonds, atol=1e-7)

    # Edge indices (source atom list)
    assert list(our_mg.edge_index[0]) == ref_mg.b2a

    # Reverse-edge indices
    assert list(our_mg.rev_edge_index) == ref_mg.b2revb

    # Degree of polymerization
    assert our_mg.degree_of_polym == pytest.approx(ref_mg.degree_of_polym)


# ---------------------------------------------------------------------------
# 3. Controlled parser is non-mutating and preserves Xn
# ---------------------------------------------------------------------------
def test_controlled_xn_repeatability(controlled):
    """Calling the controlled featurizer twice must give identical Xn."""
    mol = make_polymer_mol("[*:1]OCC[*:2]", [1.0])
    edges = ["1-2:1:1~100"]

    mg1 = controlled(mol, list(edges))
    mg2 = controlled(mol, list(edges))

    assert mg1.degree_of_polym == pytest.approx(3.0)
    assert mg2.degree_of_polym == pytest.approx(3.0)
    assert edges == ["1-2:1:1~100"]  # input list is untouched


def test_published_xn_mutation_is_reproduced(published):
    """The published parser mutates the input list; second call falls back to Xn=1."""
    mol = make_polymer_mol("[*:1]OCC[*:2]", [1.0])
    edges = ["1-2:1:1~100"]

    # Use the same list object for both calls to expose the in-place mutation.
    mg1 = published(mol, edges)
    # The published parser strips the ``~Xn`` suffix in-place.
    assert edges == ["1-2:1:1"]  # list was mutated

    mg2 = published(mol, edges)
    assert mg1.degree_of_polym == pytest.approx(3.0)
    assert mg2.degree_of_polym == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 4. Controlled normalization guard is effective
# ---------------------------------------------------------------------------
def test_controlled_normalization_guard_raises(controlled):
    """Controlled strict guard rejects weights that are not close to 1."""
    mol = make_polymer_mol("[*:1]CC[*:2].[*:3]OCC[*:4]", [0.5, 0.5])
    # Both attachment points receive an incoming weight of 0.5, not 1.0.
    rules = ["2-3:0.5:0.5", "4-1:0.5:0.5"]

    with pytest.raises(ValueError, match="sum of weights of incoming stochastic edges"):
        controlled(mol, rules)


def test_published_normalization_guard_is_dead(published):
    """Published guard does not fire for the same non-normalized weights."""
    mol = make_polymer_mol("[*:1]CC[*:2].[*:3]OCC[*:4]", [0.5, 0.5])
    rules = ["2-3:0.5:0.5", "4-1:0.5:0.5"]

    # Must not raise.
    mg = published(mol, rules)
    assert mg is not None


# ---------------------------------------------------------------------------
# 5. Controlled atom feature choice is explicit
# ---------------------------------------------------------------------------
def test_controlled_requires_explicit_atom_featurizer():
    """Controlled refuses an accidental default; caller must choose atom features."""
    with pytest.raises(ValueError, match="explicit atom_featurizer choice"):
        wDMPNNControlledPolymerMolGraphFeaturizer()


# ---------------------------------------------------------------------------
# 6. Shape/dimension sanity
# ---------------------------------------------------------------------------
def test_published_reports_133d_atom_features(published):
    assert published.atom_fdim == 133
    assert published.bond_fdim == 14


def test_controlled_reports_chosen_atom_features(controlled):
    assert controlled.atom_fdim == 133
    assert controlled.bond_fdim == 14
