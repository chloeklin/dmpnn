"""Tests for a_held_out split implementation."""
import sys, os, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts', 'python'))
import numpy as np
from pathlib import Path
from utils import generate_a_held_out_splits, save_fold_assignments, canonicalize_smiles


def test_splits_no_overlap():
    """All smiles_A groups must be exclusive across train/val/test."""
    unique_A = [f'C{i}CC' for i in range(10)]
    smiles_A = np.array([unique_A[i % 10] for i in range(100)], dtype=str)

    train_idx, val_idx, test_idx = generate_a_held_out_splits(
        smiles_A, n_datapoints=100, seed=42, n_splits=5
    )

    assert len(train_idx) == 5
    for i in range(5):
        tr, va, te = train_idx[i], val_idx[i], test_idx[i]
        tr_groups = set(smiles_A[tr])
        va_groups = set(smiles_A[va])
        te_groups = set(smiles_A[te])

        assert not (tr_groups & va_groups), f'Fold {i}: train/val overlap!'
        assert not (tr_groups & te_groups), f'Fold {i}: train/test overlap!'
        assert not (va_groups & te_groups), f'Fold {i}: val/test overlap!'

        # Full coverage
        assert len(set(tr) | set(va) | set(te)) == 100, f'Fold {i}: not all samples covered!'
        print(f'  Fold {i}: train={len(tr)} ({len(tr_groups)} groups), '
              f'val={len(va)} ({len(va_groups)} groups), '
              f'test={len(te)} ({len(te_groups)} groups) -- OK')


def test_determinism():
    """Same seed produces same splits."""
    unique_A = [f'C{i}CC' for i in range(10)]
    smiles_A = np.array([unique_A[i % 10] for i in range(100)], dtype=str)

    r1 = generate_a_held_out_splits(smiles_A, 100, seed=42, n_splits=5)
    r2 = generate_a_held_out_splits(smiles_A, 100, seed=42, n_splits=5)
    for i in range(5):
        assert np.array_equal(r1[0][i], r2[0][i]), f'Fold {i} train not deterministic'
        assert np.array_equal(r1[1][i], r2[1][i]), f'Fold {i} val not deterministic'
        assert np.array_equal(r1[2][i], r2[2][i]), f'Fold {i} test not deterministic'
    print('Determinism: PASSED')


def test_different_seed():
    """Different seed produces different splits."""
    unique_A = [f'C{i}CC' for i in range(20)]
    smiles_A = np.array([unique_A[i % 20] for i in range(200)], dtype=str)

    r1 = generate_a_held_out_splits(smiles_A, 200, seed=42, n_splits=5)
    r2 = generate_a_held_out_splits(smiles_A, 200, seed=99, n_splits=5)
    # GroupKFold is deterministic given groups, but val sub-split should differ
    any_diff = False
    for i in range(5):
        if not np.array_equal(r1[1][i], r2[1][i]):
            any_diff = True
            break
    assert any_diff, 'Different seeds should produce different val splits'
    print('Different seed: PASSED')


def test_save_fold_assignments():
    """Fold assignments are saved correctly to JSON."""
    unique_A = [f'C{i}CC' for i in range(10)]
    smiles_A = np.array([unique_A[i % 10] for i in range(100)], dtype=str)

    train_idx, val_idx, test_idx = generate_a_held_out_splits(
        smiles_A, 100, seed=42, n_splits=5
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = save_fold_assignments(
            train_idx, val_idx, test_idx,
            smiles_A, 'test_dataset', 42, Path(tmpdir)
        )
        assert out_path.exists()
        with open(out_path) as f:
            payload = json.load(f)

        assert payload['dataset'] == 'test_dataset'
        assert payload['split_type'] == 'a_held_out'
        assert payload['seed'] == 42
        assert payload['n_folds'] == 5
        assert payload['n_samples'] == 100
        assert payload['n_unique_smiles_A'] == 10

        for fold in payload['folds']:
            assert 'train_indices' in fold
            assert 'val_indices' in fold
            assert 'test_indices' in fold
            assert 'train_smiles_A' in fold
            assert 'val_smiles_A' in fold
            assert 'test_smiles_A' in fold
    print('Save fold assignments: PASSED')


def test_canonicalization():
    """Equivalent SMILES should canonicalize to the same string."""
    c1 = canonicalize_smiles('C(C)O')
    c2 = canonicalize_smiles('OCC')
    assert c1 == c2, f'{c1} != {c2}'
    print(f'Canonicalization: C(C)O -> {c1}, OCC -> {c2}, match={c1==c2} -- PASSED')


if __name__ == '__main__':
    print('=== test_splits_no_overlap ===')
    test_splits_no_overlap()
    print()
    print('=== test_determinism ===')
    test_determinism()
    print()
    print('=== test_different_seed ===')
    test_different_seed()
    print()
    print('=== test_save_fold_assignments ===')
    test_save_fold_assignments()
    print()
    print('=== test_canonicalization ===')
    test_canonicalization()
    print()
    print('All tests passed!')
