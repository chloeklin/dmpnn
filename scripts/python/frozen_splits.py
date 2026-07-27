from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def _clustered_folds(values: list[str], split_seed: int) -> list[list[str]]:
    clusters: dict[str, list[str]] = {}
    for smi in sorted(values):
        mol = Chem.MolFromSmiles(smi)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else "INVALID"
        clusters.setdefault(scaffold or f"ACYCLIC::{smi}", []).append(smi)
    capacities = [76] * (len(values) % 9) + [75] * (9 - len(values) % 9)
    bins = [[] for _ in capacities]
    remaining = capacities.copy()
    rng = np.random.default_rng(split_seed)
    items = list(clusters.values())
    rng.shuffle(items)
    items.sort(key=len, reverse=True)
    for members in items:
        pending = list(members)
        while pending:
            candidates = [i for i, rem in enumerate(remaining) if rem > 0]
            index = max(candidates, key=lambda i: (remaining[i], -len(bins[i])))
            take = min(len(pending), remaining[index])
            bins[index].extend(pending[:take])
            pending = pending[take:]
            remaining[index] -= take
    return bins


def load_frozen_b_heldout_splits(
    df: pd.DataFrame,
    split_seed: int,
    metadata_path: Path,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    if split_seed != 42:
        raise AssertionError(f"monomer_b_heldout requires fixed split_seed=42, got {split_seed}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("split_seed") != split_seed:
        raise AssertionError("monomer_b_heldout metadata split seed does not match the required fixed seed")
    monomers = sorted(df["smiles_B"].astype(str).unique())
    if metadata.get("assignment") == "Murcko scaffold balanced":
        heldout_folds = _clustered_folds(monomers, split_seed)
    else:
        shuffled = np.asarray(monomers, dtype=object)
        np.random.default_rng(split_seed).shuffle(shuffled)
        heldout_folds = [chunk.tolist() for chunk in np.array_split(shuffled, 9)]
    values = df["smiles_B"].astype(str)
    trains, vals, tests = [], [], []
    for fold, heldout in enumerate(heldout_folds):
        validation = heldout_folds[(fold + 1) % len(heldout_folds)]
        test = np.flatnonzero(values.isin(heldout))
        val = np.flatnonzero(values.isin(validation))
        train = np.flatnonzero(~values.isin(set(heldout) | set(validation)))
        frozen = metadata["folds"][fold]
        if (not np.array_equal(train, np.asarray(frozen["global_train_indices"], dtype=int)) or
                not np.array_equal(val, np.asarray(frozen["global_val_indices"], dtype=int)) or
                not np.array_equal(test, np.asarray(frozen["global_test_indices"], dtype=int)) or
                sorted(heldout) != sorted(frozen["held_out_monomer_B"])):
            raise AssertionError(f"monomer_b_heldout fold {fold} differs from frozen metadata")
        if set(values.iloc[train]) & (set(heldout) | set(validation)):
            raise AssertionError(f"monomer_b_heldout fold {fold} has B-identity leakage")
        trains.append(train)
        vals.append(val)
        tests.append(test)
    if set().union(*(set(indices) for indices in tests)) != set(range(len(df))):
        raise AssertionError("monomer_b_heldout test folds do not cover the dataset")
    return trains, vals, tests
