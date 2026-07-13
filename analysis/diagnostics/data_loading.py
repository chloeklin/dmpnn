"""Data and prediction loading utilities."""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path

from .config import (
    DATA_PATH, META_DIR, PRED_ROOT, MODELS, SPLITS, TARGETS,
    TARGET_TOKENS, SPLIT_SUBDIRS, N_FOLDS,
)


def load_dataset() -> pd.DataFrame:
    """Load ea_ip.csv with pair canonicalization (adds group_key, smilesA, smilesB)."""
    df = pd.read_csv(DATA_PATH)

    def _canon(a, b, wa, wb):
        a = "" if pd.isna(a) else str(a)
        b = "" if pd.isna(b) else str(b)
        if b < a:
            return b, a, wb, wa
        return a, b, wa, wb

    raw_A  = df['smiles_A'].astype(str).tolist()
    raw_B  = df['smiles_B'].astype(str).tolist()
    raw_fA = df['fracA'].values.astype(float)
    raw_fB = df['fracB'].values.astype(float)

    canA, canB, fA_list, fB_list = [], [], [], []
    for a, b, wa, wb in zip(raw_A, raw_B, raw_fA, raw_fB):
        a2, b2, wa2, wb2 = _canon(a, b, wa, wb)
        canA.append(a2); canB.append(b2)
        fA_list.append(wa2); fB_list.append(wb2)

    fA = np.array(fA_list, dtype=float)
    fB = np.array(fB_list, dtype=float)
    s  = fA + fB
    fA = fA / s
    fB = 1.0 - fA

    df['smilesA'] = canA;  df['smilesB'] = canB
    df['fracA']   = fA;    df['fracB']   = fB

    _r6 = lambda x: round(float(x), 6)
    df['group_key'] = [
        f"{a}||{b}||{_r6(fa)}||{_r6(fb)}"
        for a, b, fa, fb in zip(canA, canB, fA, fB)
    ]
    return df


def load_split_meta(split: str) -> list:
    """Load split metadata JSON, returning the list of fold records."""
    path = META_DIR / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return json.loads(path.read_text())["folds"]


def get_global_indices(fold: int, meta_folds: list) -> np.ndarray:
    """Retrieve global test indices for a given fold from metadata."""
    for rec in meta_folds:
        if rec["fold"] == fold:
            return np.asarray(rec["global_test_indices"], dtype=int)
    raise KeyError(f"Fold {fold} not in metadata")


def make_pred_filename(target_key: str, model: str, split: str, fold: int) -> str:
    """Construct canonical prediction filename."""
    t_tok = TARGET_TOKENS[target_key]
    return f"ea_ip__{t_tok}__{model}__{split}__fold{fold}.npz"


def load_predictions_single(model: str, target_key: str, split: str, fold: int,
                            meta_folds: list) -> dict | None:
    """Load a single fold's predictions. Returns dict or None if missing."""
    subdir = PRED_ROOT / SPLIT_SUBDIRS[split]
    fname = make_pred_filename(target_key, model, split, fold)
    fpath = subdir / fname
    if not fpath.exists():
        return None
    p = np.load(fpath, allow_pickle=True)
    y_true = p['y_true'].flatten().astype(np.float64)
    y_pred = p['y_pred'].flatten().astype(np.float64)
    global_idx = get_global_indices(fold, meta_folds)
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'global_idx': global_idx,
        'fold': fold,
        'model': model,
        'target': target_key,
        'split': split,
    }


def load_all_predictions(meta: dict[str, list]) -> list[dict]:
    """
    Load all predictions for all model/target/split/fold combos.

    Parameters
    ----------
    meta : dict mapping split name -> list of fold metadata records

    Returns
    -------
    List of prediction dicts (only successfully loaded ones).
    """
    results = []
    for split in SPLITS:
        meta_folds = meta[split]
        for model in MODELS:
            for tkey in TARGETS:
                for fold in range(N_FOLDS[split]):
                    pred = load_predictions_single(
                        model, tkey, split, fold, meta_folds
                    )
                    if pred is not None:
                        results.append(pred)
    return results


def load_all_meta() -> dict[str, list]:
    """Load metadata for all splits."""
    return {split: load_split_meta(split) for split in SPLITS}
