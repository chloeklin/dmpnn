"""
Generate canonical split metadata files from existing prediction files.

Writes JSON files to metadata/splits/:
    monomer_heldout.json
    group_disjoint.json
    pair_disjoint.json

Each file is a list of fold objects containing:
    fold, split, n_train, n_val, n_test,
    test_indices (global df row positions),
    held_out_entity (monomer A / group key / pair key),
    validation checks

Usage:
    python scripts/generate_split_metadata.py
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_PATH  = ROOT / "data" / "ea_ip.csv"
PRED_ROOT  = ROOT / "predictions"
META_DIR   = ROOT / "metadata" / "splits"
META_DIR.mkdir(parents=True, exist_ok=True)

from evaluation.naming import CANONICAL_MODELS, CANONICAL_TARGETS, make_prediction_filename


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_df() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df


# ── Monomer-heldout metadata ──────────────────────────────────────────────────

def build_monomer_heldout_metadata(df: pd.DataFrame) -> list[dict]:
    """
    Reconstruct the LOMAO split indices using generate_a_held_out_splits,
    which is the same function used during training.  This gives us the true
    global df row indices for each fold's test set, training set, and the
    identity of the held-out monomer A.
    """
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "scripts" / "python"))
    from utils import generate_a_held_out_splits

    # generate_a_held_out_splits uses smiles_A array (original, not canonicalized)
    train_splits, val_splits, test_splits, held_out_monomers = generate_a_held_out_splits(
        smiles_A=df["smiles_A"].values,
        n_datapoints=len(df),
        seed=42,
        n_splits=9,
        protocol="leave_one_A_out",
    )

    records = []
    LOMO_DIR = PRED_ROOT / "ea_ip_lomo"

    for fold in range(len(test_splits)):
        te  = np.asarray(test_splits[fold], dtype=int)
        tr  = np.asarray(train_splits[fold], dtype=int)
        held_out_A = held_out_monomers[fold] if held_out_monomers else None

        test_rows = df.iloc[te]

        # Verify: held-out monomer A present in all test rows
        test_A_set = set(test_rows["smiles_A"].unique())
        if held_out_A:
            hA_in_test = held_out_A in test_A_set
        else:
            hA_in_test = None

        # Verify: held-out monomer A absent from training set
        train_A_set = set(df.iloc[tr]["smiles_A"].unique())
        if held_out_A:
            hA_in_train = held_out_A in train_A_set
        else:
            hA_in_train = None

        leakage_check_passed = (hA_in_test is not False) and (hA_in_train is not True)

        # Cross-reference with the npz file (for n_test consistency)
        fname = make_prediction_filename("EA vs SHE (eV)", "wdmpnn", "monomer_heldout", fold)
        fpath = LOMO_DIR / fname
        source_info = {}
        if fpath.exists():
            p = np.load(fpath, allow_pickle=True)
            n_test_file = len(p["y_true"].flatten())
            source_info = {
                "source_file": fname,
                "source_sha256": _sha256_file(fpath),
                "n_test_file_matches_split": int(n_test_file) == len(te),
            }

        records.append({
            "split":                   "monomer_heldout",
            "fold":                    fold,
            "n_train":                 len(tr),
            "n_val":                   len(val_splits[fold]) if val_splits else None,
            "n_test":                  len(te),
            "global_test_indices":     te.tolist(),
            "held_out_monomer_A":      held_out_A,
            "held_out_A_in_test":      hA_in_test,
            "held_out_A_in_train":     hA_in_train,
            "leakage_check_passed":    leakage_check_passed,
            "architecture_counts":     test_rows["poly_type"].value_counts().to_dict(),
            **source_info,
        })

    return records


# ── Group/pair disjoint metadata ──────────────────────────────────────────────

def _build_gen_metadata(split: str) -> list[dict]:
    subdir = PRED_ROOT / ("ea_ip_group" if split == "group_disjoint" else "ea_ip_pair")
    records = []

    for fold in range(5):
        fname = make_prediction_filename("EA vs SHE (eV)", "wdmpnn", split, fold)
        fpath = subdir / fname
        if not fpath.exists():
            continue

        p = np.load(fpath, allow_pickle=True)
        te = p["test_indices"]
        n_train = int(p["n_train"])
        n_val   = int(p["n_val"])
        n_test  = int(p["n_test"])

        # Characterise held-out entity
        if "smiles_A" in p and "smiles_B" in p:
            smA = p["smiles_A"]
            smB = p["smiles_B"]
            fracA = p["fracA"] if "fracA" in p else None
            if split == "pair_disjoint":
                pairs = sorted(set(zip(smA.tolist(), smB.tolist())))
                held_out_entity = {"held_out_pairs": pairs[:10]}  # first 10 for brevity
            else:
                groups = sorted(set(zip(
                    smA.tolist(), smB.tolist(),
                    (fracA.tolist() if fracA is not None else [None]*len(smA)),
                )))
                held_out_entity = {"held_out_groups_sample": groups[:5]}
        else:
            held_out_entity = {}

        records.append({
            "split":            split,
            "fold":             fold,
            "n_train":          n_train,
            "n_val":            n_val,
            "n_test":           n_test,
            "global_test_indices": te.tolist(),
            **held_out_entity,
            "source_file": fname,
            "source_sha256": _sha256_file(fpath),
        })

    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading dataset...")
    df = _load_df()
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    print("\nBuilding monomer_heldout metadata...")
    lomo = build_monomer_heldout_metadata(df)
    out = META_DIR / "monomer_heldout.json"
    with open(out, "w") as f:
        json.dump({"split": "monomer_heldout", "folds": lomo}, f, indent=2)
    print(f"  Wrote {len(lomo)} fold(s) → {out.relative_to(ROOT)}")
    leakage_folds = [r["fold"] for r in lomo if not r["leakage_check_passed"]]
    if leakage_folds:
        print(f"  ⚠ Leakage detected in folds: {leakage_folds}")
    else:
        print("  ✓ No leakage detected in any fold")

    print("\nBuilding group_disjoint metadata...")
    grp = _build_gen_metadata("group_disjoint")
    out = META_DIR / "group_disjoint.json"
    with open(out, "w") as f:
        json.dump({"split": "group_disjoint", "folds": grp}, f, indent=2)
    print(f"  Wrote {len(grp)} fold(s) → {out.relative_to(ROOT)}")

    print("\nBuilding pair_disjoint metadata...")
    pair = _build_gen_metadata("pair_disjoint")
    out = META_DIR / "pair_disjoint.json"
    with open(out, "w") as f:
        json.dump({"split": "pair_disjoint", "folds": pair}, f, indent=2)
    print(f"  Wrote {len(pair)} fold(s) → {out.relative_to(ROOT)}")

    print("\nDone. Metadata written to:", META_DIR.relative_to(ROOT))


if __name__ == "__main__":
    main()
