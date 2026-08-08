#!/usr/bin/env python3
"""Am I holding a complete set of prediction files locally?

Checks the expected regen_v1, octamer_k1 and wdmpnn_original prediction matrices
against what is actually on this machine, and reports what is missing, what is
malformed, and what was deliberately not run.

    python "29-07-2026 supervisor_update/check_results.py"
    python "29-07-2026 supervisor_update/check_results.py" --verbose   # list every missing cell
    python "29-07-2026 supervisor_update/check_results.py" --deep      # also open each NPZ

Three categories are reported for every (tree, model, split) arm:

* **complete** — planned and present.
* **missing** — planned and expected, but absent or malformed.
* **not planned** — deliberately not run; present files, if any, are reported as
  pilots and do not count toward completeness.

A file is only counted as complete when all of these hold:
  * the .npz exists and is non-trivial in size
  * its .config.json provenance sidecar exists
  * the sidecar records non-zero epochs and plausible wall time
  * (--deep) the archive contains y_pred, y_pred_final, y_true and test_indices
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TARGETS = ["EA_vs_SHE_eV", "IP_vs_SHE_eV"]
FOLDS = list(range(9))
SEEDS = [42, 43, 44]
MIN_BYTES = 10_000

# Declarative inventory of every (tree, model, split) arm.
# status: "planned"    -> counts toward complete / missing
#         "not_planned" -> reported separately; present files are pilots
CELLS_PER_SPLIT = len(TARGETS) * len(FOLDS) * len(SEEDS)

PLAN = [
    # regen_v1
    {"tree": "regen_v1", "model": "hpg_hier", "split": "ea_ip_lomo", "status": "planned"},
    {"tree": "regen_v1", "model": "hpg_hier", "split": "ea_ip_lomo_b_clustered", "status": "planned"},
    {"tree": "regen_v1", "model": "wdmpnn", "split": "ea_ip_lomo", "status": "planned"},
    {"tree": "regen_v1", "model": "wdmpnn", "split": "ea_ip_lomo_b_clustered", "status": "planned"},
    {"tree": "regen_v1", "model": "hpg_hier_octamer", "split": "ea_ip_lomo", "status": "planned"},
    {"tree": "regen_v1", "model": "hpg_hier_octamer", "split": "ea_ip_lomo_b_clustered", "status": "planned"},
    {"tree": "regen_v1", "model": "hpg_hier_junction", "split": "ea_ip_lomo", "status": "planned"},
    {"tree": "regen_v1", "model": "hpg_hier_junction", "split": "ea_ip_lomo_b_clustered", "status": "planned"},
    {"tree": "regen_v1", "model": "hpg_hier_junction1", "split": "ea_ip_lomo", "status": "planned"},
    {
        "tree": "regen_v1",
        "model": "hpg_hier_junction1",
        "split": "ea_ip_lomo_b_clustered",
        "status": "not_planned",
        "reason": "R1-only by design (HANDOFF §3)",
        "scope": "tracked",  # outside the current campaign; kept for transparency
    },
    # octamer_k1
    {
        "tree": "octamer_k1",
        "model": "hpg_hier_octamer",
        "split": "ea_ip_lomo_b_clustered",
        "status": "planned",
    },
    # wdmpnn_original
    {"tree": "wdmpnn_original", "model": "wdmpnn", "split": "ea_ip_lomo", "status": "planned"},
    {
        "tree": "wdmpnn_original",
        "model": "wdmpnn",
        "split": "ea_ip_lomo_b_clustered",
        "status": "not_planned",
        "reason": "deferred for compute budget; one pilot run exists",
        "scope": "active",  # part of the current campaign, but deliberately deferred
    },
]

TREE_CONFIG = {
    "regen_v1": {"root": ROOT / "predictions" / "regen_v1", "suffix": ""},
    "octamer_k1": {"root": ROOT / "predictions" / "octamer_k1", "suffix": "__k1"},
    "wdmpnn_original": {"root": ROOT / "predictions" / "wdmpnn_original", "suffix": "__orig"},
}


def inspect(npz: Path, deep: bool) -> tuple[bool, str]:
    """Return (ok, reason-if-not)."""
    if not npz.is_file():
        return False, "npz missing"
    if npz.stat().st_size < MIN_BYTES:
        return False, f"npz suspiciously small ({npz.stat().st_size} B)"
    side = npz.with_suffix(".config.json")
    if not side.is_file():
        return False, "sidecar missing"
    try:
        meta = json.loads(side.read_text())
    except json.JSONDecodeError:
        return False, "sidecar unreadable"
    if meta.get("epochs_actually_run", 0) <= 0:
        return False, "sidecar reports 0 epochs — training was skipped"
    if meta.get("wall_time_seconds", 0) <= 60:
        return False, f"wall time {meta.get('wall_time_seconds')} s — too short to be real"
    if deep:
        try:
            import numpy as np

            with np.load(npz, allow_pickle=True) as arch:
                for key in ("y_pred", "y_pred_final", "y_true", "test_indices"):
                    if key not in arch.files:
                        return False, f"array {key} absent"
        except Exception as exc:  # noqa: BLE001
            return False, f"npz unreadable: {type(exc).__name__}"
    return True, ""


def _heldout_name(split: str) -> str:
    return "monomer_b_heldout_clustered" if split.endswith("_clustered") else "monomer_heldout"


def _cell_filename(model: str, split: str, suffix: str, target: str, fold: int, seed: int) -> str:
    heldout = _heldout_name(split)
    return f"ea_ip__{target}__{model}__{heldout}__fold{fold}__s{seed}{suffix}.npz"


def _check_tree(tree: str, deep: bool, verbose: bool) -> dict[str, int]:
    """Check all arms declared for a single prediction tree."""
    cfg = TREE_CONFIG[tree]
    root_dir = cfg["root"]
    suffix = cfg["suffix"]

    print(f"\n{root_dir.relative_to(ROOT) if root_dir.is_dir() else root_dir}:")
    if not root_dir.is_dir():
        print("  directory does not exist")
        return {
            "complete": 0,
            "missing": 0,
            "not_planned": 0,
            "pilot_present": 0,
        }

    tree_complete = 0
    tree_missing = 0
    tree_not_planned = 0
    tree_pilots = 0
    tree_missing_paths: list[str] = []
    tree_active_present = 0
    tree_active_missing = 0
    tree_active_not_planned_absent = 0

    for row in PLAN:
        if row["tree"] != tree:
            continue

        model = row["model"]
        split = row["split"]
        status = row["status"]
        reason = row.get("reason", "")

        pred_dir = root_dir / split
        split_prefix = f"{split}/" if split else ""

        if not pred_dir.is_dir():
            if status == "planned":
                print(f"  {model} [{split}]: directory missing -> {CELLS_PER_SPLIT} missing")
                tree_missing += CELLS_PER_SPLIT
                tree_missing_paths.extend([f"{split_prefix}<all>"] * CELLS_PER_SPLIT)
            else:
                print(f"  {model} [{split}]: not planned — {reason}; directory missing")
                tree_not_planned += CELLS_PER_SPLIT
            continue

        present = 0
        missing_paths: list[str] = []
        for target in TARGETS:
            for fold in FOLDS:
                for seed in SEEDS:
                    fname = _cell_filename(model, split, suffix, target, fold, seed)
                    ok, _ = inspect(pred_dir / fname, deep)
                    if ok:
                        present += 1
                    else:
                        missing_paths.append(f"{split_prefix}{fname}")

        if status == "planned":
            complete = present
            missing = CELLS_PER_SPLIT - present
            tree_complete += complete
            tree_missing += missing
            tree_missing_paths.extend(missing_paths)
            print(
                f"  {model} [{split}]: {complete}/{CELLS_PER_SPLIT} complete · "
                f"{missing} missing"
            )
        else:
            pilots = present
            absent = CELLS_PER_SPLIT - present
            tree_not_planned += CELLS_PER_SPLIT
            tree_pilots += pilots
            pilot_note = f"; {pilots} pilot(s) present" if pilots else ""
            print(
                f"  {model} [{split}]: not planned — {reason}{pilot_note} · "
                f"{absent} absent"
            )
            if verbose and missing_paths:
                for mp in missing_paths:
                    print(f"    NOT PLANNED (absent) {mp}")

        if row.get("scope", "active") == "active":
            tree_active_present += present
            if status == "planned":
                tree_active_missing += CELLS_PER_SPLIT - present
            else:
                tree_active_not_planned_absent += CELLS_PER_SPLIT - present
        if verbose and status == "planned":
            for mp in missing_paths:
                print(f"    MISSING {mp}")

    print(
        f"  -> tree total: {tree_complete} complete · {tree_missing} missing · "
        f"{tree_not_planned} not planned ({tree_pilots} pilot(s) present)"
    )

    if tree_missing_paths:
        by_seed: dict[int, int] = {}
        for mp in tree_missing_paths:
            try:
                seed_part = mp.split("__s")[-1]
                seed = int(seed_part.split("__")[0].split(".")[0])
            except (IndexError, ValueError):
                seed = -1
            by_seed[seed] = by_seed.get(seed, 0) + 1
        seed_summary = "  ".join(f"s{s}={n}" for s, n in sorted(by_seed.items()) if s != -1)
        print(f"  missing by seed: {seed_summary}")

    return {
        "complete": tree_complete,
        "missing": tree_missing,
        "not_planned": tree_not_planned,
        "pilot_present": tree_pilots,
        "active_present": tree_active_present,
        "active_missing": tree_active_missing,
        "active_not_planned_absent": tree_active_not_planned_absent,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--verbose", action="store_true", help="list every missing cell")
    ap.add_argument("--deep", action="store_true", help="open each npz and check its arrays")
    args = ap.parse_args()

    full_totals = {"complete": 0, "missing": 0, "not_planned": 0, "pilot_present": 0}
    active_totals = {"present": 0, "missing": 0, "not_planned_absent": 0}
    for tree in ("regen_v1", "octamer_k1", "wdmpnn_original"):
        tree_totals = _check_tree(tree, args.deep, args.verbose)
        for key in full_totals:
            full_totals[key] += tree_totals[key]
        for key in active_totals:
            active_totals[key] += tree_totals[f"active_{key}"]

    full_present = full_totals["complete"] + full_totals["pilot_present"]
    tracked_not_planned_cells = sum(
        CELLS_PER_SPLIT
        for row in PLAN
        if row.get("scope") == "tracked"
    )

    print("\n" + "-" * 62)
    print(
        f"Grand total: {active_totals['present']} present · "
        f"{active_totals['missing']} missing · "
        f"{active_totals['not_planned_absent']} not planned"
    )
    if tracked_not_planned_cells:
        print(
            f"  ({tracked_not_planned_cells} additional not-planned cells outside the active "
            f"campaign scope; {full_present} present across the full tracked inventory)"
        )
    if active_totals["missing"]:
        print("Run with --verbose to list every missing file.")
    elif active_totals["missing"] == 0 and active_totals["not_planned_absent"] == 0:
        print("All active-scope planned cells are present and well-formed.")


if __name__ == "__main__":
    main()
