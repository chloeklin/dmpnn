#!/usr/bin/env python3
"""Am I holding a complete set of regeneration results locally?

Checks the expected R1 (A-heldout) and R3 (B-heldout clustered) run matrices
against what is actually on this machine, and reports what is missing, what is
malformed, and what to do next.

    python "29-07-2026 supervisor_update/check_results.py"
    python "29-07-2026 supervisor_update/check_results.py" --verbose   # list every missing cell
    python "29-07-2026 supervisor_update/check_results.py" --deep      # also open each NPZ

A file is only counted as complete when all of these hold:
  * the .npz exists and is non-trivial in size
  * its .config.json provenance sidecar exists
  * the sidecar records non-zero epochs and plausible wall time
  * (--deep) the archive contains both y_pred and y_pred_final
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

STAGES = {
    "R1  A-heldout": {
        "dir": ROOT / "predictions" / "regen_v1" / "ea_ip_lomo",
        "split": "monomer_heldout",
        "models": ["hpg_hier", "wdmpnn", "hpg_hier_octamer",
                   "hpg_hier_junction", "hpg_hier_junction1"],
        "logs": ROOT / "logs" / "regen_v1" / "r1" / "tasks",
    },
    "R3  B-heldout clustered": {
        "dir": ROOT / "predictions" / "regen_v1" / "ea_ip_lomo_b_clustered",
        "split": "monomer_b_heldout_clustered",
        "models": ["hpg_hier", "wdmpnn", "hpg_hier_octamer", "hpg_hier_junction"],
        "logs": ROOT / "logs" / "regen_v1" / "r3" / "tasks",
    },
}
TARGETS = ["EA_vs_SHE_eV", "IP_vs_SHE_eV"]
FOLDS = range(9)
SEEDS = [42, 43, 44]
MIN_BYTES = 10_000


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
        except Exception as exc:                       # noqa: BLE001
            return False, f"npz unreadable: {type(exc).__name__}"
    return True, ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose", action="store_true", help="list every missing cell")
    ap.add_argument("--deep", action="store_true", help="open each npz and check its arrays")
    args = ap.parse_args()

    grand_missing = 0
    for stage, cfg in STAGES.items():
        expected, present, problems = [], [], []
        for model in cfg["models"]:
            for target in TARGETS:
                for fold in FOLDS:
                    for seed in SEEDS:
                        name = (f"ea_ip__{target}__{model}__{cfg['split']}"
                                f"__fold{fold}__s{seed}.npz")
                        cell = (model, target, fold, seed)
                        expected.append(cell)
                        ok, why = inspect(cfg["dir"] / name, args.deep)
                        (present if ok else problems).append((cell, why))

        n_exp, n_ok = len(expected), len(present)
        pct = 100.0 * n_ok / n_exp if n_exp else 0.0
        bar = "█" * int(pct // 4) + "·" * (25 - int(pct // 4))
        print(f"\n{stage}")
        print(f"  {bar}  {n_ok}/{n_exp} complete  ({pct:.1f}%)")
        if not cfg["dir"].is_dir():
            print(f"  ! prediction directory does not exist: {cfg['dir']}")

        # per-model summary makes it obvious whether a whole model failed to sync
        for model in cfg["models"]:
            got = sum(1 for c, _ in present if c[0] == model)
            want = len(TARGETS) * len(list(FOLDS)) * len(SEEDS)
            mark = "ok " if got == want else "   "
            print(f"    {mark}{model:<22} {got:>3}/{want}")

        # anything present-but-malformed is more urgent than merely absent
        malformed = [(c, w) for c, w in problems if w != "npz missing"]
        if malformed:
            print(f"  ! {len(malformed)} file(s) present but not usable:")
            for cell, why in malformed[:10]:
                print(f"      {cell[0]} {cell[1]} fold{cell[2]} s{cell[3]}: {why}")
            if len(malformed) > 10:
                print(f"      ... and {len(malformed) - 10} more")

        absent = [c for c, w in problems if w == "npz missing"]
        grand_missing += len(problems)
        if absent:
            by_seed = {s: sum(1 for c in absent if c[3] == s) for s in SEEDS}
            print(f"  missing by seed: " + "  ".join(f"s{s}={n}" for s, n in by_seed.items()))
            if args.verbose:
                for cell in absent:
                    print(f"      MISSING {cell[0]} {cell[1]} fold{cell[2]} s{cell[3]}")

        logs = cfg["logs"]
        n_logs = len(list(logs.glob("*.log"))) if logs.is_dir() else 0
        print(f"  task logs: {n_logs} in {logs.relative_to(ROOT) if logs.is_dir() else logs}")
        if n_logs == 0:
            print("    ! no task logs — the frozen-split assertion cannot be confirmed from logs")

    print("\n" + "-" * 62)
    if grand_missing == 0:
        print("All expected cells are present and well-formed. Next:")
        print("  venv/bin/python3 scripts/python/analyze_regen_v1.py")
        print("  venv/bin/python3 scripts/python/analyze_regen_v1_r3.py")
    else:
        print(f"{grand_missing} cell(s) outstanding. To pull the latest from Gadi:")
        print("  scripts/shell/download_regen_v1_artifacts.sh")
        print("Then re-run this check. To work with what you have in the meantime:")
        print("  venv/bin/python3 scripts/python/analyze_regen_v1.py --partial")
        print("  venv/bin/python3 scripts/python/analyze_regen_v1_r3.py --partial")
    print("\nTo compare against Gadi without downloading, run there:")
    print("  ls /scratch/um09/hl4138/dmpnn/predictions/regen_v1/ea_ip_lomo/*.npz | wc -l")
    print("  ls /scratch/um09/hl4138/dmpnn/predictions/regen_v1/ea_ip_lomo_b_clustered/*.npz | wc -l")
    print("  qstat -u $USER | tail -5        # anything still queued or running")


if __name__ == "__main__":
    main()
