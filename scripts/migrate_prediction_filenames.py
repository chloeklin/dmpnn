"""
Migration script: rename EA/IP prediction files to the canonical convention.

Canonical filename:
    ea_ip__{target}__{model}__{split}__fold{fold}.npz

Scans:
    predictions/ea_ip_group/
    predictions/ea_ip_pair/
    predictions/ea_ip_lomo/

For each file that uses an old name it:
  1. Derives the canonical name via evaluation.naming helpers.
  2. Copies (not moves) the file to the canonical name in the same directory.
  3. Logs old_name → new_name to predictions/migration_log.json.
  4. Skips if the canonical file already exists AND is byte-identical.
  5. Aborts (without overwriting) if canonical file exists but differs.

Usage:
    python scripts/migrate_prediction_filenames.py [--dry-run] [--force]

    --dry-run   Print what would be done; do not copy or write any files.
    --force     Overwrite existing canonical files even if they differ.
"""

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.naming import (
    standard_target_token,
    standard_model_name,
    standard_split_name,
    make_prediction_filename,
    parse_prediction_filename,
    DATASET_NAME,
    _SPLIT_DIRS,
)

PREDICTIONS_ROOT = ROOT / "predictions"
LOG_PATH = PREDICTIONS_ROOT / "migration_log.json"

# ── Pattern to detect files that need migration ───────────────────────────────
# Old names have spaces / capital letters / underscores in target field or
# use old model/split tokens.

_OLD_TARGET_MAP = {
    "EA vs SHE (eV)": "EA_vs_SHE_eV",
    "IP vs SHE (eV)": "IP_vs_SHE_eV",
}
_OLD_MODEL_TOKENS = {
    "stage2d_frac", "stage2d_2d0_arch", "stage2d_2d1_arch",
    "copoly_stage2d_frac", "copoly_stage2d_2d0_arch", "copoly_stage2d_2d1_arch",
    "wDMPNN",
}
_OLD_SPLIT_TOKENS = {"a_held_out", "lomo", "LOMO", "lomao", "LOMAO"}

# Regex patterns for each directory's old naming schemes
# group / pair:   ea_ip__{raw_target}__{old_model}__{split}__fold{N}.npz
# lomo:           ea_ip__{raw_target}__{old_model}__{split}__split{N}.npz  OR
#                 ea_ip__{raw_target}__a_held_out__split{N}.npz  (wdmpnn, no model token)

_LOMO_PATTERNS = [
    # copoly_stage2d_* style  →  ea_ip__<target>__copoly_stage2d_{variant}__a_held_out__split{N}.npz
    re.compile(
        r"^ea_ip__(?P<target>.+?)__(?P<model>copoly_stage2d_\w+)__(?P<split>a_held_out)__split(?P<fold>\d+)\.npz$"
    ),
    # wDMPNN style  →  ea_ip__<target>__a_held_out__split{N}.npz  (no model in name)
    re.compile(
        r"^ea_ip__(?P<target>.+?)__(?P<split>a_held_out)__split(?P<fold>\d+)\.npz$"
    ),
]

_GEN_PATTERN = re.compile(
    r"^ea_ip__(?P<target>.+?)__(?P<model>[\w]+)__(?P<split>group_disjoint|pair_disjoint)__fold(?P<fold>\d+)\.npz$"
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _decode_target(raw: str) -> str:
    """Return canonical target token from a raw target string in filename."""
    # raw may contain spaces (old format) or already be a token
    if raw in _OLD_TARGET_MAP:
        return _OLD_TARGET_MAP[raw]
    return standard_target_token(raw)


def _plan_lomo(fname: str) -> dict | None:
    """Return migration plan for an ea_ip_lomo file, or None if already canonical."""
    for pat in _LOMO_PATTERNS:
        m = pat.match(fname)
        if m:
            gd = m.groupdict()
            target_tok = _decode_target(gd["target"])
            # model: either explicit or inferred as wdmpnn when missing
            model_raw  = gd.get("model", "wDMPNN") or "wDMPNN"
            model_tok  = standard_model_name(model_raw)
            split_tok  = standard_split_name(gd["split"])
            fold       = int(gd["fold"])
            canonical  = make_prediction_filename(
                target_tok, model_tok, split_tok, fold
            )
            if fname == canonical:
                return None  # already canonical
            return dict(
                old=fname, new=canonical,
                target=target_tok, model=model_tok,
                split=split_tok, fold=fold,
            )
    return None


def _plan_gen(fname: str) -> dict | None:
    """Return migration plan for an ea_ip_group or ea_ip_pair file."""
    m = _GEN_PATTERN.match(fname)
    if not m:
        # Try already-canonical form
        try:
            parse_prediction_filename(fname)
            return None  # already canonical
        except ValueError:
            pass
        return None  # unrecognised
    gd = m.groupdict()
    target_tok = _decode_target(gd["target"])
    model_tok  = standard_model_name(gd["model"])
    split_tok  = standard_split_name(gd["split"])
    fold       = int(gd["fold"])
    canonical  = make_prediction_filename(target_tok, model_tok, split_tok, fold)
    if fname == canonical:
        return None
    return dict(
        old=fname, new=canonical,
        target=target_tok, model=model_tok,
        split=split_tok, fold=fold,
    )


def collect_plans(predictions_root: Path) -> list[dict]:
    plans = []
    dirs_and_planners = [
        (predictions_root / "ea_ip_lomo",  _plan_lomo),
        (predictions_root / "ea_ip_group", _plan_gen),
        (predictions_root / "ea_ip_pair",  _plan_gen),
    ]
    for dirpath, planner in dirs_and_planners:
        if not dirpath.exists():
            print(f"  [skip] {dirpath.relative_to(predictions_root)} — directory not found")
            continue
        for p in sorted(dirpath.glob("*.npz")):
            plan = planner(p.name)
            if plan is not None:
                plan["dir"] = dirpath
                plans.append(plan)
    return plans


def run_migration(
    predictions_root: Path,
    dry_run: bool,
    force: bool,
) -> None:
    plans = collect_plans(predictions_root)

    if not plans:
        print("✓ All files already use the canonical naming convention — nothing to do.")
        return

    print(f"\nFound {len(plans)} file(s) to rename:\n")
    log_entries = []
    n_copied = n_skipped = n_conflict = 0

    for plan in plans:
        dirpath  = plan["dir"]
        old_path = dirpath / plan["old"]
        new_path = dirpath / plan["new"]
        rel_old  = old_path.relative_to(predictions_root)
        rel_new  = new_path.relative_to(predictions_root)

        print(f"  {rel_old}")
        print(f"    → {rel_new}")

        if dry_run:
            print("    [dry-run] would copy")
            log_entries.append({"old": str(rel_old), "new": str(rel_new), "action": "dry-run"})
            n_copied += 1
            continue

        # Check for conflicts
        if new_path.exists():
            if _sha256(old_path) == _sha256(new_path):
                print("    [skip] canonical file already exists and is identical")
                log_entries.append({"old": str(rel_old), "new": str(rel_new), "action": "skipped-identical"})
                n_skipped += 1
                continue
            elif not force:
                print(f"    [CONFLICT] canonical file exists and differs — use --force to overwrite")
                log_entries.append({"old": str(rel_old), "new": str(rel_new), "action": "conflict"})
                n_conflict += 1
                continue
            else:
                print("    [overwrite] --force specified")

        shutil.copy2(old_path, new_path)
        print("    ✓ copied")
        log_entries.append({"old": str(rel_old), "new": str(rel_new), "action": "copied"})
        n_copied += 1

    print(f"\nSummary: {n_copied} copied, {n_skipped} skipped (identical), "
          f"{n_conflict} conflicts")

    if not dry_run and log_entries:
        # Load existing log and append
        existing = []
        if LOG_PATH.exists():
            with open(LOG_PATH) as f:
                existing = json.load(f)
        existing.extend(log_entries)
        with open(LOG_PATH, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"Log written to {LOG_PATH.relative_to(ROOT)}")

    if n_conflict > 0:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print actions without copying any files")
    parser.add_argument("--force",   action="store_true",
                        help="Overwrite canonical files even if they differ from old")
    args = parser.parse_args()

    print("EA/IP Prediction File Migration")
    print("=" * 50)
    print(f"Predictions root : {PREDICTIONS_ROOT.relative_to(ROOT)}")
    print(f"Dry run          : {args.dry_run}")
    print(f"Force overwrite  : {args.force}")

    run_migration(PREDICTIONS_ROOT, dry_run=args.dry_run, force=args.force)
