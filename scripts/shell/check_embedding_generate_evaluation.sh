#!/usr/bin/env python3
"""
Scan ROOT_DIR/results/embeddings for saved embeddings and evaluate them.

Embedding prefix format (as produced by your training code):
    {dataset}__{model}__{target}{desc}{rdkit}{batch_norm}{size}

Saved files required (for i in 0..4):
    {prefix}__X_train_split_{i}.npy
    {prefix}__X_val_split_{i}.npy
    {prefix}__X_test_split_{i}.npy
    {prefix}__feature_mask_split_{i}.npy

If and only if ALL of the above exist for i=0..4, we call evaluate_model.py
exactly once for that prefix.

CLI filters:
    --dataset, --model, --target to restrict which prefixes to consider.

Usage examples:
    python check_and_eval_embeddings.py --root-dir /scratch/um09/hl4138/dmpnn \
      --eval-script scripts/python/evaluate_model.py --dry-run

    python check_and_eval_embeddings.py --root-dir . --dataset opv_camb3lyp \
      --model AttentiveFP --auto-run
"""

import argparse
import re
import subprocess
from pathlib import Path
from collections import defaultdict

RE_SPLIT = re.compile(r'__(X_train|X_val|X_test|feature_mask)_split_([0-4])\.npy$')

def parse_prefix_parts(prefix: str):
    """
    Given a prefix like:
        dataset__model__target words maybe with spaces__desc__rdkit__batch_norm__size1024
    Return (dataset, model, target, flags_dict)

    Flags dict: {'desc':bool, 'rdkit':bool, 'batch_norm':bool, 'size':int|None}
    """
    parts = prefix.split('__')
    if len(parts) < 3:
        return None  # malformed

    dataset = parts[0]
    model   = parts[1]
    tail    = parts[2:]  # [target maybe-with-spaces, optional flags]

    # Known flag tokens
    flags = {'desc': False, 'rdkit': False, 'batch_norm': False, 'size': None}

    # target is the first element of tail until we hit a known flag token.
    # (Target itself may contain spaces; it's a single segment because we split on "__")
    if not tail:
        target = ""
    else:
        target = tail[0]

    # Parse remaining segments as flags (order-agnostic)
    for seg in tail[1:]:
        if seg == 'desc':
            flags['desc'] = True
        elif seg == 'rdkit':
            flags['rdkit'] = True
        elif seg == 'batch_norm':
            flags['batch_norm'] = True
        elif seg.startswith('size'):
            # accept sizeXXXX or size=XXXX
            m = re.match(r'size[_=]?(\d+)$', seg)
            if m:
                flags['size'] = int(m.group(1))

    return dataset, model, target, flags


def collect_complete_prefixes(emb_dir: Path):
    """
    Walk embeddings dir, group files by their 'prefix' (everything before __X_* or __feature_mask_*),
    and keep only those prefixes that have all required files for splits 0..4.
    """
    # Map prefix -> {'X_train': set(splits), 'X_val':..., 'X_test':..., 'feature_mask': set(splits)}
    groups = defaultdict(lambda: defaultdict(set))

    for path in emb_dir.rglob('*.npy'):
        name = path.name
        m = RE_SPLIT.search(name)
        if not m:
            continue
        kind = m.group(1)           # X_train / X_val / X_test / feature_mask
        split = int(m.group(2))     # 0..4
        prefix = name[:m.start()]   # everything before "__{kind}_split_i.npy"
        groups[prefix][kind].add(split)

    complete = []
    required_kinds = ['X_train', 'X_val', 'X_test', 'feature_mask']
    required_splits = set(range(5))
    for prefix, kinds in groups.items():
        ok = all(k in kinds and kinds[k] == required_splits for k in required_kinds)
        if ok:
            complete.append(prefix)
    return complete


def build_eval_cmd(eval_script: Path,
                   dataset: str,
                   model: str,
                   target: str,
                   flags: dict,
                   embeddings_prefix_path: Path):
    """
    Construct the evaluate_model.py command.

    NOTE: Adjust flag names here if your evaluator uses different options.
    """
    cmd = [
        'python3', str(eval_script),
        '--model_name', model,
        '--dataset_name', dataset,
        '--use_embeddings',
        '--embeddings_prefix', str(embeddings_prefix_path),
    ]
    if target:
        cmd += ['--target', target]
    if flags.get('desc'):
        cmd += ['--incl_desc']
    if flags.get('rdkit'):
        cmd += ['--incl_rdkit']
    if flags.get('batch_norm'):
        cmd += ['--batch_norm']
    if flags.get('size') is not None:
        cmd += ['--size', str(flags['size'])]

    return cmd


def main():
    ap = argparse.ArgumentParser(description="Check saved embeddings and run evaluator.")
    ap.add_argument('--root-dir', required=True,
                    help='Project root (embeddings under ROOT_DIR/results/embeddings)')
    ap.add_argument('--eval-script', required=True,
                    help='Path to evaluate_model.py')
    ap.add_argument('--dataset', default='', help='Filter by dataset (exact match)')
    ap.add_argument('--model', default='', help='Filter by model (exact match)')
    ap.add_argument('--target', default='', help='Filter by target (exact match)')
    ap.add_argument('--dry-run', action='store_true', help='Print actions only')
    ap.add_argument('--auto-run', action='store_true', help='Execute evaluator for each complete prefix')
    args = ap.parse_args()

    emb_dir = Path(args.root_dir).expanduser().resolve() / 'results' / 'embeddings'
    eval_script = Path(args.eval_script).expanduser().resolve()

    if not emb_dir.exists():
        print(f"ERROR: Embeddings dir not found: {emb_dir}")
        return
    if not eval_script.exists():
        print(f"ERROR: Evaluator not found: {eval_script}")
        return

    print(f"🔎 Scanning embeddings in: {emb_dir}")
    complete_prefixes = sorted(collect_complete_prefixes(emb_dir))
    if not complete_prefixes:
        print("No complete 5-split embedding sets found.")
        return

    print(f"Found {len(complete_prefixes)} complete prefix(es).")

    ran = 0
    for prefix in complete_prefixes:
        parsed = parse_prefix_parts(prefix)
        if parsed is None:
            print(f"Skipping malformed prefix: {prefix}")
            continue
        dataset, model, target, flags = parsed

        # Apply filters
        if args.dataset and args.dataset != dataset:
            continue
        if args.model and args.model != model:
            continue
        if args.target and args.target != target:
            continue

        # Build command
        embeddings_prefix_path = emb_dir / prefix
        cmd = build_eval_cmd(eval_script, dataset, model, target, flags, embeddings_prefix_path)

        # Report
        info = f"{dataset} | {model} | {target or 'ALL'} | flags={flags} | prefix='{prefix}'"
        if args.dry_run and not args.auto_run:
            print(f"📝 DRY-RUN would run: {info}")
            print("   ", " ".join(repr(c) for c in cmd))
            continue

        print(f"🚀 Evaluating: {info}")
        print("   ", " ".join(repr(c) for c in cmd))

        if args.auto_run:
            try:
                subprocess.run(cmd, check=True)
                ran += 1
            except subprocess.CalledProcessError as e:
                print(f"⚠️ Evaluation failed for prefix '{prefix}': {e}")
        else:
            # If not auto_run, just print the command once more as copy-pasteable
            print("   (copy/paste to run)")
            print("   " + " ".join(cmd))

    if args.auto_run:
        print(f"✅ Done. Evaluations launched: {ran}")
    else:
        print("✅ Done. (No commands executed; use --auto-run to execute.)")


if __name__ == '__main__':
    main()
