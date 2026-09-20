#!/usr/bin/env python3
"""Run selected corrected HTPMD representation-benchmark cells."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.htpmd_benchmark import (
    HTPMD_TARGETS,
    INFORMATION_CONDITIONS,
    REPRESENTATIONS,
    featurize_htpmd,
    fit_one_run,
    load_split_artifact,
    validate_htpmd_frame,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "htpmd.csv")
    parser.add_argument("--splits", type=Path, default=ROOT / "metadata" / "splits" / "htpmd_chemistry_disjoint.json")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "results" / "HTPMDRepresentationBenchmark")
    parser.add_argument("--representations", nargs="+", choices=REPRESENTATIONS, default=list(REPRESENTATIONS))
    parser.add_argument("--condition", choices=INFORMATION_CONDITIONS, default="dop")
    parser.add_argument("--target", choices=HTPMD_TARGETS, default="Conductivity")
    parser.add_argument("--split_id", type=int, default=0)
    parser.add_argument("--init_seed", type=int, default=42)
    parser.add_argument("--wd_convention", choices=("corrected", "historical"), default="corrected")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    frame = pd.read_csv(args.data)
    validate_htpmd_frame(frame)
    artifact = load_split_artifact(frame, args.splits)
    split = artifact["splits"][args.split_id]
    for representation in args.representations:
        graphs = featurize_htpmd(frame, representation, args.wd_convention)
        provenance = fit_one_run(
            frame=frame,
            graphs=graphs,
            representation=representation,
            condition=args.condition,
            target=args.target,
            split=split,
            split_reference=args.splits,
            init_seed=args.init_seed,
            output_dir=args.output_dir,
            wd_convention=args.wd_convention,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            max_batches=args.max_batches,
            device=args.device,
        )
        print(
            representation,
            provenance["embedding_dimension"],
            provenance["checkpoint_selected"],
        )


if __name__ == "__main__":
    main()
