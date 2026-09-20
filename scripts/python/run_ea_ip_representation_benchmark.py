#!/usr/bin/env python3
"""Run selected corrected EA/IP representation-benchmark cells."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.ea_ip_benchmark import (
    EA_IP_MODELS,
    EA_IP_TARGETS,
    certify_published_parity,
    featurize_ea_ip,
    fit_one_run,
    load_split_artifact,
    validate_ea_ip_frame,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "ea_ip.csv")
    parser.add_argument(
        "--published_data", type=Path,
        default=ROOT / "polymer-chemprop-data" / "datasets" / "vipea" / "chemprop_inputs" / "dataset-poly_chemprop.csv",
    )
    parser.add_argument(
        "--splits", type=Path,
        default=ROOT / "metadata" / "splits" / "ea_ip_chemistry_disjoint.json",
    )
    parser.add_argument(
        "--output_dir", type=Path,
        default=ROOT / "results" / "EAIPRepresentationBenchmark",
    )
    parser.add_argument("--models", nargs="+", choices=EA_IP_MODELS, default=list(EA_IP_MODELS))
    parser.add_argument("--target", choices=EA_IP_TARGETS, default=EA_IP_TARGETS[0])
    parser.add_argument("--split_id", type=int, default=0)
    parser.add_argument("--init_seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    frame = pd.read_csv(args.data)
    published = pd.read_csv(args.published_data)
    validate_ea_ip_frame(frame)
    parity = certify_published_parity(frame, published)
    artifact = load_split_artifact(frame, args.splits)
    split = artifact["splits"][args.split_id]
    print("published_parity", parity)
    for model in args.models:
        inputs = featurize_ea_ip(frame, model)
        provenance = fit_one_run(
            frame=frame,
            inputs=inputs,
            model_name=model,
            target=args.target,
            split=split,
            split_reference=args.splits,
            init_seed=args.init_seed,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            max_batches=args.max_batches,
            device=args.device,
        )
        print(model, provenance["embedding_dimension"], provenance["checkpoint_selected"])


if __name__ == "__main__":
    main()
