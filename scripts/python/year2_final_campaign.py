#!/usr/bin/env python3
"""Generate, validate, and resolve the frozen Year 2 corrected campaign manifests."""

from __future__ import annotations

import argparse
import csv
import shlex
from itertools import product
from pathlib import Path

NAMESPACE = "year2_corrected_final_2026"
ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = ROOT / "manifests" / NAMESPACE
HTPMD_MANIFEST = MANIFEST_DIR / "htpmd_cells.tsv"
EAIP_MANIFEST = MANIFEST_DIR / "eaip_cells.tsv"
SPLITS = ((0, 42), (1, 43), (2, 44), (3, 45), (4, 46))
INIT_SEED = 42

HTPMD_MODELS = (
    ("dmpnn", "dmpnn"),
    ("gin", "gin"),
    ("gat", "gat"),
    ("wdmpnn-published", "wd_published"),
    ("hpg-published", "hpg_published"),
)
HTPMD_CONDITIONS = (
    ("native", "conda_native"),
    ("dop", "condb_dop"),
    ("state", "condc_dop_molality"),
)
HTPMD_TARGETS = (
    ("Conductivity", "conductivity"),
    ("TFSI Diffusivity", "tfsi_diffusivity"),
    ("Li Diffusivity", "li_diffusivity"),
    ("Poly Diffusivity", "polymer_diffusivity"),
    ("Transference Number", "transference_number"),
)
EAIP_MODELS = (
    ("dmpnn", "dmpnn"),
    ("gin", "gin"),
    ("gat", "gat"),
    ("wd-published", "wd_published"),
    ("wd-published-xn8", "wd_published_xn8"),
)
EAIP_TARGETS = (
    ("EA vs SHE (eV)", "ea"),
    ("IP vs SHE (eV)", "ip"),
)
FIELDS = (
    "array_index",
    "canonical_run_id",
    "dataset",
    "runner_model_value",
    "model_slug",
    "condition_value",
    "condition_slug",
    "target_value",
    "target_slug",
    "split_index",
    "split_seed",
    "init_seed",
    "output_dir",
)


def expected_rows(dataset: str) -> list[dict[str, str]]:
    rows = []
    if dataset == "htpmd":
        combinations = product(HTPMD_MODELS, HTPMD_CONDITIONS, HTPMD_TARGETS, SPLITS)
        for array_index, ((model, model_slug), (condition, condition_slug), (target, target_slug), (split_index, split_seed)) in enumerate(combinations):
            run_id = (
                f"htpmd__{model_slug}__{condition_slug}__{target_slug}__"
                f"split{split_index:02d}_seed{split_seed}__init{INIT_SEED}"
            )
            rows.append({
                "array_index": str(array_index),
                "canonical_run_id": run_id,
                "dataset": "htpmd",
                "runner_model_value": model,
                "model_slug": model_slug,
                "condition_value": condition,
                "condition_slug": condition_slug,
                "target_value": target,
                "target_slug": target_slug,
                "split_index": str(split_index),
                "split_seed": str(split_seed),
                "init_seed": str(INIT_SEED),
                "output_dir": f"results/{NAMESPACE}/htpmd/{run_id}",
            })
        return rows
    if dataset == "eaip":
        combinations = product(EAIP_MODELS, EAIP_TARGETS, SPLITS)
        for array_index, ((model, model_slug), (target, target_slug), (split_index, split_seed)) in enumerate(combinations):
            run_id = (
                f"eaip__{model_slug}__{target_slug}__"
                f"split{split_index:02d}_seed{split_seed}__init{INIT_SEED}"
            )
            rows.append({
                "array_index": str(array_index),
                "canonical_run_id": run_id,
                "dataset": "eaip",
                "runner_model_value": model,
                "model_slug": model_slug,
                "condition_value": "",
                "condition_slug": "",
                "target_value": target,
                "target_slug": target_slug,
                "split_index": str(split_index),
                "split_seed": str(split_seed),
                "init_seed": str(INIT_SEED),
                "output_dir": f"results/{NAMESPACE}/eaip/{run_id}",
            })
        return rows
    raise ValueError(f"Unknown dataset {dataset!r}")


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"Unexpected manifest header in {path}")
        return list(reader)


def validate_manifest(dataset: str, path: Path) -> list[dict[str, str]]:
    actual = read_manifest(path)
    expected = expected_rows(dataset)
    expected_count = 375 if dataset == "htpmd" else 50
    if len(actual) != expected_count:
        raise ValueError(f"{dataset} manifest has {len(actual)} rows, expected {expected_count}")
    if actual != expected:
        for index, (actual_row, expected_row) in enumerate(zip(actual, expected)):
            if actual_row != expected_row:
                raise ValueError(f"{dataset} manifest row {index} differs: {actual_row} != {expected_row}")
        raise ValueError(f"{dataset} manifest differs from the frozen Cartesian product")
    indices = [int(row["array_index"]) for row in actual]
    run_ids = [row["canonical_run_id"] for row in actual]
    output_dirs = [row["output_dir"] for row in actual]
    if indices != list(range(expected_count)):
        raise ValueError(f"{dataset} array indices are not contiguous and zero-based")
    if len(set(run_ids)) != expected_count:
        raise ValueError(f"{dataset} canonical run IDs are not unique")
    if len(set(output_dirs)) != expected_count:
        raise ValueError(f"{dataset} output directories are not unique")
    for row in actual:
        if int(row["init_seed"]) != INIT_SEED:
            raise ValueError(f"Unexpected initialization seed in {row}")
        split_index = int(row["split_index"])
        if int(row["split_seed"]) != dict(SPLITS)[split_index]:
            raise ValueError(f"Incorrect split seed mapping in {row}")
        prefix = f"results/{NAMESPACE}/{dataset}/"
        if not row["output_dir"].startswith(prefix):
            raise ValueError(f"Output path escapes the campaign namespace: {row['output_dir']}")
        if any(token in row["output_dir"] for token in ("checkpoints/", "predictions/", "HTPMDRepresentationBenchmark", "EAIPRepresentationBenchmark")):
            raise ValueError(f"Output path points to a legacy root: {row['output_dir']}")
    return actual


def command_for(row: dict[str, str], project_dir: Path = ROOT) -> list[str]:
    output_dir = project_dir / row["output_dir"]
    common = ["--output_dir", str(output_dir), "--split_id", row["split_index"], "--init_seed", row["init_seed"], "--batch_size", "64", "--max_epochs", "300", "--patience", "30", "--device", "cuda"]
    if row["dataset"] == "htpmd":
        return [
            "python3", "scripts/python/run_htpmd_representation_benchmark.py",
            "--data", str(project_dir / "data" / "htpmd.csv"),
            "--splits", str(project_dir / "metadata" / "splits" / "htpmd_chemistry_disjoint.json"),
            "--representations", row["runner_model_value"],
            "--condition", row["condition_value"],
            "--target", row["target_value"],
            "--wd_convention", "corrected",
            *common,
        ]
    return [
        "python3", "scripts/python/run_ea_ip_representation_benchmark.py",
        "--data", str(project_dir / "data" / "ea_ip.csv"),
        "--published_data", str(project_dir / "polymer-chemprop-data" / "datasets" / "vipea" / "chemprop_inputs" / "dataset-poly_chemprop.csv"),
        "--splits", str(project_dir / "metadata" / "splits" / "ea_ip_chemistry_disjoint.json"),
        "--models", row["runner_model_value"],
        "--target", row["target_value"],
        *common,
    ]


def validate_commands(rows: list[dict[str, str]], project_dir: Path = ROOT) -> list[str]:
    commands = []
    for row in rows:
        command = shlex.join(command_for(row, project_dir))
        if "--max_batches" in command:
            raise ValueError(f"Frozen command contains --max_batches: {command}")
        if "--init_seed 42" not in command:
            raise ValueError(f"Frozen command has incorrect initialization seed: {command}")
        if row["dataset"] == "htpmd" and "--wd_convention corrected" not in command:
            raise ValueError(f"HTPMD command lacks corrected wD convention: {command}")
        commands.append(command)
    if len(commands) != len(set(commands)):
        raise ValueError("Resolved commands are not unique")
    return commands


def print_shell_resolution(dataset: str, array_index: int, project_dir: Path) -> None:
    path = HTPMD_MANIFEST if dataset == "htpmd" else EAIP_MANIFEST
    rows = validate_manifest(dataset, path)
    if not 0 <= array_index < len(rows):
        raise ValueError(f"Array index {array_index} is out of range for {dataset}")
    row = rows[array_index]
    command_parts = command_for(row, project_dir)
    assignments = {
        "CANONICAL_RUN_ID": row["canonical_run_id"],
        "DATASET": row["dataset"],
        "RUNNER_MODEL": row["runner_model_value"],
        "MODEL_SLUG": row["model_slug"],
        "CONDITION": row["condition_value"],
        "CONDITION_SLUG": row["condition_slug"],
        "TARGET": row["target_value"],
        "TARGET_SLUG": row["target_slug"],
        "SPLIT_INDEX": row["split_index"],
        "SPLIT_SEED": row["split_seed"],
        "INIT_SEED": row["init_seed"],
        "CELL_DIR": str(project_dir / row["output_dir"]),
        "EXACT_PYTHON_COMMAND": shlex.join(command_parts),
    }
    for key, value in assignments.items():
        print(f"{key}={shlex.quote(value)}")
    print("PYTHON_COMMAND=(" + " ".join(shlex.quote(part) for part in command_parts) + ")")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifests", action="store_true")
    parser.add_argument("--dry-run", choices=("htpmd", "eaip", "all"))
    parser.add_argument("--resolve", choices=("htpmd", "eaip"))
    parser.add_argument("--array-index", type=int)
    parser.add_argument("--project-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.write_manifests:
        write_manifest(HTPMD_MANIFEST, expected_rows("htpmd"))
        write_manifest(EAIP_MANIFEST, expected_rows("eaip"))
    if args.resolve:
        if args.array_index is None:
            parser.error("--resolve requires --array-index")
        print_shell_resolution(args.resolve, args.array_index, args.project_dir)
        return
    datasets = ("htpmd", "eaip") if args.dry_run in (None, "all") else (args.dry_run,)
    all_commands = []
    for dataset in datasets:
        path = HTPMD_MANIFEST if dataset == "htpmd" else EAIP_MANIFEST
        rows = validate_manifest(dataset, path)
        commands = validate_commands(rows, args.project_dir)
        all_commands.extend(commands)
        print(f"{dataset}: rows={len(rows)} unique_ids={len({row['canonical_run_id'] for row in rows})} unique_outputs={len({row['output_dir'] for row in rows})}")
        if args.dry_run:
            for row, command in zip(rows, commands):
                print(f"{row['array_index']}\t{row['canonical_run_id']}\t{command}")
    print(f"total_cells={len(all_commands)} unique_commands={len(set(all_commands))}")


if __name__ == "__main__":
    main()
