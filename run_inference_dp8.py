#!/usr/bin/env python3
"""Original polymer-chemprop inference; configure paths below or use CLI options.

Transitions are counted on linear A/B strings, with no end-to-start wrap.
Cross weights are split equally over the four A/B attachment pairings.
The original API normalizes incoming weights: residual same-type weights
are 1-P(B->A) on A and 1-P(A->B) on B. These are a graph normalization
completion, not measured AA/BB probabilities for a finite linear chain.
Omitted Xn means Xn=1 in the original implementation, not removal of its
DP dependence. Only load trusted checkpoints (the upstream loader uses pickle).
"""

import argparse
import csv
import logging
import math
from pathlib import Path
import re
import sys
import tempfile

SEQUENCES_CSV = "[PATH_TO_DP8_CSV]"
ACIDS_CSV = "[PATH_TO_ACIDS_CSV]"
BROMIDES_CSV = "[PATH_TO_BROMIDES_CSV]"
MODEL_CHECKPOINT = "[PATH_TO_MODEL_CHECKPOINT]"
POLYMER_CHEMPROP = "[PATH_TO_POLYMER_CHEMPROP]"
STORAGE_PATH = Path("/scratch/um09/hl4138/dmpnn")


def find_column(headers, aliases):
    normalize = lambda value: re.sub(r"[^a-z0-9]", "", value.lower())
    for alias in aliases:
        matches = [h for h in headers if normalize(h) == normalize(alias)]
        if len(matches) == 1:
            return matches[0]
    raise ValueError(f"Expected one of {aliases}; found columns {headers}")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if not rows:
            raise ValueError(f"Empty CSV: {path}")
        return reader.fieldnames, rows


def monomer_lookup(path):
    headers, rows = read_csv(path)
    smiles = headers[0] if headers == ["0"] else find_column(headers, ["smiles", "monomer_smiles", "canonical_smiles"])
    ids = [h for h in headers if h.lower() in {"id", "monomer_id", "index", "unnamed: 0"}]
    if len(ids) > 1:
        raise ValueError(f"Ambiguous monomer IDs in {path}: {ids}")
    lookup = {}
    for i, row in enumerate(rows):
        key = row[ids[0]].strip() if ids else str(i)
        if key in lookup:
            raise ValueError(f"Duplicate monomer ID {key} in {path}")
        lookup[key] = row[smiles].strip()
    return lookup


def transitions(sequence, dp):
    sequence = sequence.strip().upper()
    if len(sequence) != dp or set(sequence) != {"A", "B"}:
        raise ValueError(f"Expected exactly {dp} A/B symbols: {sequence!r}")
    if sequence.count("A") != dp // 2:
        raise ValueError(f"Expected 50:50 composition: {sequence!r}")
    pairs = list(zip(sequence, sequence[1:]))
    ab = pairs.count(("A", "B")) / sequence.count("A")
    ba = pairs.count(("B", "A")) / sequence.count("B")
    if not all(math.isfinite(p) and 0 <= p <= 1 for p in (ab, ba)):
        raise ValueError(f"Invalid transition probabilities for {sequence!r}: {ab}, {ba}")
    return ab, ba


def attachment_smiles(smiles, acid):
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid monomer SMILES: {smiles}")
    offset = 1 if acid else 3
    wildcards = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if wildcards:
        if len(wildcards) != 2:
            raise ValueError("Each monomer must have exactly two attachment points")
        rw = Chem.RWMol(mol)
        for i, idx in enumerate(sorted(wildcards, key=lambda idx: mol.GetAtomWithIdx(idx).GetAtomMapNum())):
            rw.GetAtomWithIdx(idx).SetAtomMapNum(offset + i)
        return Chem.MolToSmiles(rw)
    pattern = Chem.MolFromSmarts("[B](-[O])(-[O])" if acid else "[c]-[Br]")
    matches = mol.GetSubstructMatches(pattern)
    sites = [(match[0], match[1:]) if acid else (match[1], ()) for match in matches]
    if len(sites) != 2:
        raise ValueError(f"Expected two boronic-acid or aryl-Br sites: {smiles}")
    rw = Chem.RWMol(mol)
    remove = []
    for i, (idx, leaving) in enumerate(sites):
        atom = rw.GetAtomWithIdx(idx)
        atom.SetAtomicNum(0)
        atom.SetFormalCharge(0)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
        atom.SetAtomMapNum(offset + i)
        remove.extend(leaving)
    for idx in sorted(set(remove), reverse=True):
        rw.RemoveAtom(idx)
    Chem.SanitizeMol(rw)
    return Chem.MolToSmiles(rw)


def polymer_string(a, b, ab, ba, xn=None):
    edges = [(1, 2, 1 - ba, 1 - ba), (3, 4, 1 - ab, 1 - ab)]
    edges += [(i, j, ab / 2, ba / 2) for i in (1, 2) for j in (3, 4)]
    rules = "".join(f"<{i}-{j}:{w12:.17g}:{w21:.17g}" for i, j, w12, w21 in edges)
    return f"{a}.{b}|0.5|0.5|{rules}" + (f"~{xn}" if xn is not None else "")


def main(dp=8, sequences_csv=SEQUENCES_CSV):
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in [("sequences", sequences_csv), ("acids", ACIDS_CSV), ("bromides", BROMIDES_CSV), ("checkpoint", MODEL_CHECKPOINT), ("code-root", POLYMER_CHEMPROP)]:
        parser.add_argument(f"--{name}", default=default)
    parser.add_argument("--output", type=Path, default=STORAGE_PATH / "inference_results" / f"inference_dp{dp}.csv")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    for name in ("sequences", "acids", "bromides", "checkpoint", "code_root"):
        value = getattr(args, name)
        if "[PATH_TO_" in value or not Path(value).exists():
            parser.error(f"Configure --{name.replace('_', '-')}: {value}")
    if args.output.exists():
        parser.error(f"Refusing to overwrite existing output: {args.output}")
    root = Path(args.code_root).resolve()
    sys.path.insert(0, str(root))
    import chemprop
    if root not in Path(chemprop.__file__).resolve().parents:
        raise RuntimeError(f"chemprop imported from {chemprop.__file__}, not {root}")
    from chemprop.args import PredictArgs
    from chemprop.train import make_predictions
    from chemprop.utils import load_args
    from chemprop.features import set_polymer
    from chemprop.data import get_data_from_smiles

    training = load_args(args.checkpoint)
    if not getattr(training, "polymer", False) or training.dataset_type != "regression":
        raise ValueError("Expected an original polymer-chemprop polymer regression checkpoint")
    if getattr(training, "number_of_molecules", 1) != 1:
        raise ValueError("Expected one polymer input per datapoint")
    if any(getattr(training, name, None) for name in ("features_path", "features_generator", "atom_descriptors", "bond_features_path")):
        raise ValueError("Checkpoint requires extra features not supplied by this experiment")
    tasks = list(training.task_names)
    targets = {}
    for target in ("EA", "IP"):
        matches = [i for i, task in enumerate(tasks) if re.match(rf"^{target}(?:$|[^A-Za-z])", task, flags=re.I)]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous {target} targets in checkpoint: {tasks}")
        if matches:
            targets[target] = matches[0]
        else:
            logging.warning("Checkpoint has no %s target; corresponding outputs will be blank", target)
    if not targets:
        raise ValueError(f"No identifiable EA/IP targets in checkpoint: {tasks}")
    logging.info("Checkpoint tasks in trained order: %s", tasks)
    headers, rows = read_csv(args.sequences)
    columns = {name: find_column(headers, aliases) for name, aliases in {
        "pair_id": ["pair_id", "poly_id", "monomer_pair_id"],
        "sequence": ["sequence", "sequence_string", "seq"],
        "J": ["J", "junction_count"],
        "run_length_entropy": ["run_length_entropy", "run_entropy"],
        "true_EA": ["true_EA", "EA", "EA (eV)", "EA vs SHE (eV)"],
        "true_IP": ["true_IP", "IP", "IP (eV)", "IP vs SHE (eV)"],
    }.items()}
    acids, bromides = monomer_lookup(args.acids), monomer_lookup(args.bromides)
    monomers, encoded, records = {}, [], []
    strata = {}
    failures = 0
    for line, row in enumerate(rows, 2):
        record = {key: row[value] for key, value in columns.items()}
        try:
            ab, ba = transitions(record["sequence"], dp)
            match = re.fullmatch(r"(\d+)[_:-](\d+)", record["pair_id"].strip())
            if not match:
                raise ValueError("pair_id must contain acid and bromide IDs, e.g. 0_12")
            pair = tuple(str(int(value)) for value in match.groups())
            if pair not in monomers:
                monomers[pair] = (attachment_smiles(acids[pair[0]], True), attachment_smiles(bromides[pair[1]], False))
            encoded.append(polymer_string(*monomers[pair], ab, ba))
            records.append(record)
            strata.setdefault((record["pair_id"], record["J"]), set()).add((ab, ba))
        except (ValueError, KeyError) as error:
            logging.error("CSV row %d, pair=%r, sequence=%r: %s", line, record["pair_id"], record["sequence"], error)
            failures += 1
    if failures:
        raise ValueError(f"{failures} invalid rows logged; no predictions written")
    mixed = sum(len(values) > 1 for values in strata.values())
    logging.info("%d of %d (pair_id,J) groups have differing directed transition inputs", mixed, len(strata))
    with tempfile.TemporaryDirectory(prefix=f"wdmpnn_dp{dp}_") as tmp:
        for label, xn in (("no_xn", None), ("with_xn", dp)):
            inputs = [s + (f"~{xn}" if xn is not None else "") for s in encoded]
            set_polymer(True)
            valid = get_data_from_smiles(smiles=[[s] for s in inputs], skip_invalid_smiles=False)
            if len(valid) != len(inputs) or any(any(mol is None for mol in datum.mol) for datum in valid):
                raise ValueError("Invalid polymer graph; refusing misaligned predictions")
            input_path = Path(tmp) / f"{label}_inputs.csv"
            with input_path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["smiles"])
                writer.writerows([[s] for s in inputs])
            prediction_args = PredictArgs().parse_args([
                "--test_path", str(input_path),
                "--preds_path", str(Path(tmp) / f"{label}.csv"),
                "--checkpoint_path", str(Path(args.checkpoint).resolve()),
                "--no_cuda", "--num_workers", "0", "--batch_size", "50",
            ])
            predictions = make_predictions(args=prediction_args, smiles=[[s] for s in inputs])
            if len(predictions) != len(records):
                raise ValueError("Prediction count differs from input count")
            seen = {}
            for record, polymer, prediction in zip(records, inputs, predictions):
                if prediction is None or len(prediction) != len(tasks) or not all(math.isfinite(float(p)) for p in prediction):
                    raise ValueError(f"Invalid prediction for {record['pair_id']}: {prediction}")
                if polymer in seen and any(not math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6) for a, b in zip(seen[polymer], prediction)):
                    raise ValueError("Identical graph inputs produced inconsistent predictions")
                seen[polymer] = prediction
                for target in ("EA", "IP"):
                    record[f"pred_{target}_{label}"] = prediction[targets[target]] if target in targets else ""
    fields = list(columns) + [f"pred_{target}_{label}" for label in ("no_xn", "with_xn") for target in ("EA", "IP")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    logging.info("Wrote %d rows to %s", len(records), args.output)


if __name__ == "__main__":
    main()
