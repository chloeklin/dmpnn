#!/usr/bin/env python3
"""Generate analytic wD-MPNN inputs without model inference.

Count linear transitions without an end-to-start wrap, dividing by total
source-monomer counts. Fixed J alone does not guarantee equal directed counts.
Cross weights are distributed over four attachment pairings. Residual same-type
weights complete the original format's incoming-weight normalization; they are
not measured finite-chain AA/BB frequencies. No degree-of-polymerization suffix
is added. The equality check concerns encoded inputs, not measured predictions.
"""

import argparse
import csv
import logging
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
SEQUENCES_CSV = ROOT / "data/eaip_variant/dp8_cls.csv"
DATASET_CSV = ROOT / "data/eaip_variant/dataset.csv"
EA_IP_CSV = ROOT / "data/ea_ip.csv"
OUTPUT_DIR = ROOT / "inference_results"


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


def pair_key(value):
    match = re.fullmatch(r"(\d+)[_:-](\d+)", value.strip())
    if not match:
        raise ValueError(f"Expected an acid/bromide pair ID such as 0_342: {value!r}")
    return "_".join(str(int(part)) for part in match.groups())


def monomer_lookup(dataset_path, ea_ip_path):
    headers, rows = read_csv(ea_ip_path)
    a_col = find_column(headers, ["smiles_A"])
    b_col = find_column(headers, ["smiles_B"])
    input_col = find_column(headers, ["WDMPNN_Input"])
    labelled = {}
    for row in rows:
        key = (row[a_col].strip(), row[b_col].strip())
        prefix = row[input_col].split("|", 1)[0].strip()
        labels = re.findall(r"\[\*:(\d+)\]", prefix)
        if sorted(labels) != ["1", "2", "3", "4"] or "." not in prefix:
            raise ValueError(f"Expected two attachment-labelled monomers: {prefix!r}")
        if key in labelled and labelled[key] != prefix:
            raise ValueError(f"Conflicting labelled SMILES for monomers {key}")
        labelled[key] = prefix
    headers, rows = read_csv(dataset_path)
    pair_col = find_column(headers, ["poly_id", "pair_id"])
    a_col = find_column(headers, ["monoA", "smiles_A"])
    b_col = find_column(headers, ["monoB", "smiles_B"])
    pairs = {}
    for row in rows:
        pair = pair_key(row[pair_col])
        monomers = (row[a_col].strip(), row[b_col].strip())
        if pair in pairs and pairs[pair] != monomers:
            raise ValueError(f"Conflicting monomer identities for pair {pair}")
        pairs[pair] = monomers
    return pairs, labelled


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


def polymer_string(prefix, ab, ba):
    edges = [(1, 2, 1 - ba, 1 - ba), (3, 4, 1 - ab, 1 - ab)]
    edges += [(i, j, ab / 2, ba / 2) for i in (1, 2) for j in (3, 4)]
    rules = "".join(f"<{i}-{j}:{w12:.17g}:{w21:.17g}" for i, j, w12, w21 in edges)
    return f"{prefix}|0.5|0.5|{rules}"


def check_strata(records):
    strata = {}
    for record in records:
        key = (pair_key(record["pair_id"]), int(record["J"]))
        strata.setdefault(key, set()).add(record["wdmpnn_input"])
    differing = 0
    for (pair, junctions), inputs in strata.items():
        try:
            assert len(inputs) == 1, f"{len(inputs)} distinct wD-MPNN inputs"
        except AssertionError as error:
            logging.warning("pair_id=%s, J=%s: %s; fixed J does not imply identical directed inputs", pair, junctions, error)
            differing += 1
    logging.info("Input equality check: %d/%d groups identical; %d differ", len(strata) - differing, len(strata), differing)
    return differing


def main(dp=8, sequences_csv=SEQUENCES_CSV):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", type=Path, default=sequences_csv)
    parser.add_argument("--dataset", type=Path, default=DATASET_CSV)
    parser.add_argument("--ea-ip", type=Path, default=EA_IP_CSV)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / f"wdmpnn_inputs_dp{dp}.csv")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    for name in ("sequences", "dataset", "ea_ip"):
        if not getattr(args, name).is_file():
            parser.error(f"Missing --{name.replace('_', '-')} CSV: {getattr(args, name)}")
    if args.output.exists():
        parser.error(f"Refusing to overwrite existing output: {args.output}")
    pairs, labelled = monomer_lookup(args.dataset, args.ea_ip)
    headers, rows = read_csv(args.sequences)
    columns = {name: find_column(headers, aliases) for name, aliases in {
        "pair_id": ["pair_id", "poly_id", "monomer_pair_id"],
        "sequence": ["sequence", "sequence_string", "seq"],
        "J": ["J", "junction_count"],
        "true_EA": ["true_EA", "EA", "EA (eV)", "EA vs SHE (eV)"],
        "true_IP": ["true_IP", "IP", "IP (eV)", "IP vs SHE (eV)"],
    }.items()}
    records, failures = [], 0
    for line, row in enumerate(rows, 2):
        try:
            record = {key: row[value] for key, value in columns.items()}
            pair = pair_key(record["pair_id"])
            if pair not in pairs:
                raise ValueError(f"Pair {pair} missing from {args.dataset}")
            monomers = pairs[pair]
            if monomers not in labelled:
                raise ValueError(f"Pair {pair} monomer SMILES not matched exactly in {args.ea_ip}")
            ab, ba = transitions(record["sequence"], dp)
            junctions = int(record["J"])
            sequence = record["sequence"].strip().upper()
            measured = sum(a != b for a, b in zip(sequence, sequence[1:]))
            if junctions != measured:
                logging.warning("CSV row %d: provided J=%d differs from linear boundary count %d", line, junctions, measured)
            record["wdmpnn_input"] = polymer_string(labelled[monomers], ab, ba)
            records.append(record)
        except (ValueError, TypeError, KeyError) as error:
            logging.error("CSV row %d: %s", line, error)
            failures += 1
    if failures:
        raise ValueError(f"{failures} invalid rows logged; no output written")
    check_strata(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["pair_id", "sequence", "J", "wdmpnn_input", "true_EA", "true_IP"]
    with args.output.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    logging.info("Wrote %d analytic inputs to %s", len(records), args.output)


if __name__ == "__main__":
    main()
