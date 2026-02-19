#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bigsmiles_to_wdmpnn.py

Reads a CSV with columns:
  smiles_A, smiles_B, fracA, fracB, BigSMILES   (optional: Xn and others)
Writes the same CSV plus a new column:
  wdmpnn_input

Heuristics supported:
- Multiple { ... } blocks; we pick the two that best match smiles_A/smiles_B
- Alternatives inside a block separated by commas -> choose best match against target monomer
- Endpoints: first & last bonding tokens per chosen block -> [*:n], globally numbered
- Bond block: symmetric 0.25 weights; includes self, within, between pairs

Usage:
  python bigsmiles_to_wdmpnn.py input.csv output.csv
"""

import argparse
import csv
import re
import sys
from typing import List, Tuple, Dict, Optional

# Regexes
BLOCKS_RE = re.compile(r"\{([^{}]+)\}")                  # top-level {...} (no nesting assumed)
BOND_TOKEN_RE = re.compile(r"\[(?:>|<|\$)?\]")           # [], [>], [<], [$]
WHITESPACE_RE = re.compile(r"\s+")
LABEL_IN_BRACKETS_RE = re.compile(r"\[\*\s*:\s*(\d+)\]")  # only [*:n] to precollect existing labels

def _autolabel_psmiles(psmiles: str, start_label: int = 1) -> Tuple[str, List[int]]:
    """
    Auto-number wildcard sites in a PSMILES fragment.

    Recognized wildcard forms:
      - [*]         -> becomes [*:k]
      - [*:n]       -> preserved as [*:n]
      - * (bare)    -> becomes [*:k]
    Rules:
      - Left-to-right numbering for unlabeled wildcards.
      - If any [*:n] already exist, new labels continue from max(existing)+1.
      - '*' characters that occur inside non-wildcard bracket atoms (e.g. '[CH3*]')
        are left untouched (not treated as wildcard).

    Returns:
      labeled_psmiles, sorted_unique_labels
    """
    s = (psmiles or "").strip()
    if not s:
        raise ValueError("Empty PSMILES fragment for homopolymer.")

    # Pre-collect existing labels from explicit [*:n]
    existing = [int(x) for x in LABEL_IN_BRACKETS_RE.findall(s)]
    next_label = max(existing) + 1 if existing else start_label
    labels_seen: List[int] = []
    out = []
    i = 0
    n = len(s)

    while i < n:
        ch = s[i]
        if ch == '[':
            # find the matching ']'
            j = s.find(']', i + 1)
            if j == -1:
                # no closing ']' -> copy rest and stop
                out.append(s[i:])
                break

            inner = s[i+1:j].strip()  # contents inside [ ... ]
            # Wildcard bracket forms we accept: '*' or '*:n' (nothing else)
            if inner == '*' or inner.startswith('*:'):
                # parse label if present
                lab = None
                if inner.startswith('*:'):
                    numpart = inner[2:].strip()
                    if numpart.isdigit():
                        lab = int(numpart)

                if lab is None:
                    lab = next_label
                    next_label += 1

                out.append(f"[*:{lab}]")
                labels_seen.append(lab)
            else:
                # Not a pure wildcard atom; copy as-is
                out.append(s[i:j+1])

            i = j + 1
        elif ch == '*':
            # Bare wildcard atom outside brackets -> label it
            lab = next_label
            next_label += 1
            out.append(f"[*:{lab}]")
            labels_seen.append(lab)
            i += 1
        else:
            out.append(ch)
            i += 1

    unique_labels = sorted(set(labels_seen))
    if not unique_labels:
        raise ValueError("No wildcard sites found ('*', '[*]', or '[*:n]').")

    return "".join(out), unique_labels

def homopolymer_psmiles_to_wdmpnn(psmiles: str,
                                  frac: float = 1.0,
                                  xn: Optional[str] = None,
                                  weight: float = 0.5) -> str:
    """
    Convert a homopolymer PSMILES (single-monomer fragment with '*' or [*] sites)
    into a wDMPNN input string by auto-numbering the sites and constructing bonds.

    For homopolymers:
    - Fraction is always 1.0
    - Bond weight is 0.5 (not 0.25 like copolymers)
    - Only bond between the two wildcard sites (no self-pairs)
    - No Xn suffix by default
    
    Output:
      "<PSMILES_with_labels>|<frac>|<bonds>"
    Example:
      "CCC([*:1])C([*:2])CC|1.0|<1-2:0.5:0.5"
    """
    s = (psmiles or "").strip()
    if not s:
        raise ValueError("Empty PSMILES fragment for homopolymer.")
    if "|" in s or "<" in s or "~" in s:
        raise ValueError("Input appears to already contain fractions/bonds/Xn; pass only the monomer PSMILES.")

    labeled, labels = _autolabel_psmiles(s, start_label=1)
    
    # For homopolymers: only bond between the two wildcard sites (no self-pairs)
    if len(labels) != 2:
        raise ValueError(f"Homopolymer must have exactly 2 wildcard sites, found {len(labels)}")
    
    # Single bond between the two sites with weight 0.5
    bonds = f"<{labels[0]}-{labels[1]}:{weight}:{weight}"
    
    # No Xn suffix for homopolymers (kept simple)
    frac_str = f"|{float(frac)}|"
    return f"{labeled}{frac_str}{bonds}"


def _strip_tokens(s: str) -> str:
    """Remove bonding tokens and whitespace for crude similarity checks."""
    s = BOND_TOKEN_RE.sub("", s)
    s = WHITESPACE_RE.sub("", s)
    return s

def _simple_match_score(candidate: str, target: str) -> int:
    """
    Score how well 'candidate' matches 'target' (both token-stripped).
    Use length of the longest common substring (cheap O(n^2) dynamic approach avoided; we approximate by
    scanning target substrings presence). For speed & simplicity, use decreasing window scan.
    """
    cand = _strip_tokens(candidate)
    targ = _strip_tokens(target)
    n = len(targ)
    if n == 0 or len(cand) == 0:
        return 0
    # quick substring window search from longest to shortest
    for L in range(min(len(cand), n), 0, -1):
        for i in range(0, n - L + 1):
            if targ[i:i+L] in cand:
                return L
    return 0

def _split_alternatives(block_inner: str) -> List[str]:
    """
    Split a block 'inner' text by commas at top level (no nested braces here by assumption).
    If no comma, return [block_inner].
    """
    # BigSMILES often separates alternatives by commas directly inside a single brace block.
    parts = [p.strip() for p in block_inner.split(",")]
    return [p for p in parts if p != ""]

def _choose_two_positions(spans: List[Tuple[int,int]]) -> List[int]:
    """Pick FIRST and LAST token start positions; if only one, duplicate; if none, empty."""
    if not spans:
        return []
    if len(spans) == 1:
        return [spans[0][0], spans[0][0]]
    return [spans[0][0], spans[-1][0]]

def _block_to_psmiles(block_inner: str, next_label: int) -> Tuple[str, int, List[int]]:
    """
    Convert one chosen alternative (inner text) to a PSMILES fragment:
    - Replace first & last bonding tokens with [*:label], drop other tokens.
    Return (fragment, updated_next_label, labels_used_in_fragment)
    """
    spans = [(m.start(), m.end()) for m in BOND_TOKEN_RE.finditer(block_inner)]
    chosen_starts = _choose_two_positions(spans)

    out = []
    i = 0
    labels_used: List[int] = []
    labels_to_place: List[int] = []

    if len(chosen_starts) >= 2:
        labels_to_place = [next_label, next_label + 1]
    elif len(chosen_starts) == 1:
        labels_to_place = [next_label, next_label]   # degenerate: both ends same site
    else:
        labels_to_place = []

    label_idx = 0
    spans_iter = iter(spans)
    current = next(spans_iter, None)

    while i < len(block_inner):
        if current and i == current[0]:
            if i in chosen_starts and label_idx < len(labels_to_place):
                lab = labels_to_place[label_idx]
                out.append(f"[*:{lab}]")
                if not labels_used or labels_used[-1] != lab:
                    labels_used.append(lab)
                label_idx += 1
            # drop non-chosen tokens
            i = current[1]
            current = next(spans_iter, None)
        else:
            out.append(block_inner[i])
            i += 1

    # advance label counter by unique assignments
    unique_assigned = [lab for k, lab in enumerate(labels_to_place) if k == 0 or lab != labels_to_place[k-1]]
    next_label += len(unique_assigned)

    frag = "".join(out).strip()
    return frag, next_label, labels_used

def _select_block_alternative(block_inner: str, target_smiles_hint: Optional[str]) -> str:
    """
    If the block has comma-separated alternatives, pick the one that best matches target_smiles_hint.
    If hint is None or tie, pick the first.
    """
    alts = _split_alternatives(block_inner)
    if len(alts) == 1 or not target_smiles_hint:
        return alts[0]
    # choose best by simple match score
    best = alts[0]
    best_score = _simple_match_score(best, target_smiles_hint)
    for alt in alts[1:]:
        sc = _simple_match_score(alt, target_smiles_hint)
        if sc > best_score:
            best, best_score = alt, sc
    return best

def _pick_two_blocks(big: str, hintA: Optional[str], hintB: Optional[str]) -> Tuple[str, str]:
    """
    From all top-level blocks, pick two to represent monomer A and B.
    Strategy:
      - If exactly two blocks exist, use them (A=first, B=second) but still resolve alternatives by hints.
      - If >2 blocks, pick the two with the best individual match to hintA and hintB respectively.
        (Greedy: best for A gets assigned to A; then remove it and pick best for B from remaining.)
      - If <2, raise.
    """
    blocks = BLOCKS_RE.findall(big)
    if len(blocks) < 2:
        raise ValueError("Found fewer than 2 BigSMILES blocks")

    if len(blocks) == 2:
        A = _select_block_alternative(blocks[0], hintA)
        B = _select_block_alternative(blocks[1], hintB)
        return A, B

    # >2 blocks: score each against each hint, pick greedily
    scoredA = [(blk, _simple_match_score(blk, hintA or "")) for blk in blocks]
    scoredA.sort(key=lambda x: x[1], reverse=True)
    pickA = scoredA[0][0]
    remaining = [blk for blk in blocks if blk is not pickA]
    scoredB = [(blk, _simple_match_score(blk, hintB or "")) for blk in remaining]
    scoredB.sort(key=lambda x: x[1], reverse=True)
    pickB = scoredB[0][0]
    # resolve alternatives inside each pick
    return _select_block_alternative(pickA, hintA), _select_block_alternative(pickB, hintB)

def _labels_pairs(labels_A: List[int], labels_B: List[int]) -> List[Tuple[int,int]]:
    """All pairs: self, within A, within B, and between (canonicalized)."""
    pairs = []
    # self
    for k in labels_A + labels_B:
        pairs.append((k, k))
    # within A
    la = sorted(labels_A)
    for i in range(len(la)):
        for j in range(i+1, len(la)):
            pairs.append((la[i], la[j]))
    # within B
    lb = sorted(labels_B)
    for i in range(len(lb)):
        for j in range(i+1, len(lb)):
            pairs.append((lb[i], lb[j]))
    # between
    for i in labels_A:
        for j in labels_B:
            if i <= j:
                pairs.append((i, j))
            else:
                pairs.append((j, i))
    # dedupe & sort
    pairs = sorted(set(pairs))
    return pairs

def _bonds_block(labels_A: List[int], labels_B: List[int], w: float = 0.25) -> str:
    return "".join(f"<{i}-{j}:{w}:{w}" for (i, j) in _labels_pairs(labels_A, labels_B))

def bigsmiles_to_psmiles_pair(big: str, smiles_A_hint: Optional[str], smiles_B_hint: Optional[str]) -> Tuple[str, List[int], str, List[int]]:
    """
    Produce PSMILES fragments for A and B and the star label lists used in each.
    """
    blockA_text, blockB_text = _pick_two_blocks(big, smiles_A_hint, smiles_B_hint)

    next_label = 1
    fragA, next_label, labels_A = _block_to_psmiles(blockA_text, next_label)
    fragB, next_label, labels_B = _block_to_psmiles(blockB_text, next_label)

    if len(labels_A) == 0 or len(labels_B) == 0:
        raise ValueError("At least one block has no bonding tokens to place endpoints.")

    return fragA, labels_A, fragB, labels_B

def make_wdmpnn_input(big: str, fracA: float, fracB: float,
                      smiles_A_hint: Optional[str], smiles_B_hint: Optional[str],
                      xn: Optional[str]) -> str:
    fragA, labels_A, fragB, labels_B = bigsmiles_to_psmiles_pair(big, smiles_A_hint, smiles_B_hint)
    monomers = f"{fragA}.{fragB}"
    fractions = f"|{fracA}|{fracB}|"
    bonds = _bonds_block(labels_A, labels_B, w=0.25)
    dp = f"~{xn.strip()}" if (xn is not None and str(xn).strip() != "") else ""
    return f"{monomers}{fractions}{bonds}{dp}"

def _to_float(x, field):
    try:
        return float(str(x).strip())
    except Exception:
        raise ValueError(f"Invalid number in {field}: {x}")

def process_row_copolymer(row: Dict[str, str]) -> Dict[str, str]:
    """Process a row for copolymer (requires BigSMILES, fracA, fracB)."""
    big = (row.get("BigSMILES") or "").strip()
    if not big:
        raise ValueError("Missing BigSMILES")
    fracA = _to_float(row.get("fracA"), "fracA")
    fracB = _to_float(row.get("fracB"), "fracB")
    smiles_A_hint = (row.get("smiles_A") or "").strip() or None
    smiles_B_hint = (row.get("smiles_B") or "").strip() or None
    xn = row.get("Xn")  # optional

    wdmpnn = make_wdmpnn_input(big, fracA, fracB, smiles_A_hint, smiles_B_hint, xn)
    out = dict(row)
    out["wdmpnn_input"] = wdmpnn
    return out


def process_row_homopolymer(row: Dict[str, str]) -> Dict[str, str]:
    """Process a row for homopolymer (requires PSMILES or smiles column)."""
    # Try both column names
    smiles = (row.get("psmiles") or row.get("PSMILES") or
              row.get("smiles") or row.get("SMILES") or "").strip()
    if not smiles:
        raise ValueError("Missing psmiles/PSMILES or smiles/SMILES column")

    # Get fraction (default to 1.0 if not provided)
    frac_str = row.get("frac") or row.get("fraction") or "1.0"
    frac = _to_float(frac_str, "frac")

    # Get degree of polymerization (Xn) — accept either "Xn" or "DoP"
    xn = row.get("Xn") or row.get("DoP")

    # Get weight (optional, default 0.25)
    weight_str = row.get("weight") or "0.25"
    try:
        weight = float(weight_str)
    except Exception:
        weight = 0.25

    # Build the wDMPNN input string
    wdmpnn = homopolymer_psmiles_to_wdmpnn(smiles, frac, xn, weight)

    out = dict(row)
    out["wdmpnn_input"] = wdmpnn
    return out





def main():
    parser = argparse.ArgumentParser(
        description='Convert BigSMILES (copolymer) or PSMILES (homopolymer) to wDMPNN input format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\nExamples:
  # Copolymer mode (requires: BigSMILES, fracA, fracB, optional: smiles_A, smiles_B, Xn)
  python make_wdmpnn_input.py input.csv output.csv --mode copolymer
  
  # Homopolymer mode (requires: PSMILES/psmiles, optional: frac, Xn, weight)
  python make_wdmpnn_input.py input.csv output.csv --mode homopolymer
"""
    )
    parser.add_argument('input_csv', help='Input CSV file')
    parser.add_argument('output_csv', help='Output CSV file')
    parser.add_argument('--mode', choices=['copolymer', 'homopolymer'], default='copolymer',
                       help='Processing mode: copolymer (default) or homopolymer')
    
    args = parser.parse_args()
    
    # Select processing function based on mode
    if args.mode == 'homopolymer':
        process_func = process_row_homopolymer
        print(f"Processing in HOMOPOLYMER mode...")
    else:
        process_func = process_row_copolymer
        print(f"Processing in COPOLYMER mode...")
    
    # Read input CSV
    with open(args.input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        print("Warning: Input CSV is empty")
        sys.exit(1)
    
    print(f"Processing {len(rows)} rows...")
    
    # Process rows
    out_rows = []
    errors = 0
    for idx, r in enumerate(rows, 1):
        try:
            out_rows.append(process_func(r))
        except Exception as e:
            errors += 1
            rr = dict(r)
            rr["wdmpnn_input"] = f"ERROR: {e}"
            out_rows.append(rr)
            print(f"  Row {idx}: ERROR - {e}")
    
    # Write output CSV
    fieldnames = list(rows[0].keys()) + ["wdmpnn_input"]
    with open(args.output_csv, "w", newline="", encoding="utf-8") as g:
        w = csv.DictWriter(g, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    
    print(f"\nCompleted: {len(rows) - errors} successful, {errors} errors")
    print(f"Output written to: {args.output_csv}")

if __name__ == "__main__":
    main()
