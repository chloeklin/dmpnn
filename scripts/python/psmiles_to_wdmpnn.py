#!/usr/bin/env python3
"""
Convert PSMILES column to wDMPNN format for homopolymers.

Reads a CSV file, finds the psmiles column (case-insensitive), optionally uses DoP column,
and adds a new WDMPNN_Input column with the format:
    <PSMILES_with_labels>|1.0|<1-2:0.5:0.5

Usage:
    python psmiles_to_wdmpnn.py htpmd.csv htpmd_with_wdmpnn.csv
    python psmiles_to_wdmpnn.py htpmd.csv htpmd_with_wdmpnn.csv --psmiles-col custom_psmiles
    
Note: Paths are relative to the repository's data/ directory
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional

# Regexes
LABEL_IN_BRACKETS_RE = re.compile(r"\[\*\s*:\s*(\d+)\]")

def _autolabel_psmiles(psmiles: str, start_label: int = 1) -> Tuple[str, List[int]]:
    """
    Auto-number wildcard sites in a PSMILES fragment.
    
    Recognized wildcard forms:
      - [*]         -> becomes [*:k]
      - [*:n]       -> preserved as [*:n]
      - * (bare)    -> becomes [*:k]
    
    Returns:
      (labeled_psmiles, sorted_unique_labels)
    """
    s = (psmiles or "").strip()
    if not s:
        raise ValueError("Empty PSMILES fragment")
    
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
            # Find matching ']'
            j = s.find(']', i + 1)
            if j == -1:
                out.append(s[i:])
                break
            
            inner = s[i+1:j].strip()
            # Wildcard bracket forms: '*' or '*:n'
            if inner == '*' or inner.startswith('*:'):
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
                # Not a pure wildcard; copy as-is
                out.append(s[i:j+1])
            
            i = j + 1
        elif ch == '*':
            # Bare wildcard outside brackets
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
        raise ValueError("No wildcard sites found ('*', '[*]', or '[*:n]')")
    
    return "".join(out), unique_labels


def psmiles_to_wdmpnn(psmiles: str, dop: Optional[str] = None, weight: float = 0.5) -> str:
    """
    Convert a homopolymer PSMILES to wDMPNN input format.
    
    Format: <PSMILES_with_labels>|1.0|<1-2:0.5:0.5~DoP
    
    Args:
        psmiles: PSMILES string with wildcards (* or [*])
        dop: Optional degree of polymerization (appended as ~DoP if provided)
        weight: Bond weight (default 0.5 for homopolymers)
    
    Returns:
        wDMPNN input string
    
    Examples:
        Without DoP: "[*:1]CC[*:2]|1.0|<1-2:0.5:0.5"
        With DoP:    "[*:1]CC[*:2]|1.0|<1-2:0.5:0.5~50"
    """
    s = (psmiles or "").strip()
    if not s:
        raise ValueError("Empty PSMILES")
    
    # Check if already in wDMPNN format
    if "|" in s and "<" in s:
        return s  # Already formatted
    
    # Auto-label wildcards
    labeled, labels = _autolabel_psmiles(s, start_label=1)
    
    # Validate exactly 2 wildcard sites for homopolymer
    if len(labels) != 2:
        raise ValueError(f"Homopolymer must have exactly 2 wildcard sites, found {len(labels)}")
    
    # Create wDMPNN format: <psmiles>|1.0|<1-2:0.5:0.5
    frac = 1.0
    bonds = f"<{labels[0]}-{labels[1]}:{weight}:{weight}"
    
    # Add DoP suffix if provided
    dop_suffix = ""
    if dop is not None:
        dop_str = str(dop).strip()
        if dop_str and dop_str.lower() != 'nan':
            dop_suffix = f"~{dop_str}"
    
    return f"{labeled}|{frac}|{bonds}{dop_suffix}"


def find_column(headers: List[str], possible_names: List[str]) -> Optional[str]:
    """Find a column by checking multiple possible names (case-insensitive)."""
    headers_lower = {h.lower(): h for h in headers}
    for name in possible_names:
        if name.lower() in headers_lower:
            return headers_lower[name.lower()]
    return None


def resolve_path(path_str: str) -> Path:
    """
    Resolve a path relative to the repository root's data/ directory.
    
    If path is absolute, use as-is.
    If path starts with 'data/', use relative to repo root.
    Otherwise, assume it's a filename in data/ directory.
    """
    path = Path(path_str)
    
    # If absolute path, use as-is
    if path.is_absolute():
        return path
    
    # Find repository root (look for pyproject.toml or .git)
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent  # scripts/python -> scripts -> repo_root
    
    # If path starts with 'data/', resolve from repo root
    if str(path).startswith('data/'):
        return repo_root / path
    
    # Otherwise, assume it's a filename in data/ directory
    return repo_root / 'data' / path


def main():
    parser = argparse.ArgumentParser(
        description='Convert PSMILES column to wDMPNN format for homopolymers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect psmiles and DoP columns
  python psmiles_to_wdmpnn.py input.csv output.csv
  
  # Specify custom column names
  python psmiles_to_wdmpnn.py input.csv output.csv --psmiles-col my_psmiles --dop-col my_dop
  
  # Specify output column name
  python psmiles_to_wdmpnn.py input.csv output.csv --output-col custom_wdmpnn
"""
    )
    parser.add_argument('input_csv', help='Input CSV file')
    parser.add_argument('output_csv', help='Output CSV file')
    parser.add_argument('--psmiles-col', default=None,
                       help='Name of PSMILES column (auto-detected if not specified)')
    parser.add_argument('--dop-col', default=None,
                       help='Name of DoP column (auto-detected if not specified)')
    parser.add_argument('--output-col', default='WDMPNN_Input',
                       help='Name of output column (default: WDMPNN_Input)')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip rows that already have wDMPNN format in psmiles column')
    
    args = parser.parse_args()
    
    # Resolve paths relative to data/ directory
    input_path = resolve_path(args.input_csv)
    output_path = resolve_path(args.output_csv)
    
    # Read input CSV
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        headers = reader.fieldnames
    
    if not rows:
        print("ERROR: Input CSV is empty")
        sys.exit(1)
    
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Read {len(rows)} rows")
    print(f"Columns: {', '.join(headers)}")
    print()
    
    # Find PSMILES column
    if args.psmiles_col:
        psmiles_col = args.psmiles_col
        if psmiles_col not in headers:
            print(f"ERROR: Specified PSMILES column '{psmiles_col}' not found")
            sys.exit(1)
    else:
        psmiles_col = find_column(headers, ['psmiles', 'PSMILES', 'smiles', 'SMILES'])
        if not psmiles_col:
            print("ERROR: Could not find PSMILES column. Available columns:", headers)
            print("Please specify with --psmiles-col")
            sys.exit(1)
    
    print(f"✓ Using PSMILES column: '{psmiles_col}'")
    
    # Find DoP column (optional)
    if args.dop_col:
        dop_col = args.dop_col if args.dop_col in headers else None
        if not dop_col:
            print(f"WARNING: Specified DoP column '{args.dop_col}' not found, ignoring")
    else:
        dop_col = find_column(headers, ['DoP', 'dop', 'Xn', 'xn', 'degree_of_polymerization'])
    
    if dop_col:
        print(f"✓ Using DoP column: '{dop_col}'")
    else:
        print("  No DoP column found (optional)")
    
    print()
    
    # Process rows
    output_rows = []
    success_count = 0
    error_count = 0
    skip_count = 0
    
    for idx, row in enumerate(rows, 1):
        psmiles = row.get(psmiles_col, "").strip()
        dop = row.get(dop_col, "") if dop_col else None
        
        try:
            # Skip if already in wDMPNN format and skip_existing is set
            if args.skip_existing and "|" in psmiles and "<" in psmiles:
                row[args.output_col] = psmiles
                skip_count += 1
            else:
                wdmpnn = psmiles_to_wdmpnn(psmiles, dop)
                row[args.output_col] = wdmpnn
                success_count += 1
        except Exception as e:
            row[args.output_col] = f"ERROR: {e}"
            error_count += 1
            print(f"  Row {idx}: ERROR - {e}")
            print(f"    PSMILES: {psmiles}")
        
        output_rows.append(row)
    
    # Add output column to headers if not present
    output_headers = list(headers)
    if args.output_col not in output_headers:
        output_headers.append(args.output_col)
    
    # Write output CSV
    with open(output_path, "w", newline="", encoding="utf-8") as g:
        writer = csv.DictWriter(g, fieldnames=output_headers)
        writer.writeheader()
        writer.writerows(output_rows)
    
    print()
    print("=" * 60)
    print(f"✓ Completed: {success_count} successful, {error_count} errors, {skip_count} skipped")
    print(f"✓ Output written to: {args.output_csv}")
    print("=" * 60)
    
    # Show sample outputs
    if success_count > 0:
        print("\nSample wDMPNN outputs:")
        shown = 0
        for row in output_rows:
            if not row[args.output_col].startswith("ERROR"):
                print(f"  {row[args.output_col]}")
                shown += 1
                if shown >= 3:
                    break


if __name__ == "__main__":
    main()
