#!/usr/bin/env python3
"""Build a Markdown comparison table for HPG-2Stage variants vs baselines.

Read-only analysis script — reads aggregate CSVs, writes a single
Markdown file.  No model code is imported or modified.

Usage:
    python scripts/python/build_hpg2stage_comparison_table.py

Output:
    results/hpg2stage_comparison_table.md
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
OUTPUT_PATH = RESULTS_DIR / "hpg2stage_comparison_table.md"

TARGETS = ["EA vs SHE (eV)", "IP vs SHE (eV)"]

# ── Model registry: display_name → (csv_path_per_target_or_single, is_per_target) ─
# For per-target files, {target} is replaced at runtime.
# For single files (wDMPNN), data is filtered by the target column.
MODELS = {
    "HPG_frac": {
        "dir": "HPG",
        "pattern": "ea_ip__hpg_frac__a_held_out__target_{target}_results.csv",
        "per_target": True,
    },
    "wDMPNN": {
        "dir": "wDMPNN",
        "pattern": "ea_ip__a_held_out_results.csv",
        "per_target": False,
    },
    "HPG-2Stage-Frac": {
        "dir": "DMPNN",
        "pattern": "ea_ip__copoly_mix_meta__poly_type__a_held_out__target_{target}_results.csv",
        "per_target": True,
    },
    "HPG-2Stage-Interact-fixed": {
        "dir": "DMPNN",
        # New code adds fusion suffix; fall back to old file without it.
        "pattern": "ea_ip__copoly_mix_pair_meta__fusion_sum_fusion__poly_type__a_held_out__target_{target}_results.csv",
        "fallback": "ea_ip__copoly_mix_pair_meta__poly_type__a_held_out__target_{target}_results.csv",
        "per_target": True,
    },
    "HPG-2Stage-Interact-learned": {
        "dir": "DMPNN",
        "pattern": "ea_ip__copoly_mix_pair_meta__fusion_scalar_residual_fusion__poly_type__a_held_out__target_{target}_results.csv",
        "per_target": True,
    },
}


def load_target_df(spec: dict, target: str) -> pd.DataFrame | None:
    """Return a DataFrame for a single target, or None if the file is missing."""
    base = RESULTS_DIR / spec["dir"]

    if spec["per_target"]:
        path = base / spec["pattern"].format(target=target)
        if not path.exists() and "fallback" in spec:
            path = base / spec["fallback"].format(target=target)
        if not path.exists():
            return None
        df = pd.read_csv(path)
    else:
        path = base / spec["pattern"]
        if not path.exists():
            return None
        df = pd.read_csv(path)
        df = df[df["target"] == target]

    if df.empty:
        return None
    return df


def fmt(mean: float, std: float, decimals: int = 4) -> str:
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"


def main() -> None:
    rows: list[dict] = []
    missing: list[str] = []

    for name, spec in MODELS.items():
        row = {"Model": name}
        for target in TARGETS:
            short = "EA" if "EA" in target else "IP"
            df = load_target_df(spec, target)
            if df is None:
                missing.append(f"{name} / {target}")
                row[f"{short} RMSE"] = "—"
                row[f"{short} R²"] = "—"
                continue
            rmse_mean = df["test/rmse"].mean()
            rmse_std = df["test/rmse"].std()
            r2_mean = df["test/r2"].mean()
            r2_std = df["test/r2"].std()
            row[f"{short} RMSE"] = fmt(rmse_mean, rmse_std)
            row[f"{short} R²"] = fmt(r2_mean, r2_std)
        rows.append(row)

    # ── Build Markdown ───────────────────────────────────────────────
    cols = ["Model", "EA RMSE", "EA R²", "IP RMSE", "IP R²"]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [
        "# HPG-2Stage Comparison — EA/IP, monomer-disjoint split",
        "",
        header,
        sep,
    ]
    for row in rows:
        line = "| " + " | ".join(str(row.get(c, "—")) for c in cols) + " |"
        lines.append(line)

    if missing:
        lines += [
            "",
            "**Pending results (CSVs not yet found):**",
            "",
        ]
        for m in missing:
            lines.append(f"- {m}")

    lines.append("")  # trailing newline
    text = "\n".join(lines)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(text)
    print(text)
    print(f"\nWritten to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
