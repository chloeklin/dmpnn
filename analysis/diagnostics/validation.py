"""Step 1: Validate evaluation inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MODELS, SPLITS, TARGETS, N_FOLDS, STEP_DIRS, MODEL_DISPLAY
from .data_loading import load_predictions_single
from .grouping import build_fold_df, filter_matched_groups


def run_validation(df: pd.DataFrame, meta: dict[str, list]) -> pd.DataFrame:
    """
    Validate all prediction inputs and produce inventory table.

    Returns the inventory DataFrame and writes:
      - 01_validation/evaluation_inventory.csv
      - 01_validation/evaluation_validation_report.md
    """
    out_dir = STEP_DIRS['01_validation']
    rows = []
    issues = []

    for split in SPLITS:
        meta_folds = meta[split]
        for tkey in TARGETS:
            for fold in range(N_FOLDS[split]):
                # Track test indices across models for cross-check
                model_indices = {}
                for model in MODELS:
                    pred = load_predictions_single(model, tkey, split, fold, meta_folds)
                    row = {
                        'split': split,
                        'target': tkey,
                        'fold': fold,
                        'model': model,
                    }
                    if pred is None:
                        row.update({
                            'n_test': 0, 'n_matched_samples': 0,
                            'n_matched_groups': 0,
                            'y_true_mean': np.nan, 'y_true_std': np.nan,
                            'y_pred_mean': np.nan, 'y_pred_std': np.nan,
                            'y_true_min': np.nan, 'y_true_max': np.nan,
                            'y_pred_min': np.nan, 'y_pred_max': np.nan,
                            'prediction_scale': 'MISSING',
                            'validation_passed': False,
                        })
                        issues.append(
                            f"MISSING: {model}/{tkey}/{split}/fold{fold}"
                        )
                        rows.append(row)
                        continue

                    yt = pred['y_true']
                    yp = pred['y_pred']
                    gidx = pred['global_idx']
                    model_indices[model] = set(gidx.tolist())

                    # Check 1: lengths match
                    len_ok = (len(yt) == len(yp) == len(gidx))

                    # Check 2: scale check (physical units eV)
                    # EA is negative: -5 to 2 eV. IP is positive: 0 to ~4 eV.
                    # Use a single range covering both: -5.0 to 4.5 eV for mean.
                    # The dataset IP max is ~3.98 eV; some folds (e.g. fold 2,
                    # difluorobenzene monomers) have fold-mean > 2.0 eV and are
                    # still perfectly valid physical predictions.
                    yt_mean = float(yt.mean())
                    yt_std = float(yt.std())
                    scale_ok = (-5.0 < yt_mean < 4.5) and (0.05 < yt_std < 3.0)
                    scale_str = 'eV' if scale_ok else 'SUSPECT'

                    # Check 3: indices in bounds
                    idx_ok = (gidx.min() >= 0 and gidx.max() < len(df))

                    # Check 4: architecture labels present
                    if idx_ok:
                        test_rows = df.iloc[gidx]
                        arch_present = 'poly_type' in test_rows.columns
                        arch_ok = arch_present and test_rows['poly_type'].notna().all()
                    else:
                        arch_ok = False

                    # Check 5: no duplicate indices
                    dup_ok = (len(set(gidx.tolist())) == len(gidx))

                    # Matched groups
                    if idx_ok:
                        fdf = build_fold_df(df, yt, yp, gidx)
                        matched = filter_matched_groups(fdf)
                        n_matched_samples = len(matched)
                        n_matched_groups = matched['group_key'].nunique()
                    else:
                        n_matched_samples = 0
                        n_matched_groups = 0

                    # Monomer-heldout specific checks
                    monomer_ok = True
                    if split == 'monomer_heldout' and idx_ok:
                        fold_meta = next(
                            (r for r in meta_folds if r['fold'] == fold), None
                        )
                        if fold_meta and 'held_out_monomer_A' in fold_meta:
                            hom = fold_meta['held_out_monomer_A']
                            test_A = test_rows['smiles_A'].astype(str).values
                            test_B = test_rows['smiles_B'].astype(str).values
                            has_hom = np.array(
                                [(a == hom or b == hom) for a, b in zip(test_A, test_B)]
                            )
                            monomer_ok = has_hom.all()
                            if not monomer_ok:
                                issues.append(
                                    f"MONOMER_LEAK: {model}/{tkey}/{split}/fold{fold} "
                                    f"- not all test have held-out monomer"
                                )

                    passed = all([len_ok, scale_ok, idx_ok, arch_ok, dup_ok, monomer_ok])

                    row.update({
                        'n_test': len(yt),
                        'n_matched_samples': n_matched_samples,
                        'n_matched_groups': n_matched_groups,
                        'y_true_mean': yt_mean,
                        'y_true_std': yt_std,
                        'y_pred_mean': float(yp.mean()),
                        'y_pred_std': float(yp.std()),
                        'y_true_min': float(yt.min()),
                        'y_true_max': float(yt.max()),
                        'y_pred_min': float(yp.min()),
                        'y_pred_max': float(yp.max()),
                        'prediction_scale': scale_str,
                        'validation_passed': passed,
                    })
                    if not passed:
                        issues.append(
                            f"FAILED: {model}/{tkey}/{split}/fold{fold} "
                            f"len={len_ok} scale={scale_ok} idx={idx_ok} "
                            f"arch={arch_ok} dup={dup_ok} monomer={monomer_ok}"
                        )
                    rows.append(row)

                # Cross-check: all models same test rows within fold
                if len(model_indices) > 1:
                    ref_model = list(model_indices.keys())[0]
                    ref_idx = model_indices[ref_model]
                    for m, idx_set in model_indices.items():
                        if idx_set != ref_idx:
                            issues.append(
                                f"INDEX_MISMATCH: {split}/{tkey}/fold{fold} "
                                f"{ref_model} vs {m}"
                            )

    inv_df = pd.DataFrame(rows)
    inv_df.to_csv(out_dir / 'evaluation_inventory.csv', index=False)

    # Write report
    n_total = len(inv_df)
    n_passed = inv_df['validation_passed'].sum()
    n_missing = (inv_df['n_test'] == 0).sum()

    lines = [
        "# Evaluation Validation Report\n",
        f"- Total combinations: {n_total}",
        f"- Passed: {n_passed}",
        f"- Missing predictions: {n_missing}",
        f"- Failed validations: {n_total - n_passed - n_missing}",
        "",
    ]
    if issues:
        lines.append("## Issues\n")
        for iss in issues:
            lines.append(f"- {iss}")
    else:
        lines.append("## All validations passed ✓")

    lines.append("\n## Inventory Summary\n")
    for split in SPLITS:
        sub = inv_df[inv_df['split'] == split]
        avail = sub[sub['n_test'] > 0]
        lines.append(f"### {split}")
        lines.append(f"- Available: {len(avail)}/{len(sub)} model/target/fold combos")
        if len(avail) > 0:
            lines.append(f"- Mean n_test: {avail['n_test'].mean():.0f}")
            lines.append(f"- Mean n_matched_groups: {avail['n_matched_groups'].mean():.0f}")
        lines.append("")

    (out_dir / 'evaluation_validation_report.md').write_text('\n'.join(lines))
    print(f"  Step 1 complete: {out_dir}")
    return inv_df
