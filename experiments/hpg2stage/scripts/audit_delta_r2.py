#!/usr/bin/env python3
"""Standalone ΔR² audit for Stage2D paper outputs.

Verifies that architecture-deviation metrics are computed using predicted
within-test-group means, and reports diagnostics for every model/split/target.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / 'experiments' / 'hpg2stage' / 'scripts'))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts' / 'python'))

import generate_stage2d_paper_outputs as g2d


def build_raw_group_key(df_sub):
    """Build group key from raw smiles_A, smiles_B, fracA, fracB."""
    a = df_sub['smiles_A'].astype(str).values
    b = df_sub['smiles_B'].astype(str).values
    fa = df_sub['fracA'].astype(float).values
    fb = df_sub['fracB'].astype(float).values
    return np.array([f"{x}||{y}||{fx:.6f}||{fy:.6f}" for x, y, fx, fy in zip(a, b, fa, fb)])


def inspect_frac_group_spread(preds, df, target_key, use_raw_group=False):
    """Find the Frac group with the largest raw prediction spread and print details."""
    max_spread = -1.0
    best_group = None
    best_gdf = None

    for p in preds:
        if p['indices'] is None:
            continue
        valid = p['indices'] >= 0
        yt = np.asarray(p['y_true'])[valid]
        yp = np.asarray(p['y_pred'])[valid]
        indices = np.asarray(p['indices'])[valid]
        df_sub = df.iloc[indices].copy()
        arch = df_sub['poly_type'].values

        if use_raw_group:
            groups = build_raw_group_key(df_sub)
            group_label = "raw_group_key"
        else:
            groups = df_sub['group_key'].values
            group_label = "canonical_group_key"

        gdf = pd.DataFrame({
            'y_true': yt,
            'y_pred': yp,
            'group': groups,
            'arch': arch,
            'index': indices,
            'smilesA': df_sub['smilesA'].values,
            'smilesB': df_sub['smilesB'].values,
            'raw_smilesA': df_sub['smiles_A'].values,
            'raw_smilesB': df_sub['smiles_B'].values,
            'fracA': df_sub['fracA'].values,
            'fracB': df_sub['fracB'].values,
            'wdmpnn_input': df_sub['WDMPNN_Input'].values if 'WDMPNN_Input' in df_sub.columns else None,
        })
        ga = gdf.groupby('group')['arch'].nunique()
        multi = ga[ga >= 2].index
        gdf_m = gdf[gdf['group'].isin(multi)]
        if len(gdf_m) == 0:
            continue
        spread = gdf_m.groupby('group')['y_pred'].apply(lambda x: x.max() - x.min())
        if spread.max() > max_spread:
            max_spread = spread.max()
            best_group = spread.idxmax()
            best_gdf = gdf_m[gdf_m['group'] == best_group].copy()

    if best_gdf is None or len(best_gdf) == 0:
        print("  No multi-architecture group found.")
        return None

    print(f"\n  Worst-case Frac group for {target_key} ({group_label}): spread={max_spread:.6f} eV")
    print(f"  group_key = {best_group}")
    print(f"  canonical smilesA = {best_gdf['smilesA'].iloc[0]}")
    print(f"  canonical smilesB = {best_gdf['smilesB'].iloc[0]}")
    print(f"  canonical fracA   = {best_gdf['fracA'].iloc[0]:.6f}")
    print(f"  canonical fracB   = {best_gdf['fracB'].iloc[0]:.6f}")
    print("  Rows:")
    for _, row in best_gdf.sort_values('arch').iterrows():
        print(f"    arch={row['arch']:12s} idx={row['index']:6d} y_true={row['y_true']:8.4f} y_pred={row['y_pred']:8.4f}")
        print(f"      raw_smiles_A = {row['raw_smilesA']}")
        print(f"      raw_smiles_B = {row['raw_smilesB']}")
        if row['wdmpnn_input'] is not None:
            print(f"      WDMPNN_Input = {row['wdmpnn_input']}")
    return best_gdf


def verify_test_ids(preds, df, target_key):
    """Check that test_ids point to rows whose true target matches saved y_true."""
    target_col = g2d.TARGETS[target_key]
    mismatches = 0
    total = 0
    for fold, p in enumerate(preds):
        if p['indices'] is None:
            continue
        valid = p['indices'] >= 0
        indices = p['indices'][valid]
        yt_saved = p['y_true'][valid]
        yt_df = df.iloc[indices][target_col].values.astype(float)
        bad = np.abs(yt_saved - yt_df) > 1e-5
        mismatches += bad.sum()
        total += len(yt_saved)
        if bad.any():
            print(f"  [test_id mismatch] fold {fold}: {bad.sum()} / {len(yt_saved)} mismatches")
    print(f"  [test_id check] {target_key}: {mismatches}/{total} mismatches across all folds")
    return mismatches == 0


def build_value_to_index_map(df):
    """Build a dict mapping (EA, IP) value pairs to original CSV index."""
    ea_vals = df[g2d.TARGETS['EA']].values.astype(float)
    ip_vals = df[g2d.TARGETS['IP']].values.astype(float)
    mapping = {}
    for idx, (ea, ip) in enumerate(zip(ea_vals, ip_vals)):
        key = (round(float(ea), 6), round(float(ip), 6))
        if key in mapping:
            print(f"  [WARNING] duplicate (EA, IP) pair at indices {mapping[key]} and {idx}")
        mapping[key] = idx
    return mapping


def correct_indices_with_both_targets(preds_ea, preds_ip, df):
    """Remap predictions to original CSV indices using both EA and IP y_true values."""
    value_map = build_value_to_index_map(df)
    corrected = []
    for fold, (pea, pip) in enumerate(zip(preds_ea, preds_ip)):
        n = len(pea['y_true'])
        assert len(pip['y_true']) == n, f"fold {fold}: EA/IP length mismatch"
        new_indices = np.full(n, -1, dtype=int)
        for j in range(n):
            ea = round(float(pea['y_true'][j]), 6)
            ip = round(float(pip['y_true'][j]), 6)
            key = (ea, ip)
            if key in value_map:
                new_indices[j] = value_map[key]
            else:
                print(f"  [WARNING] fold {fold} row {j}: no matching (EA, IP) pair for ({ea}, {ip})")
        corrected.append({
            'y_true': pea['y_true'].copy(),
            'y_pred': pea['y_pred'].copy(),
            'indices': new_indices,
            'fold': pea['fold'],
        })
    return corrected


def run_audit():
    df = g2d.load_dataset()

    # Set normalization params (required by HPG2Stage loaders)
    g2d._NORM_PARAMS = g2d.estimate_normalization_params()

    models = [
        ('Frac', 'frac', False),
        ('wDMPNN', None, True),
        ('2D0-arch', '2d0_arch', False),
        ('2D1-arch', '2d1_arch', False),
    ]
    splits = ['a_held_out', 'group_disjoint', 'pair_disjoint']

    print("=" * 70)
    print("DELTA-R² AUDIT")
    print("=" * 70)
    print("Group means are computed from predictions within the test set only.")
    print("Metrics: R2_mean, ArchR2_mean, true Δ mean/std, pred Δ mean/std.")
    print("For Frac, max within-group prediction spread is also reported.")

    for mname, msuffix, is_wdmpnn in models:
        for split in splits:
            for tkey in ['EA', 'IP']:
                if is_wdmpnn:
                    if split == 'a_held_out':
                        preds = g2d.load_wdmpnn_predictions(tkey)
                    else:
                        preds, _ = g2d.load_wdmpnn_gen_predictions(tkey, split)
                else:
                    if split == 'a_held_out':
                        preds = g2d.load_hpg2stage_predictions(msuffix, tkey)
                    else:
                        preds = g2d.load_gen_predictions(msuffix, tkey, split)

                if not preds:
                    continue

                m = g2d.compute_metrics_from_preds(preds, df)
                r2_mean = m['R2_mean']
                arch_r2_mean = m['ArchR2_mean']

                details = []
                for p in preds:
                    if p['indices'] is None:
                        continue
                    d = g2d.audit_archdev_details(p['y_true'], p['y_pred'], p['indices'], df)
                    if d:
                        details.append(d)

                if not details:
                    continue

                avg_true_delta_mean = np.mean([d['true_delta_mean'] for d in details])
                avg_true_delta_std = np.mean([d['true_delta_std'] for d in details])
                avg_pred_delta_mean = np.mean([d['pred_delta_mean'] for d in details])
                avg_pred_delta_std = np.mean([d['pred_delta_std'] for d in details])
                max_spread = np.max([d['max_group_pred_spread'] for d in details]) if mname == 'Frac' else np.nan

                print(f"\n{mname:8s} | {split:14s} | {tkey}")
                print(f"  R2_mean                      = {r2_mean:.4f}")
                print(f"  ArchR2_mean                  = {arch_r2_mean:.4f}")
                print(f"  true Δ mean ± std            = {avg_true_delta_mean:.6f} ± {avg_true_delta_std:.6f}")
                print(f"  pred Δ mean ± std            = {avg_pred_delta_mean:.6f} ± {avg_pred_delta_std:.6f}")
                if mname == 'Frac':
                    print(f"  max within-group pred spread = {max_spread:.6f}")
                    if split == 'a_held_out' and tkey == 'EA':
                        verify_test_ids(preds, df, tkey)
                        print("\n  -- Correcting test indices using both EA and IP y_true --")
                        preds_ea = g2d.load_hpg2stage_predictions('frac', 'EA')
                        preds_ip = g2d.load_hpg2stage_predictions('frac', 'IP')
                        preds_corrected = correct_indices_with_both_targets(preds_ea, preds_ip, df)
                        # Verify corrected indices
                        total = sum(len(p['indices']) for p in preds_corrected)
                        matched = sum(np.sum(p['indices'] >= 0) for p in preds_corrected)
                        print(f"  Corrected index matches: {matched}/{total}")
                        inspect_frac_group_spread(preds_corrected, df, 'EA', use_raw_group=False)
                        inspect_frac_group_spread(preds_corrected, df, 'EA', use_raw_group=True)

    print("\n" + "=" * 70)


if __name__ == '__main__':
    run_audit()
