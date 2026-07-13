"""Step 8: Chemical novelty of held-out monomers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from .config import (
    MODELS, TARGETS, N_FOLDS, STEP_DIRS, FOLD_MONOMER_NAMES,
)
from .data_loading import load_predictions_single, load_split_meta
from .grouping import build_fold_df, add_group_means, filter_matched_groups


def _compute_monomer_descriptors(smiles: str) -> dict:
    """Compute basic RDKit descriptors for a monomer SMILES."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
    except ImportError:
        return {'mw': np.nan, 'n_aromatic_rings': np.nan, 'n_rings': np.nan,
                'n_heteroatoms': np.nan, 'n_N': np.nan, 'n_S': np.nan,
                'n_F': np.nan, 'formal_charge': np.nan,
                'hbd': np.nan, 'hba': np.nan}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {'mw': np.nan, 'n_aromatic_rings': np.nan, 'n_rings': np.nan,
                'n_heteroatoms': np.nan, 'n_N': np.nan, 'n_S': np.nan,
                'n_F': np.nan, 'formal_charge': np.nan,
                'hbd': np.nan, 'hba': np.nan}

    ring_info = mol.GetRingInfo()
    n_aromatic = sum(1 for ring in ring_info.BondRings()
                     if all(mol.GetBondWithIdx(bi).GetIsAromatic() for bi in ring))

    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    return {
        'mw': float(Descriptors.MolWt(mol)),
        'n_aromatic_rings': n_aromatic,
        'n_rings': ring_info.NumRings(),
        'n_heteroatoms': sum(1 for a in mol.GetAtoms() if a.GetSymbol() not in ('C', 'H')),
        'n_N': atoms.count('N'),
        'n_S': atoms.count('S'),
        'n_F': atoms.count('F'),
        'formal_charge': int(Chem.GetFormalCharge(mol)),
        'hbd': rdMolDescriptors.CalcNumHBD(mol),
        'hba': rdMolDescriptors.CalcNumHBA(mol),
    }


def _compute_tanimoto_similarity(smiles_query: str, smiles_set: list[str],
                                  radius: int = 2) -> dict:
    """Compute Tanimoto similarity of query to a set of SMILES."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit import DataStructs
    except ImportError:
        return {'max_tanimoto': np.nan, 'mean_top5': np.nan,
                'nearest_smiles': ''}

    mol_q = Chem.MolFromSmiles(smiles_query)
    if mol_q is None:
        return {'max_tanimoto': np.nan, 'mean_top5': np.nan,
                'nearest_smiles': ''}
    fp_q = AllChem.GetMorganFingerprintAsBitVect(mol_q, radius, nBits=2048)

    sims = []
    for smi in smiles_set:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=2048)
        sim = DataStructs.TanimotoSimilarity(fp_q, fp)
        sims.append((sim, smi))

    if not sims:
        return {'max_tanimoto': np.nan, 'mean_top5': np.nan,
                'nearest_smiles': ''}

    sims.sort(reverse=True)
    max_sim = sims[0][0]
    top5 = [s[0] for s in sims[:5]]
    nearest = sims[0][1]

    return {
        'max_tanimoto': float(max_sim),
        'mean_top5': float(np.mean(top5)),
        'nearest_smiles': nearest,
    }


def run_novelty(df: pd.DataFrame, meta: dict[str, list]) -> pd.DataFrame:
    """
    Step 8: Chemical novelty analysis for monomer-heldout folds.
    Saves heldout_monomer_novelty.csv.
    """
    out_dir = STEP_DIRS['07_monomer_novelty']
    split = 'monomer_heldout'
    meta_folds = meta[split]
    n_folds = N_FOLDS[split]

    # Collect all unique monomers_A in the dataset
    all_monomers_A = set(df['smiles_A'].astype(str).unique())

    rows = []
    for fold in range(n_folds):
        fold_meta = next((r for r in meta_folds if r['fold'] == fold), None)
        if fold_meta is None:
            continue

        held_out = fold_meta.get('held_out_monomer_A', '')
        if not held_out:
            continue

        # Training monomers = all except held-out
        train_monomers = sorted(all_monomers_A - {held_out})

        # Chemical descriptors
        desc = _compute_monomer_descriptors(held_out)

        # Tanimoto similarity
        tani = _compute_tanimoto_similarity(held_out, train_monomers)

        row = {
            'fold': fold,
            'monomer_name': FOLD_MONOMER_NAMES.get(fold, ''),
            'monomer_smiles': held_out,
            **desc,
            **tani,
        }

        # Performance metrics per model
        for model in MODELS:
            for tkey in TARGETS:
                pred = load_predictions_single(model, tkey, split, fold, meta_folds)
                if pred is None:
                    row[f'{model}_{tkey}_r2'] = np.nan
                    row[f'{model}_{tkey}_gm_r2'] = np.nan
                    row[f'{model}_{tkey}_delta_r2'] = np.nan
                    continue

                yt, yp = pred['y_true'], pred['y_pred']
                gidx = pred['global_idx']

                # Overall R²
                row[f'{model}_{tkey}_r2'] = float(r2_score(yt, yp)) if len(yt) > 2 else np.nan

                # Group-mean and delta R²
                fdf = build_fold_df(df, yt, yp, gidx)
                matched = filter_matched_groups(fdf)
                if len(matched) > 10:
                    matched = add_group_means(matched)
                    # Group-mean R²
                    gm = matched.groupby('group_key').agg(
                        y_bar_t=('y_true', 'mean'),
                        y_bar_p=('y_pred', 'mean'),
                    )
                    if len(gm) > 2 and gm['y_bar_t'].std() > 1e-10:
                        row[f'{model}_{tkey}_gm_r2'] = float(
                            r2_score(gm['y_bar_t'], gm['y_bar_p'])
                        )
                    else:
                        row[f'{model}_{tkey}_gm_r2'] = np.nan

                    # Delta R²
                    dt = matched['delta_true'].values
                    dp = matched['delta_pred'].values
                    if np.std(dt) > 1e-10:
                        row[f'{model}_{tkey}_delta_r2'] = float(r2_score(dt, dp))
                    else:
                        row[f'{model}_{tkey}_delta_r2'] = np.nan
                else:
                    row[f'{model}_{tkey}_gm_r2'] = np.nan
                    row[f'{model}_{tkey}_delta_r2'] = np.nan

        rows.append(row)

    nov_df = pd.DataFrame(rows)
    nov_df.to_csv(out_dir / 'heldout_monomer_novelty.csv', index=False)
    print(f"  Step 8 (novelty) complete: {out_dir}")
    return nov_df
