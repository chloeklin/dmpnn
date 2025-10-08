#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train on the PAE Tg dataset using either:
  (a) PAPER-211: composition-weighted (A,B) repeat-pair descriptors from ru_desc.csv
  (b) MONO-211 : composition-weighted monomer-only descriptors (computed from SMILES)

This script reuses your existing training loop 'train(...)' and utilities.
It only replaces the data-loading/featurisation step to handle the two-file PAE format.
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from utils import compute_group_id


# --- RDKit only needed for MONO-211 ---
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    RDKit_AVAILABLE = True
except Exception:
    RDKit_AVAILABLE = False

# === import your existing modules ===
from tabular_utils import (
    preprocess_descriptor_data,  # used by your train loop
    save_preprocessing_objects,  # used by your train loop
    build_features,              # NOT used here for descriptors (we precompute)
    eval_regression, eval_binary, eval_multi
)
from utils import (
    set_seed, load_existing_results, save_combined_results,
    prepare_target_data, build_sklearn_models, setup_training_environment,
    load_and_preprocess_data,  # NOT used here (we have a custom loader)
    determine_split_strategy, generate_data_splits, group_splits
)

# ---------------------------- Your train() (copied as-is) ----------------------------
# NOTE: reusing your provided train() without changes.
import joblib
from sklearn.preprocessing import StandardScaler

def train(df, y, target_name, descriptor_columns, replicates, seed, out_dir, args, existing_results=None, smiles_column="smiles"):
    logger = logging.getLogger(__name__)
    if args.task_type == "reg":
        valid_mask = ~np.isnan(y)
    else:
        valid_mask = ~pd.isna(y)
    if not valid_mask.all():
        n_invalid = (~valid_mask).sum()
        logger.info(f"Filtering out {n_invalid} samples with NaN target values for {target_name}")
        df_valid = df.iloc[valid_mask].reset_index(drop=True)
        y_valid = y[valid_mask]
    else:
        df_valid = df
        y_valid = y

    n_splits, local_reps = determine_split_strategy(len(y_valid), replicates)

    if args.polymer_type == "copolymer":
        train_indices, val_indices, test_indices = [], [], []
        for r in range(local_reps):
            tr, va, te = group_splits(df_valid, y_valid, args.task_type, n_splits, seed + r)
            train_indices.extend(tr)
            val_indices.extend(va)
            test_indices.extend(te)
    else:
        train_indices, val_indices, test_indices = generate_data_splits(args, y_valid, n_splits, local_reps, seed)

    detailed_rows = []
    for i, (train_idx, val_idx, test_idx) in enumerate(zip(train_indices, val_indices, test_indices)):
        # Build features using your existing function (ab/rdkit); we only want descriptor block here.
        ab_block, descriptor_block, feat_names = build_features(
            df_valid, train_idx, descriptor_columns,
            args.polymer_type, use_rdkit=args.incl_rdkit,
            use_ab=args.incl_ab, smiles_column=smiles_column
        )
        orig_desc_names = [n for n in feat_names if not n.startswith('AB_')]

        if ab_block is not None:
            ab_tr, ab_val, ab_te = ab_block[train_idx], ab_block[val_idx], ab_block[test_idx]
            ab_names = [name for name in feat_names if name.startswith('AB_')]

        if descriptor_block is not None:
            (desc_tr_selected, desc_val_selected, desc_te_selected, selected_desc_names,
             preprocessing_metadata, imputer, constant_mask, corr_mask) = preprocess_descriptor_data(
                descriptor_block, train_idx, val_idx, test_idx, orig_desc_names, logger
            )
            if ab_block is not None:
                X_tr = np.concatenate([ab_tr, desc_tr_selected], axis=1)
                X_val = np.concatenate([ab_val, desc_val_selected], axis=1)
                X_te = np.concatenate([ab_te, desc_te_selected], axis=1)
                feat_names = ab_names + selected_desc_names
            else:
                X_tr, X_val, X_te = desc_tr_selected, desc_val_selected, desc_te_selected
                feat_names = selected_desc_names
        else:
            X_tr, X_val, X_te = ab_tr, ab_val, ab_te
            feat_names = ab_names

        if descriptor_block is not None:
            save_preprocessing_objects(out_dir, i, preprocessing_metadata, imputer, constant_mask, corr_mask, selected_desc_names)

        if X_tr.shape[1] == 0:
            logger.warning("Feature selection yielded 0 columns; reverting to AB features only for this split.")
            X_tr, X_val, X_te = ab_tr, ab_val, ab_te
            feat_names = ab_names

        target_scaler = None
        if args.task_type == 'reg':
            target_scaler = StandardScaler()
            y_tr = target_scaler.fit_transform(y_valid[train_idx].reshape(-1, 1)).flatten()
            y_val = target_scaler.transform(y_valid[val_idx].reshape(-1, 1)).flatten()
            y_te = y_valid[test_idx]
        else:
            y_tr, y_val, y_te = y_valid[train_idx], y_valid[val_idx], y_valid[test_idx]

        num_classes = len(np.unique(y_tr)) if args.task_type != "reg" else None
        model_specs = build_sklearn_models(args.task_type, num_classes, scaler_flag=True)

        for name, (model, needs_scaler) in model_specs.items():
            if (existing_results and target_name in existing_results and
                i in existing_results[target_name] and name in existing_results[target_name][i]):
                logging.info(f"Skipping {target_name} split {i} model {name} (already completed)")
                continue

            from sklearn.preprocessing import StandardScaler as _SS
            scaler = _SS() if needs_scaler else None
            if scaler is not None:
                Xtr_fit = scaler.fit_transform(X_tr); Xval_fit = scaler.transform(X_val); Xte_fit = scaler.transform(X_te)
                joblib.dump(scaler, args.out_dir / f"feature_scaler_split_{i}_{name}.pkl" if hasattr(args, "out_dir") else Path(args.results_dir) / f"feature_scaler_split_{i}_{name}.pkl")
            else:
                Xtr_fit, Xval_fit, Xte_fit = X_tr, X_val, X_te

            if args.task_type == "reg":
                if name == "XGB":
                    model.set_params(early_stopping_rounds=30, eval_metric="rmse")
                    model.fit(Xtr_fit, y_tr, eval_set=[(Xval_fit, y_val)], verbose=False)
                else:
                    model.fit(Xtr_fit, y_tr)
                y_pred = model.predict(Xte_fit)
                if target_scaler is not None:
                    y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                metrics = eval_regression(y_te, y_pred)

            elif args.task_type == "binary":
                if name == "XGB":
                    model.set_params(early_stopping_rounds=30, eval_metric="logloss")
                    model.fit(Xtr_fit, y_tr, eval_set=[(Xval_fit, y_val)], verbose=False)
                else:
                    model.fit(Xtr_fit, y_tr)
                y_prob = getattr(model, "predict_proba", None)
                prob1 = y_prob(Xte_fit)[:,1] if y_prob is not None else None
                y_pred = model.predict(Xte_fit)
                metrics = eval_binary(y_te, y_pred, prob1)

            else:  # multi
                if name == "XGB":
                    model.set_params(early_stopping_rounds=30, eval_metric="mlogloss")
                    model.fit(Xtr_fit, y_tr, eval_set=[(Xval_fit, y_val)], verbose=False)
                else:
                    model.fit(Xtr_fit, y_tr)
                y_proba = getattr(model, "predict_proba", None)
                proba = y_proba(Xte_fit) if y_proba is not None else None
                y_pred = model.predict(Xte_fit)
                metrics = eval_multi(y_te, y_pred, proba)

            row = {"target": target_name, "split": i, "model": name}
            row.update({k: float(v) for k, v in metrics.items()})
            detailed_rows.append(row)
    return detailed_rows

# ---------------------------- PAE-specific utilities ----------------------------

A_SIDE = [("a1","ratio_a1","smilesA1"), ("a2","ratio_a2","smilesA2"), ("a3","ratio_a3","smilesA3"), ("a4","ratio_a4","smilesA4")]
B_SIDE = [("b1","ratio_b1","smilesB1"), ("b2","ratio_b2","smilesB2"), ("b3","ratio_b3","smilesB3"), ("b4","ratio_b4","smilesB4")]

def _extract_side(row, side_cols):
    triples = []
    for name_col, ratio_col, smi_col in side_cols:
        name = row.get(name_col)
        ratio = row.get(ratio_col)
        smi = row.get(smi_col)
        if pd.notna(name) and pd.notna(ratio) and float(ratio) > 0:
            triples.append((str(name).strip(), float(ratio), str(smi) if pd.notna(smi) else None))
    return triples

def _norm_weights(triples):
    s = sum(w for _, w, _ in triples)
    if s <= 0: 
        return []
    return [(nm, w/s, smi) for (nm, w, smi) in triples]

def load_ru_desc(path_ru: Path):
    ru = pd.read_csv(path_ru)
    ru["A_key_norm"] = ru["A_key"].astype(str).str.strip().str.upper()
    ru["B_key_norm"] = ru["B_key"].astype(str).str.strip().str.upper()

    non_desc = {"A_key","B_key","A_key_norm","B_key_norm"}
    desc_cols = [c for c in ru.columns if c not in non_desc]

    pair_map = {}
    for _, r in ru.iterrows():
        ak, bk = r.A_key_norm, r.B_key_norm
        vec = r[desc_cols].to_numpy(dtype=float)
        pair_map[(ak, bk)] = vec
        pair_map[(bk, ak)] = vec   # <- add the reverse
    return ru, desc_cols, pair_map


# ---- RDKit descriptor helpers (MONO-211) ----

def _build_rdkit_func_table():
    """
    Build a name->callable table from RDKit's registered descriptors.
    We will later subset to the exact 211 names present in ru_desc header.
    """
    name_to_func = {name: func for (name, func) in Descriptors.descList}  # ~200+ standard descriptors
    return name_to_func

def mono_211_from_smiles(smiles: str, ordered_names, name_to_func):
    """Compute all RDKit descriptors then select/align to 'ordered_names'."""
    mol = Chem.MolFromSmiles(smiles)  # remove polymer anchor if present
    if mol is None:
        return None
    # compute full set we know
    values = {}
    for nm, fn in name_to_func.items():
        try:
            values[nm] = float(fn(mol))
        except Exception:
            values[nm] = np.nan
    # align to requested names; missing become nan (we'll impute in your pipeline)
    return np.array([values.get(nm, np.nan) for nm in ordered_names], dtype=float)

# ---------------------------- Feature builders ----------------------------

def build_paper_211(pae_df: pd.DataFrame, desc_cols, pair_map):
    feats, keep_idx, missing = [], [], []
    for i, row in pae_df.iterrows():
        A = _norm_weights(_extract_side(row, A_SIDE))
        B = _norm_weights(_extract_side(row, B_SIDE))
        if not A or not B:
            continue
        vec = np.zeros(len(desc_cols), dtype=float)
        ok = True
        for an, wa, _ in A:
            ak = an.strip().upper()
            for bn, wb, _ in B:
                bk = bn.strip().upper()
                pair = pair_map.get((ak, bk))
                if pair is None:
                    ok = False
                    missing.append((ak, bk))
                    break
                vec += (wa * wb) * pair
            if not ok: break
        if not ok: 
            continue
        feats.append(vec)
        keep_idx.append(i)
    X = pd.DataFrame(feats, columns=desc_cols).reset_index(drop=True)
    y = pae_df.loc[keep_idx, "Tg"].to_numpy(dtype=float)
    info = {"n_rows_in": len(pae_df), "n_rows_out": len(keep_idx), "missing_pairs": pd.Series(missing).value_counts().to_dict()}
    return X, y, keep_idx, info

def build_mono_211(pae_df: pd.DataFrame, desc_cols):
    if not RDKit_AVAILABLE:
        raise RuntimeError("RDKit not available; cannot compute MONO-211 features.")
    name_to_func = _build_rdkit_func_table()
    feats, keep_idx = [], []
    cache = {}
    for i, row in pae_df.iterrows():
        A = _norm_weights(_extract_side(row, A_SIDE))
        B = _norm_weights(_extract_side(row, B_SIDE))
        if not A and not B:
            continue
        vec = np.zeros(len(desc_cols), dtype=float)
        ok = True
        # A-side
        for _, wA, smi in A:
            if not smi: ok=False; break
            if smi not in cache:
                cache[smi] = mono_211_from_smiles(smi, desc_cols, name_to_func)
            v = cache[smi]
            if v is None:
                ok=False; break
            vec += wA * v
        if not ok: 
            continue
        # B-side
        for _, wB, smi in B:
            if not smi: ok=False; break
            if smi not in cache:
                cache[smi] = mono_211_from_smiles(smi, desc_cols, name_to_func)
            v = cache[smi]
            if v is None:
                ok=False; break
            vec += wB * v
        if not ok:
            continue
        feats.append(vec)
        keep_idx.append(i)
    X = pd.DataFrame(feats, columns=desc_cols).reset_index(drop=True)
    y = pae_df.loc[keep_idx, "Tg"].to_numpy(dtype=float)
    info = {"n_rows_in": len(pae_df), "n_rows_out": len(keep_idx)}
    return X, y, keep_idx, info

# ------------------------------- Main --------------------------------

def main():
    p = argparse.ArgumentParser(description="Train on PAE Tg with PAPER-211 vs MONO-211 descriptors")
    
    p.add_argument("--repr", type=str, choices=["paper", "mono", "both"], default="both",
                   help="Representation to use: paper (pair 211), mono (monomer 211), or both")
    # passthroughs to your infra
    p.add_argument("--pae_wide", type=Path, help="Path to pae_wide.csv",default="data/pae_wide.csv")
    p.add_argument("--ru_desc",  type=Path, help="Path to ru_desc.csv (211 pair descriptors)",default="data/ru_desc.csv")
    p.add_argument('--task_type', type=str, choices=['reg', 'binary', 'multi'], default="reg")
    p.add_argument("--polymer_type", type=str, choices=["homo", "copolymer"], default="copolymer")
    p.add_argument("--dataset_name", type=str, default="pae_tg")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("pae_tg")

    # === Your environment/setup (directories, seeds, model registry, etc.) ===
    setup = setup_training_environment(args, model_type="tabular")
    results_dir     = setup['results_dir']
    feat_select_dir = setup['feat_select_dir']
    SEED            = setup['SEED']
    REPLICATES      = setup['REPLICATES']

    # store for scalers (train loop expects args.out_dir sometimes)
    args.results_dir = results_dir
    args.out_dir = results_dir

    set_seed(SEED)

    # === Load input tables ===
    pae = pd.read_csv(args.pae_wide)
    gids_full = compute_group_id(pae)
    ru, desc_cols, pair_map = load_ru_desc(args.ru_desc)
    logger.info(f"Loaded pae_wide: {len(pae)} rows; ru_desc: {len(ru)} pairs; 211 columns detected.")

    # === Build feature matrices ===
    runs = []
    # PAPER-211
    if args.repr in ("paper", "both"):
        Xp, y_p, keep_p, info_p = build_paper_211(pae, desc_cols, pair_map)
        logger.info(f"[PAPER-211] usable rows: {info_p['n_rows_out']}/{info_p['n_rows_in']}")
        if info_p.get("missing_pairs"):
            logger.info(f"[PAPER-211] missing pair counts (top): {dict(list(info_p['missing_pairs'].items())[:8])}")
        # Create DataFrame in one go
        df_paper = pd.DataFrame({
            'Tg': y_p,
            'group_id': gids_full.iloc[keep_p].reset_index(drop=True).values
        })
        # Add descriptor columns efficiently
        df_paper = pd.concat([df_paper, Xp], axis=1)
        runs.append(("paper211", df_paper, y_p, desc_cols))

    # MONO-211
    if args.repr in ("mono", "both"):
        Xm, y_m, keep_m, info_m = build_mono_211(pae, desc_cols)
        logger.info(f"[MONO-211 ] usable rows: {info_m['n_rows_out']}/{info_m['n_rows_in']}")
        # Create DataFrame in one go
        df_mono = pd.DataFrame({
            'Tg': y_m,
            'group_id': gids_full.iloc[keep_m].reset_index(drop=True).values
        })
        # Add descriptor columns efficiently
        df_mono = pd.concat([df_mono, Xm], axis=1)
        runs.append(("mono211", df_mono, y_m, desc_cols))


    if not runs:
        raise SystemExit("No representation selected.")

    # === Common flags: only descriptors (no AB, no on-the-fly RDKit) ===
    args.incl_desc  = False
    args.incl_ab    = False
    args.incl_rdkit = True



    # === Train for each representation; save results separately ===
    tabular_results_dir = results_dir / "tabular"
    tabular_results_dir.mkdir(exist_ok=True)

    for tag, df_in, y_vec, cols in runs:
        detailed_csv = tabular_results_dir / f"{args.dataset_name}_{tag}.csv"
        existing_results = load_existing_results(detailed_csv, logger)

        # One target: Tg
        tcol = "Tg"
        out_dir = feat_select_dir / f"{tcol}_{tag}"
        out_dir.mkdir(parents=True, exist_ok=True)

        rows = train(
            df=df_in, y=y_vec, target_name=tcol,
            descriptor_columns=cols, replicates=REPLICATES,
            seed=SEED, out_dir=out_dir, args=args,
            existing_results=existing_results.get(tcol, {}),
            smiles_column="smiles"  # unused for this path
        )

        # Save combined results
        existing_df = None
        if detailed_csv.exists() and existing_results:
            try:
                existing_df = pd.read_csv(detailed_csv)
            except Exception as e:
                logger.warning(f"Could not reload existing results: {e}")
        save_combined_results(detailed_csv, existing_df, rows, logger)

    logger.info("Done.")

if __name__ == "__main__":
    main()
