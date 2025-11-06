import logging
import numpy as np
import torch
from pathlib import Path
from typing import Optional, List, Dict

from chemprop import data, featurizers

from utils import (
    build_sklearn_models,
)

from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score, roc_auc_score
)

logger = logging.getLogger(__name__)

def fit_and_score_baselines(
    X_train, y_train, X_val, y_val, X_test, y_test,
    task_type: str,
    n_classes: Optional[int],
    *,
    dataset_name: str,
    model_name: str,
    target: str,
    replicate: int,
    variant_label: str
) -> Dict[str, Dict[str, float]]:
    """
    Trains sklearn baselines and returns a dict mapping model name to metrics:
    {model_name: {dataset, encoder, variant, replicate, target, test/metric...}}
    """
    specs = build_sklearn_models(task_type, n_classes, scaler_flag=True)
    results: Dict[str, Dict[str, float]] = {}

    # scale y for regression (inverse on preds later)
    target_scaler = None
    if task_type == "reg":
        from sklearn.preprocessing import StandardScaler
        target_scaler = StandardScaler()
        yt = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        yv = target_scaler.transform(y_val.reshape(-1, 1)).ravel()
    else:
        yt, yv = y_train, y_val

    for name, (model, needs_scaler) in specs.items():
        # feature scaling if needed
        if needs_scaler:
            from sklearn.preprocessing import StandardScaler
            xs = StandardScaler()
            Xt = xs.fit_transform(X_train)
            Xv = xs.transform(X_val)
            Xs = xs.transform(X_test)
        else:
            Xt, Xv, Xs = X_train, X_val, X_test

        if task_type == "reg":
            if name.upper().startswith("XGB"):
                model.set_params(early_stopping_rounds=30, eval_metric="rmse")
                model.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
            else:
                model.fit(Xt, yt)
            y_pred = model.predict(Xs)
            if target_scaler is not None:
                y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()

            row = {
                "dataset":  dataset_name,
                "encoder":  model_name,
                "variant":  variant_label,
                "test/mae": mean_absolute_error(y_test, y_pred),
                "test/r2":  r2_score(y_test, y_pred),
                "test/rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            }
            results[name] = row

        else:
            if name.upper().startswith("XGB"):
                eval_metric = "mlogloss" if task_type == "multi" else "logloss"
                model.set_params(early_stopping_rounds=30, eval_metric=eval_metric)
                model.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
            else:
                model.fit(Xt, yt)

            y_hat = model.predict(Xs)
            avg = "macro" if task_type == "multi" else "binary"
            row = {
                "dataset":   dataset_name,
                "encoder":   model_name,
                "variant":   variant_label,
                "test/accuracy": accuracy_score(y_test, y_hat),
                "test/f1":       f1_score(y_test, y_hat, average=avg),
            }
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(Xs)
                try:
                    if task_type == "binary":
                        row["test/roc_auc"] = roc_auc_score(y_test, proba[:, 1])
                    else:
                        from sklearn.preprocessing import label_binarize
                        y_bin = label_binarize(y_test, classes=list(range(n_classes)))
                        row["test/roc_auc"] = roc_auc_score(y_bin, proba, average="macro", multi_class="ovr")
                except Exception:
                    pass
            results[name] = row

    return results


def build_dmpnn_loaders(args, setup_info, target, train_idx, val_idx, test_idx, smis, df_input,
                        combined_descriptor_data):
    from utils import create_all_data
    small_molecule_models = ["DMPNN", "DMPNN_DiffPool", "PPG"]

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer() \
        if args.model_name in small_molecule_models else featurizers.PolymerMolGraphFeaturizer()

    ys = df_input.loc[:, target].astype(float).values
    if args.task_type != 'reg': ys = ys.astype(int)
    ys = ys.reshape(-1, 1)

    all_data = create_all_data(smis, ys, combined_descriptor_data, args.model_name)
    train_data, val_data, test_data = data.split_data_by_indices(
        all_data, [train_idx], [val_idx], [test_idx]
    )
    # index 0 because split_data_by_indices returns lists per split
    train = data.MoleculeDataset(train_data[0], featurizer)
    val   = data.MoleculeDataset(val_data[0],   featurizer)
    test  = data.MoleculeDataset(test_data[0],  featurizer)

    # Regression target scaling like train_graph.py
    if args.task_type == 'reg':
        scaler = train.normalize_targets()
        val.normalize_targets(scaler)
        # test targets intentionally unscaled
    else:
        scaler = None

    num_workers = setup_info['num_workers']
    use_workers = num_workers if torch.cuda.is_available() else 0

    train_loader = data.build_dataloader(train, batch_size=args.batch_size, num_workers=use_workers, shuffle=False)
    val_loader   = data.build_dataloader(val,   batch_size=args.batch_size, num_workers=use_workers, shuffle=False)
    test_loader  = data.build_dataloader(test,  batch_size=args.batch_size, num_workers=use_workers, shuffle=False)
    return train_loader, val_loader, test_loader

def extract_embeddings_for_rep(args, setup_info, target, rep, tr, va, te,
                               smis, df_input, combined_descriptor_data,
                               ckpt_path: Path, emb_dir: Path, embedding_prefix: str,
                               n_classes_per_target: Dict, device: torch.device,
                               keep_eps: float = 1e-8):
    """Returns (X_train, X_val, X_test, keep_mask) and writes cache to both primary & temp."""
    smiles_column = setup_info['smiles_column']
    
    # Build loaders
    if args.model_name == "AttentiveFP":
        from attentivefp_utils import build_attentivefp_loaders, create_attentivefp_model, export_attentivefp_embeddings
        checkpoint_file = ckpt_path / "best.pt"
        if not checkpoint_file.exists():
            logger.warning(f"[{target}] split {rep}: missing checkpoint; skipping split for baselines")
            return None
        # Build loaders
        df_tr = df_input.iloc[tr].reset_index(drop=True)
        df_va = df_input.iloc[va].reset_index(drop=True)
        df_te = df_input.iloc[te].reset_index(drop=True)
        train_loader, val_loader, test_loader, _ = build_attentivefp_loaders(
            args, df_tr, df_va, df_te, smiles_column, target, eval=True
        )
        # Model + weights
        model = create_attentivefp_model(
            task_type=args.task_type,
            n_classes=n_classes_per_target.get(target, None),
            hidden_channels=args.hidden,
            num_layers=2,
            num_timesteps=2,
            dropout=0.0,
        ).to(device)
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict({k: v.to(device) for k, v in state_dict.items()})

        export_attentivefp_embeddings(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            emb_dir=emb_dir,
            embedding_prefix=embedding_prefix,
            split_idx=rep,
            logger=logger,
            overwrite=True if args.force_recompute else False,
        )
        

    else:
        from utils import pick_best_checkpoint, get_encodings_from_loader
        # DMPNN-family
        train_loader, val_loader, test_loader = build_dmpnn_loaders(
            args, setup_info, target, tr, va, te, smis, df_input, combined_descriptor_data
        )
        ckpt = pick_best_checkpoint(ckpt_path)
        if not ckpt:
            logger.warning(f"[rep {rep}] missing checkpoint dir")
            return None
        logger.info(f"[rep {rep}] checkpoint: {ckpt}")
        map_location = None if torch.cuda.is_available() else torch.device('cpu')
        if args.model_name == "DMPNN_DiffPool":
            # If you’ve got a helper in utils to instantiate, call that; otherwise leave as-is.
            from chemprop import nn, models
            base_mp_cls = nn.BondMessagePassing
            depth = getattr(args, "diffpool_depth", 1)
            ratio = getattr(args, "diffpool_ratio", 0.5)
            mp = nn.BondMessagePassingWithDiffPool(base_mp_cls=base_mp_cls, depth=depth, ratio=ratio)
            desc_dim = (combined_descriptor_data.shape[1] if combined_descriptor_data is not None else 0)
            input_dim = mp.output_dim + desc_dim
            if args.task_type == 'reg':
                predictor = nn.RegressionFFN(output_transform=None, n_tasks=1, input_dim=input_dim)
            elif args.task_type == 'binary':
                predictor = nn.BinaryClassificationFFN(input_dim=input_dim)
            else:
                n_classes = max(2, int(df_input[target].dropna().nunique()))
                predictor = nn.MulticlassClassificationFFN(n_classes=n_classes, input_dim=input_dim)
            mpnn = models.MPNN(message_passing=mp, agg=nn.IdentityAggregation(),
                               predictor=predictor, batch_norm=args.batch_norm, metrics=[])
            import torch as _t
            state = _t.load(ckpt, map_location=map_location)
            mpnn.load_state_dict(state["state_dict"], strict=False)
            mpnn.eval()
        else:
            from chemprop import models
            mpnn = models.MPNN.load_from_checkpoint(str(ckpt), map_location=map_location)
            mpnn.eval()
        X_train = get_encodings_from_loader(mpnn, train_loader)
        X_val   = get_encodings_from_loader(mpnn, val_loader)
        X_test  = get_encodings_from_loader(mpnn, test_loader)
        std_train = X_train.std(axis=0)
        keep = std_train > keep_eps
        from utils import embedding_files
        emb_dir.mkdir(parents=True, exist_ok=True)
        mask_f, Xtr_f, Xva_f, Xte_f = embedding_files(emb_dir, embedding_prefix, rep)
        np.save(mask_f, keep)
        np.save(Xtr_f, X_train[:, keep])
        np.save(Xva_f,  X_val[:,  keep])
        np.save(Xte_f,  X_test[:, keep])
