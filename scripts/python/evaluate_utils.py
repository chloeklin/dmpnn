from utils import (
    build_sklearn_models,
)

from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score, roc_auc_score
)


def fit_and_score_baselines(X_train, y_train, X_val, y_val, X_test, y_test, task_type: str, n_classes: int | None) -> Dict[str, Dict[str, float]]:
    """Train a panel of sklearn baselines and return test metrics per model name."""
    specs = build_sklearn_models(task_type, n_classes, scaler_flag=True)
    scores = {}
    for name in model_specs.keys():
            if (i, name) not in scores:
                scores[(i, name)] = {
                    "dataset": args.dataset_name,
                    "encoder": args.model_name,
                    "variant": variant_label,
                    "replicate": i,
                    "model": name
                }
    target_scaler = None
    if task_type == 'reg':
        from sklearn.preprocessing import StandardScaler
        target_scaler = StandardScaler()
        yt = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        yv = target_scaler.transform(y_val.reshape(-1, 1)).flatten()
        # Keep y_test original for final evaluation
    else:
        yt = y_train
        yv = y_val
    for name, (model, needs_scaler) in specs.items():
        xs = None
        if needs_scaler:
            from sklearn.preprocessing import StandardScaler
            xs = StandardScaler()
            Xt = xs.fit_transform(X_train)
            Xv = xs.transform(X_val)
            Xs = xs.transform(X_test)
        else:
            Xt, Xv, Xs = X_train, X_val, X_test

        if task_type == "reg":
            if name == "XGB":
                model.set_params(early_stopping_rounds=30, eval_metric="rmse")
                model.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
            else:
                model.fit(Xt, yt)
            y_pred = model.predict(Xs)
            # inverse-transform if scaler used on y
            if target_scaler is not None:
                y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            scores[(i,name)] = {
                f"{target}_mae": mean_absolute_error(y_test, y_pred),
                f"{target}_r2": r2_score(y_test, y_pred),
                f"{target}_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            }
        else:
            # classification
            if name == "XGB":
                # Use appropriate eval_metric for classification task
                eval_metric = "mlogloss" if args.task_type == "multi" else "logloss"
                model.set_params(early_stopping_rounds=30, eval_metric=eval_metric)
                model.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
            else:
                model.fit(Xt, yt)

            y_hat = model.predict(Xs)
            avg = "macro" if task_type == "multi" else "binary"
            entry = {
                f"{target}_acc": accuracy_score(y_test, y_hat),
                f"{target}_f1": f1_score(y_test, y_hat, average=avg),
            }
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(Xs)
                try:
                    if task_type == "binary":
                        entry[f"{target}_roc_auc"] = roc_auc_score(y_test, proba[:, 1])
                    else:
                        from sklearn.preprocessing import label_binarize
                        y_bin = label_binarize(y_test, classes=list(range(n_classes)))
                        entry[f"{target}_roc_auc"] = roc_auc_score(y_bin, proba, average="macro", multi_class="ovr")
                except Exception:
                    pass
            scores[name] = entry
    return scores