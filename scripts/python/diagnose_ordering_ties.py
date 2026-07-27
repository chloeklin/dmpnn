from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.python.analyze_regen_v1 import FOLDS, MODELS, OLD_DIR, TARGETS, prediction_path


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "ea_ip.csv")
    rows = []
    for model in MODELS:
        for target in TARGETS:
            for fold in FOLDS:
                with np.load(prediction_path(OLD_DIR, model, target, fold, 42), allow_pickle=True) as archive:
                    indices = archive["test_indices"].astype(int)
                    y_true = archive["y_true"].astype(float)
                    y_pred = archive["y_pred"].astype(float)
                frame = df.iloc[indices][["smiles_A", "smiles_B", "fracA", "poly_type"]].copy().reset_index(drop=True)
                frame["y_true"] = y_true
                frame["y_pred"] = y_pred
                frame["group"] = frame.smiles_A.astype(str) + "||" + frame.smiles_B.astype(str) + "||" + frame.fracA.astype(str)
                valid = frame.groupby("group").poly_type.nunique()
                frame = frame[frame.group.isin(valid[valid >= 2].index)]
                exact_ties = 0
                pair_count = 0
                group_scores = {"strict": [], "half_credit": [], "ties_correct": []}
                for _, group in frame.groupby("group"):
                    true_values = group.y_true.to_numpy()
                    pred_values = group.y_pred.to_numpy()
                    pairs = [(i, j) for i in range(len(group)) for j in range(i + 1, len(group)) if true_values[i] != true_values[j]]
                    scores = {key: [] for key in group_scores}
                    for i, j in pairs:
                        product = (true_values[i] - true_values[j]) * (pred_values[i] - pred_values[j])
                        tied = pred_values[i] == pred_values[j]
                        exact_ties += int(tied)
                        pair_count += 1
                        scores["strict"].append(float(product > 0))
                        scores["half_credit"].append(0.5 if tied else float(product > 0))
                        scores["ties_correct"].append(float(product >= 0))
                    for key in group_scores:
                        group_scores[key].append(np.mean(scores[key]))
                rows.append({
                    "model": model,
                    "target": target,
                    "fold": fold,
                    "exact_pred_ties": exact_ties,
                    "informative_pairs": pair_count,
                    "exact_tie_rate": exact_ties / pair_count,
                    **{key: float(np.mean(value)) for key, value in group_scores.items()},
                })
    detail = pd.DataFrame(rows)
    summary = detail.groupby("model", as_index=False).agg(
        exact_pred_ties=("exact_pred_ties", "sum"),
        informative_pairs=("informative_pairs", "sum"),
        folds_with_ties=("exact_pred_ties", lambda values: int((values > 0).sum())),
        maximum_fold_ties=("exact_pred_ties", "max"),
    )
    summary["exact_tie_rate"] = summary.exact_pred_ties / summary.informative_pairs
    output = ROOT / "analysis" / "model_diagnostics" / "_ordering_tie_diagnostic.csv"
    detail.to_csv(output, index=False)
    summary = summary.sort_values("exact_pred_ties", ascending=False)
    summary.to_csv(output.with_name("_ordering_tie_summary.csv"), index=False)
    convention_medians = detail[detail.model == "hpg_hier_octamer"].groupby("target", as_index=False)[["strict", "half_credit", "ties_correct"]].median()
    report = [
        "# Ordering Tie Diagnostic",
        "",
        "## Exact prediction ties by model",
        "",
        summary.to_markdown(index=False),
        "",
        "## Exact prediction ties by model, target, and fold",
        "",
        detail[["model", "target", "fold", "exact_pred_ties", "informative_pairs", "exact_tie_rate"]].to_markdown(index=False),
        "",
        "## Tie conventions",
        "",
        "The committed old inline metric used `sign_product > 0`; exact prediction ties therefore scored 0. The canonical module initially copied that rule. The selected canonical convention now gives an exact prediction tie 0.5 credit, representing expected accuracy under random tie breaking.",
        "",
        "Old inline expression:",
        "",
        "```python",
        "scores.append(np.mean([(yt[i] - yt[j]) * (yp[i] - yp[j]) > 0 for i, j in pairs]))",
        "```",
        "",
        "Selected canonical expression:",
        "",
        "```python",
        "0.5 if pred_values[i] == pred_values[j]",
        "else float((true_values[i] - true_values[j]) * (pred_values[i] - pred_values[j]) > 0)",
        "```",
        "",
        "## Octamer median ordering under each convention",
        "",
        convention_medians.to_markdown(index=False),
        "",
        "The frozen Phase-1 reference used half credit: EA `0.818263` rounds to `0.81826`, and IP `0.827061` rounds to `0.82706`. The strict convention produced `0.818182` and `0.826979`. No other model has an exact prediction tie in these 90 seed-42 cells.",
        "",
    ]
    output.with_suffix(".md").write_text("\n".join(report))
    print(summary.to_string(index=False))
    print("\nOctamer by fold:")
    print(detail[detail.model == "hpg_hier_octamer"].to_string(index=False))
    print("\nOctamer convention medians:")
    print(convention_medians.to_string(index=False))


if __name__ == "__main__":
    main()
