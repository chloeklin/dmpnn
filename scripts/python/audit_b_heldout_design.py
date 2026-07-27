from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "ea_ip.csv"
META_DIR = ROOT / "metadata" / "splits"
REPORT = ROOT / "analysis" / "model_diagnostics" / "_dataset_design_audit.md"
SEED = 42
TARGETS = {"EA": "EA vs SHE (eV)", "IP": "IP vs SHE (eV)"}
sys.path.insert(0, str(ROOT / "scripts" / "python"))
from utils import generate_a_held_out_splits


def r2(y: pd.Series | np.ndarray, p: pd.Series | np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    denom = np.square(y - y.mean()).sum()
    return float(1 - np.square(y - p).sum() / denom) if denom else float("nan")


def role_indices(df: pd.DataFrame, role: str, folds: list[list[str]]) -> list[dict]:
    values = df[f"smiles_{role}"].astype(str)
    records = []
    for fold, heldout in enumerate(folds):
        test = np.flatnonzero(values.isin(heldout))
        validation = np.flatnonzero(values.isin(folds[(fold + 1) % len(folds)]))
        train = np.flatnonzero(~values.isin(set(heldout) | set(folds[(fold + 1) % len(folds)])))
        records.append({"fold": fold, "train_indices": train.tolist(), "val_indices": validation.tolist(), "test_indices": test.tolist()})
    return records


def random_folds(values: list[str]) -> list[list[str]]:
    shuffled = np.array(sorted(values), dtype=object)
    np.random.default_rng(SEED).shuffle(shuffled)
    return [part.tolist() for part in np.array_split(shuffled, 9)]


def clustered_folds(values: list[str]) -> list[list[str]]:
    clusters: dict[str, list[str]] = {}
    for smi in sorted(values):
        mol = Chem.MolFromSmiles(smi)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else "INVALID"
        clusters.setdefault(scaffold or f"ACYCLIC::{smi}", []).append(smi)
    capacities = [76] * (len(values) % 9) + [75] * (9 - len(values) % 9)
    bins = [[] for _ in capacities]
    remaining = capacities.copy()
    rng = np.random.default_rng(SEED)
    items = list(clusters.values())
    rng.shuffle(items)
    items.sort(key=len, reverse=True)
    for members in items:
        pending = list(members)
        while pending:
            candidates = [i for i, rem in enumerate(remaining) if rem > 0]
            index = max(candidates, key=lambda i: (remaining[i], -len(bins[i])))
            take = min(len(pending), remaining[index])
            bins[index].extend(pending[:take])
            pending = pending[take:]
            remaining[index] -= take
    if any(remaining):
        raise AssertionError("clustered B fold allocation did not fill all capacities")
    return bins


def freeze(df: pd.DataFrame, name: str, folds: list[list[str]], method: str) -> list[dict]:
    records = role_indices(df, "B", folds)
    all_indices = set(range(len(df)))
    frozen = []
    for fold, record in enumerate(records):
        test, val, train = map(lambda key: np.asarray(record[key], dtype=int), ("test_indices", "val_indices", "train_indices"))
        if set(test) & set(val) or set(test) & set(train) or set(val) & set(train):
            raise AssertionError(f"B fold {fold} has row leakage")
        if set(test) | set(val) | set(train) != all_indices:
            raise AssertionError(f"B fold {fold} does not partition rows")
        test_b = sorted(folds[fold])
        val_b = sorted(folds[(fold + 1) % 9])
        train_b = sorted(set(df.smiles_B.astype(str)) - set(test_b) - set(val_b))
        frozen.append({"split": name, "fold": fold, "split_seed": SEED, "assignment": method, "n_train": len(train), "n_val": len(val), "n_test": len(test), "held_out_monomer_B": test_b, "validation_monomer_B": val_b, "train_monomer_B": train_b, "global_train_indices": train.tolist(), "global_val_indices": val.tolist(), "global_test_indices": test.tolist()})
    tests = [set(record["global_test_indices"]) for record in frozen]
    if any(tests[i] & tests[j] for i in range(9) for j in range(i)) or set().union(*tests) != all_indices:
        raise AssertionError(f"{name} test folds are not a disjoint full-data partition")
    (META_DIR / f"{name}.json").write_text(json.dumps({"split": name, "split_seed": SEED, "assignment": method, "validation_design": "fold k+1 serves as B-disjoint validation for test fold k", "folds": frozen}, indent=2) + "\n")
    return frozen


def factor_variance(df: pd.DataFrame, target: str, keys: list[str]) -> float:
    y = df[target]
    pred = df.groupby(keys)[target].transform("mean")
    return r2(y, pred)


def grouped_metrics(frame: pd.DataFrame) -> dict:
    key = ["smiles_A", "smiles_B", "fracA"]
    valid = frame.groupby(key).poly_type.nunique()
    matched = frame.merge(valid[valid >= 2].rename("n_arch"), on=key, how="inner")
    group = matched.groupby(key, as_index=False)[["y_true", "y_pred"]].mean()
    return {"group_mean_r2": r2(group.y_true, group.y_pred), "overall_r2": r2(frame.y_true, frame.y_pred), "mae": float(np.abs(frame.y_true - frame.y_pred).mean()), "bias": float((frame.y_pred - frame.y_true).mean())}


def null_metrics(df: pd.DataFrame, splits: list[dict], target: str, blind_role: str) -> list[dict]:
    rows = []
    keys = ["smiles_B", "fracA", "poly_type"] if blind_role == "A" else ["smiles_A", "fracA", "poly_type"]
    for record in splits:
        train_indices = record["train_indices"] if "train_indices" in record else record["global_train_indices"]
        test_indices = record["test_indices"] if "test_indices" in record else record["global_test_indices"]
        train = df.iloc[train_indices]
        test = df.iloc[test_indices].copy()
        lookup = train.groupby(keys)[target].mean()
        global_mean = train[target].mean()
        test["y_true"] = test[target]
        test["y_pred"] = [lookup.get(tuple(row), global_mean) for row in test[keys].itertuples(index=False, name=None)]
        for null, prediction in ((f"{blind_role}-blind", test.y_pred), ("global-mean", np.full(len(test), global_mean))):
            data = test.copy()
            data.y_pred = prediction
            rows.append({"fold": record["fold"], "null": null, **grouped_metrics(data)})
    return rows


def fingerprints(values: list[str]) -> dict[str, object]:
    return {smi: AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2, nBits=2048) for smi in values}


def novelty(folds: list[dict], role: str, fp: dict[str, object], df: pd.DataFrame) -> list[dict]:
    results = []
    for record in folds:
        held = record.get(f"held_out_monomer_{role}")
        train = record.get(f"train_monomer_{role}")
        if held is None:
            held = sorted(df.iloc[record["test_indices"]][f"smiles_{role}"].astype(str).unique())
            train = sorted(df.iloc[record["train_indices"]][f"smiles_{role}"].astype(str).unique())
        scores = [max(DataStructs.TanimotoSimilarity(fp[item], fp[other]) for other in train) for item in held]
        results.append({"fold": record["fold"], "min": float(np.min(scores)), "median": float(np.median(scores)), "max": float(np.max(scores))})
    return results


def table(rows: list[dict], cols: list[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(f"{row.get(col, ''):.5f}" if isinstance(row.get(col), (float, np.floating)) else str(row.get(col, '')) for col in cols) + " |")
    return "\n".join(out)


def main() -> None:
    df = pd.read_csv(DATA)
    b_values = sorted(df.smiles_B.astype(str).unique())
    random_records = freeze(df, "monomer_b_heldout", random_folds(b_values), "random")
    clustered_records = freeze(df, "monomer_b_heldout_clustered", clustered_folds(b_values), "Murcko scaffold balanced")
    a_train, a_val, a_test, held_a = generate_a_held_out_splits(df.smiles_A.astype(str).values, len(df), seed=SEED, n_splits=9, protocol="leave_one_A_out")
    a_records = [{"fold": i, "train_indices": np.asarray(a_train[i], dtype=int).tolist(), "val_indices": np.asarray(a_val[i], dtype=int).tolist(), "test_indices": np.asarray(a_test[i], dtype=int).tolist(), "held_out_monomer_A": [held_a[i]], "validation_monomer_A": sorted(df.iloc[a_val[i]].smiles_A.astype(str).unique()), "train_monomer_A": sorted(df.iloc[a_train[i]].smiles_A.astype(str).unique())} for i in range(9)]
    pair_counts = df.groupby(["smiles_A", "smiles_B"]).size()
    combo_counts = df.groupby(["fracA", "fracB", "poly_type"]).size().reset_index(name="count")
    design = {"rows": len(df), "unique_A": df.smiles_A.nunique(), "unique_B": df.smiles_B.nunique(), "roles_disjoint": not bool(set(df.smiles_A.astype(str)) & set(df.smiles_B.astype(str))), "pair_count_min": int(pair_counts.min()), "pair_count_median": float(pair_counts.median()), "pair_count_max": int(pair_counts.max()), "n_pairs": len(pair_counts)}
    variance = []
    for short, target in TARGETS.items():
        within = 1 - factor_variance(df, target, ["smiles_A", "smiles_B", "fracA"])
        variance.append({"target": short, "A_identity": factor_variance(df, target, ["smiles_A"]), "B_identity": factor_variance(df, target, ["smiles_B"]), "A_plus_B": factor_variance(df, target, ["smiles_A", "smiles_B"]), "fracA": factor_variance(df, target, ["fracA"]), "poly_type": factor_variance(df, target, ["poly_type"]), "within_A_B_fracA": within})
    fp_a = fingerprints(sorted(df.smiles_A.astype(str).unique()))
    fp_b = fingerprints(b_values)
    a_novelty = novelty(a_records, "A", fp_a, df)
    random_novelty = novelty(random_records, "B", fp_b, df)
    clustered_novelty = novelty(clustered_records, "B", fp_b, df)
    duplicate_counts = []
    duplicate_pairs = []
    for record in random_records:
        fold = record["fold"]
        held = record["held_out_monomer_B"]
        train = record["train_monomer_B"]
        maxima = []
        for held_smi in held:
            best_smi, similarity = max(
                ((train_smi, DataStructs.TanimotoSimilarity(fp_b[held_smi], fp_b[train_smi])) for train_smi in train),
                key=lambda item: item[1],
            )
            maxima.append(similarity)
            if similarity >= 0.99:
                duplicate_pairs.append({"fold": fold, "held_out_smiles_B": held_smi, "training_smiles_B": best_smi, "max_tanimoto": similarity})
        duplicate_counts.append({"fold": fold, "n_heldout_B": len(held), "n_max_ge_0_99": int(np.sum(np.asarray(maxima) >= 0.99)), "n_max_ge_0_95": int(np.sum(np.asarray(maxima) >= 0.95)), "n_max_ge_0_90": int(np.sum(np.asarray(maxima) >= 0.90))})
    null_rows = []
    for label, records, role in (("A-heldout", a_records, "A"), ("B-heldout random", random_records, "B"), ("B-heldout clustered", clustered_records, "B")):
        for short, target in TARGETS.items():
            for row in null_metrics(df, records, target, role):
                row.update({"split": label, "target": short})
                null_rows.append(row)
    null_summary = []
    for (split, target, null), group in pd.DataFrame(null_rows).groupby(["split", "target", "null"]):
        null_summary.append({"split": split, "target": target, "null": null, **{f"{metric}_{stat}": float(group[metric].agg(stat)) for metric in ["group_mean_r2", "overall_r2", "mae", "bias"] for stat in ["median", "mean"]}})
    headroom = [{"split": row["split"], "target": row["target"], "null": row["null"], "metric": metric, "null_floor_median": row[f"{metric}_median"], "headroom_to_1": 1 - row[f"{metric}_median"]} for row in null_summary for metric in ["group_mean_r2", "overall_r2"]]
    a_means = df.groupby("smiles_A")[list(TARGETS.values())].mean()
    b_means = df.groupby("smiles_B")[list(TARGETS.values())].mean()
    a_mean_rows = [{"smiles_A": item, "EA_mean": float(row[TARGETS["EA"]]), "IP_mean": float(row[TARGETS["IP"]])} for item, row in a_means.iterrows()]
    lines = ["# Dataset Design Audit", "", "## 0.1 Design", "", f"Rows: `{design['rows']}`. Unique A: `{design['unique_A']}`. Unique B: `{design['unique_B']}`. A and B sets disjoint: `{design['roles_disjoint']}`.", "", f"The factorial claim `9 × 682 × 7 = 42,966` is {'confirmed' if design['rows'] == 9 * 682 * 7 else 'refuted'}. All `{design['n_pairs']}` A/B pairs are present with per-pair rows min/median/max = `{design['pair_count_min']}` / `{design['pair_count_median']:.0f}` / `{design['pair_count_max']}`.", "", "Distinct `(fracA, fracB, poly_type)` cells across the full dataset:", "", table(combo_counts.to_dict('records'), ["fracA", "fracB", "poly_type", "count"]), "", f"Rows per A min/median/max: `{df.groupby('smiles_A').size().min()}` / `{df.groupby('smiles_A').size().median():.0f}` / `{df.groupby('smiles_A').size().max()}`. Rows per B min/median/max: `{df.groupby('smiles_B').size().min()}` / `{df.groupby('smiles_B').size().median():.0f}` / `{df.groupby('smiles_B').size().max()}`.", "", "## 0.2 Signal axis", "", table(variance, ["target", "A_identity", "B_identity", "A_plus_B", "fracA", "poly_type", "within_A_B_fracA"]), "", "Mean EA and IP by A monomer:", "", table(a_mean_rows, ["smiles_A", "EA_mean", "IP_mean"]), "", "Mean target ranges by monomer role:", "", table([{ "target": short, "A_mean_min": float(a_means[target].min()), "A_mean_max": float(a_means[target].max()), "A_mean_spread": float(a_means[target].max()-a_means[target].min()), "B_mean_min": float(b_means[target].min()), "B_mean_max": float(b_means[target].max()), "B_mean_spread": float(b_means[target].max()-b_means[target].min())} for short, target in TARGETS.items()], ["target", "A_mean_min", "A_mean_max", "A_mean_spread", "B_mean_min", "B_mean_max", "B_mean_spread"]), "", "## 0.3 Frozen B-heldout splits", "", "Random and Murcko-scaffold-balanced metadata are frozen at split seed 42. Fold k uses fold k+1 as validation, so validation and test B identities are disjoint and both fixed across model seeds. This leaves 530–532 B monomers for training and yields validation/test rows near the A-heldout 4,774-row size.", "", table([{"assignment": "random", "n_train_min": min(x['n_train'] for x in random_records), "n_val_min": min(x['n_val'] for x in random_records), "n_test_min": min(x['n_test'] for x in random_records), "n_test_max": max(x['n_test'] for x in random_records)}, {"assignment": "clustered", "n_train_min": min(x['n_train'] for x in clustered_records), "n_val_min": min(x['n_val'] for x in clustered_records), "n_test_min": min(x['n_test'] for x in clustered_records), "n_test_max": max(x['n_test'] for x in clustered_records)}], ["assignment", "n_train_min", "n_val_min", "n_test_min", "n_test_max"]), "", "The existing A-heldout generator uses one whole distinct A identity for validation: it excludes both the test A and a second A from training, producing 33,418 train / 4,774 validation / 4,774 test rows per fold.", "", "## 0.4 Role-matched novelty", "", "Morgan r=2, 2048-bit maximum Tanimoto to the actual role-matched training identities. This reuses the fingerprint convention in `analysis/diagnostics/novelty.py`; the computation is generalized here to frozen B folds.", "", "A-heldout:", "", table(a_novelty, ["fold", "min", "median", "max"]), "", "Random B-heldout:", "", table(random_novelty, ["fold", "min", "median", "max"]), "", "### Near-duplicate random B monomers", "", "Counts are held-out B monomers whose maximum Morgan similarity to the fold's B-training identities reaches each threshold. These identities remain in the frozen split and must be reported both in full-fold and filtered (`max Tanimoto < 0.95`) Step-1 metrics.", "", table(duplicate_counts, ["fold", "n_heldout_B", "n_max_ge_0_99", "n_max_ge_0_95", "n_max_ge_0_90"]), "", "Morgan-identical-or-near-identical (`≥0.99`) held-out/training pairs:", "", table(duplicate_pairs, ["fold", "held_out_smiles_B", "training_smiles_B", "max_tanimoto"]), "", "Clustered B-heldout:", "", table(clustered_novelty, ["fold", "min", "median", "max"]), "", "## 0.5 Null floors", "", "All nulls use train-only lookup means and the same matched group key `(smiles_A, smiles_B, fracA)` with at least two `poly_type` values. The role-blind null is A-blind for A-heldout and B-blind for B-heldout; global-mean is an absolute reference.", "", table(null_rows, ["split", "target", "fold", "null", "group_mean_r2", "overall_r2", "mae", "bias"]), "", "Median and mean across folds:", "", table(null_summary, list(null_summary[0].keys())), "", "## 0.6 Verdict", "", "Headroom is `1 - median null R²`; values near zero are degenerate for that metric.", "", table(headroom, ["split", "target", "null", "metric", "null_floor_median", "headroom_to_1"]), "", "The A-heldout EA chemistry metric is demonstrably constrained: its A-blind median floor is 0.67571 and individual folds reach 0.96114. This directly qualifies the EA chemistry headline in `variant_results_report.md` (the sections titled `What changed / what flipped` and `Headline`): current A-heldout group-mean R² rankings cannot by themselves establish useful unseen-chemistry learning.", "", "Random B-heldout is worth GPU for both EA and IP. Its B-blind group-mean floors are 0.41966 (EA) and 0.56042 (IP), leaving 0.58034 and 0.43958 R² headroom. The B-blind floor does not collapse to the global mean because it averages roughly 530 seen B identities conditional on A, and A identity itself explains substantial variance. Clustered B-heldout is also viable and is chemically harder by its lower median nearest-neighbor similarities, but it should remain a follow-up until random B-heldout establishes the baseline comparison.", "", "Architecture variance within `(A, B, fracA)` is 0.979% for EA and 1.459% for IP, confirming the known 1–4% scale. Architecture-recovery metrics remain meaningful but require the Step 1 paired comparison; no existing model ranking is changed by Step 0 alone.", ""]
    REPORT.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
