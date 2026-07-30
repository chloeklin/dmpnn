"""R3 analysis — clustered B-heldout regeneration.

Differs from the A-heldout analysis in three ways that matter:

1. Null floors are the **B-blind** floors for the clustered split, taken per fold
   from `_dataset_design_audit.md` (not a median across folds).

2. **The nine folds are not exchangeable.** Murcko clustering of the 682 B monomers
   yields 112 scaffolds, but one has 317 members (46%) and another 109 (16%). The
   capacity-balanced packer therefore splits those two families across folds. The
   result is two structurally different kinds of fold:

       group S (within-scaffold)  - the held-out monomers share a scaffold whose
                                    other members remain in training
       group D (cross-scaffold)   - the held-out monomers are whole small scaffold
                                    families, absent from training

   Fold membership is DERIVED here from the frozen split plus scaffolds, not
   hard-coded, and the derivation is printed in the report. Paired sign tests are
   run within each group; the pooled nine-fold test is reported only as secondary,
   with the heterogeneity stated.

3. Metrics are reported twice: full folds, and filtered to drop held-out B monomers
   with a training near-duplicate at Tanimoto >= 0.95.

The split is capacity-balanced Murcko scaffold packing. It is NOT scaffold-disjoint.

Usage:
    python scripts/python/analyze_regen_v1_r3.py [--partial]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, linregress
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation.metrics import compute_copolymer_metrics
from analyze_regen_v1 import (
    markdown,
    read_markdown_table,
    holm_adjust,
    load_metrics,
    flag_undertrained,
    UNDERTRAINED_BEST_EPOCH_THRESHOLD,
)

# `flag_undertrained` still adds the per-run `undertrained` diagnostic column below, but
# this module no longer builds or writes an "*_excluding_undertrained" variant of any
# table. See the comment above `UNDERTRAINED_BEST_EPOCH_THRESHOLD` in analyze_regen_v1.py:
# `best_epoch` is an epoch count, and epoch counts are not comparable across the
# batch-size boundary between HPG (batch 64) and wDMPNN (batch 512), so a fixed epoch
# cutoff flagged HPG and wDMPNN cells at very different rates and would have dropped the
# three-seed replicate unit asymmetrically had it been used to exclude runs.

DATA_PATH = ROOT / "data" / "ea_ip.csv"
PREDICTION_DIR = ROOT / "predictions" / "regen_v1" / "ea_ip_lomo_b_clustered"
FROZEN_SPLIT = ROOT / "metadata" / "splits" / "monomer_b_heldout_clustered.json"
DESIGN_AUDIT = ROOT / "analysis" / "model_diagnostics" / "_dataset_design_audit.md"
OUTPUT_PATH = ROOT / "analysis" / "model_diagnostics" / "_regen_v1_r3_results.md"

SPLIT = "monomer_b_heldout_clustered"
MODELS = ("hpg_hier", "wdmpnn", "hpg_hier_octamer", "hpg_hier_junction")
TARGETS = {"EA": "EA_vs_SHE_eV", "IP": "IP_vs_SHE_eV"}
SEEDS = (42, 43, 44)
FOLDS = tuple(range(9))
METRICS = ("group_mean_r2", "delta_r2", "ordering", "overall_r2", "mae", "rmse",
           "group_mean_rmse", "mean_signed_bias", "compression_ratio")
COMPARISON_METRICS = ("group_mean_r2", "delta_r2", "ordering", "overall_r2", "mae", "rmse")

NEAR_DUPLICATE_TANIMOTO = 0.95
WITHIN_SCAFFOLD_SHARE_THRESHOLD = 0.5
HOMOGENEOUS_SCAFFOLD_MAX = 2
MORGAN_RADIUS = 2
MORGAN_BITS = 2048


def prediction_path(model: str, target: str, fold: int, seed: int) -> Path:
    return PREDICTION_DIR / f"ea_ip__{TARGETS[target]}__{model}__{SPLIT}__fold{fold}__s{seed}.npz"


def null_floors() -> pd.DataFrame:
    floor = read_markdown_table(DESIGN_AUDIT, {"split", "target", "fold", "null", "group_mean_r2", "overall_r2", "mae"})
    floor = floor[(floor["split"] == "B-heldout clustered") & (floor["null"] == "B-blind")].copy()
    if floor.empty:
        raise SystemExit("Could not locate B-heldout clustered / B-blind floors in the design audit")
    floor["fold"] = pd.to_numeric(floor.fold)
    floor["null_group_mean_r2"] = pd.to_numeric(floor.group_mean_r2)
    floor["null_overall_r2"] = pd.to_numeric(floor.overall_r2)
    floor["null_mae"] = pd.to_numeric(floor.mae)
    keep = ["target", "fold", "null_group_mean_r2", "null_overall_r2", "null_mae"]
    if "rmse" in floor.columns:                     # present once the audit is re-run
        floor["null_rmse"] = pd.to_numeric(floor.rmse)
        keep.append("null_rmse")
    return floor[keep]


# --------------------------------------------------------------------------- #
# Scaffold structure: fold grouping and near-duplicate identification
# --------------------------------------------------------------------------- #

def scaffold_structure() -> tuple[pd.DataFrame, dict[int, set[str]], dict]:
    """Return per-fold scaffold composition, near-duplicate sets, and cluster stats.

    Requires RDKit. If unavailable the caller falls back to unstratified,
    unfiltered reporting and says so explicitly in the report.
    """
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    folds_json = json.loads(FROZEN_SPLIT.read_text())["folds"]

    all_b = sorted({smiles for fold in folds_json for smiles in fold["held_out_monomer_B"]})
    mols = {smiles: Chem.MolFromSmiles(smiles) for smiles in all_b}
    scaffold_of = {}
    for smiles, mol in mols.items():
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else "INVALID"
        scaffold_of[smiles] = scaffold or f"ACYCLIC::{smiles}"
    fingerprint_of = {
        smiles: AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_BITS)
        for smiles, mol in mols.items()
        if mol is not None
    }

    cluster_sizes = pd.Series(list(scaffold_of.values())).value_counts()
    stats = {
        "n_monomers": len(all_b),
        "n_scaffolds": int(cluster_sizes.size),
        "largest_cluster": int(cluster_sizes.iloc[0]),
        "second_cluster": int(cluster_sizes.iloc[1]) if cluster_sizes.size > 1 else 0,
        "singletons": int((cluster_sizes == 1).sum()),
    }
    stats["top_two_share"] = (stats["largest_cluster"] + stats["second_cluster"]) / stats["n_monomers"]

    rows = []
    near_duplicates: dict[int, set[str]] = {}
    for fold in folds_json:
        index = int(fold["fold"])
        held = list(fold["held_out_monomer_B"])
        train = list(fold["train_monomer_B"])
        train_scaffolds = {scaffold_of[smiles] for smiles in train}
        shared = [smiles for smiles in held if scaffold_of[smiles] in train_scaffolds]

        train_fps = [fingerprint_of[s] for s in train if s in fingerprint_of]
        contaminated = set()
        max_similarities = []
        for smiles in held:
            if smiles not in fingerprint_of or not train_fps:
                continue
            similarities = DataStructs.BulkTanimotoSimilarity(fingerprint_of[smiles], train_fps)
            highest = max(similarities)
            max_similarities.append(highest)
            if highest >= NEAR_DUPLICATE_TANIMOTO:
                contaminated.add(smiles)
        near_duplicates[index] = contaminated

        rows.append({
            "fold": index,
            "n_held_out_B": len(held),
            "n_distinct_scaffolds": len({scaffold_of[s] for s in held}),
            "n_with_same_scaffold_in_train": len(shared),
            "same_scaffold_share": len(shared) / len(held) if held else np.nan,
            "median_max_tanimoto": float(np.median(max_similarities)) if max_similarities else np.nan,
            "max_max_tanimoto": float(np.max(max_similarities)) if max_similarities else np.nan,
            "n_near_duplicates_ge_0.95": len(contaminated),
        })

    composition = pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)
    composition["fold_group"] = np.where(
        composition.same_scaffold_share > WITHIN_SCAFFOLD_SHARE_THRESHOLD, "S_within_scaffold", "D_cross_scaffold"
    )
    # A group-D fold can still be chemically homogeneous: fold 5 holds out 76 members of a
    # single scaffold family whose remaining members sit in its validation fold, not training.
    # It is legitimately cross-scaffold, but it samples ONE chemistry where folds 7 and 8 sample
    # 28-29. Flag it so it can be excluded as a sensitivity check rather than silently pooled.
    composition["scaffold_homogeneous"] = composition.n_distinct_scaffolds <= HOMOGENEOUS_SCAFFOLD_MAX
    return composition, near_duplicates, stats


# --------------------------------------------------------------------------- #
# Cells, pooled context, comparisons
# --------------------------------------------------------------------------- #

def build_cells(detail_subset: pd.DataFrame, arrays: dict, df: pd.DataFrame,
                drop_b: dict[int, set[str]] | None = None) -> tuple[pd.DataFrame, dict]:
    """Three-seed averaged metrics per (model, target, fold).

    `drop_b` optionally removes test rows whose smiles_B is a near-duplicate of a
    training monomer, giving the filtered variant of every metric.
    """
    b_values = df["smiles_B"].astype(str).to_numpy()
    cell_rows = []
    averaged_group_means = {}
    for model in MODELS:
        for target in TARGETS:
            for fold in FOLDS:
                subset = detail_subset[(detail_subset.model == model) & (detail_subset.target == target) & (detail_subset.fold == fold)]
                if subset.empty:
                    continue
                payloads = [arrays[(model, target, fold, seed)] for seed in subset.seed.tolist()]
                first = payloads[0]
                if any(not np.array_equal(first["indices"], item["indices"]) or not np.array_equal(first["y_true"], item["y_true"]) for item in payloads[1:]):
                    raise AssertionError(f"Rows differ across seeds for {model} {target} fold {fold}")
                averaged = np.mean(np.stack([item["y_pred"] for item in payloads]), axis=0)
                y_true, indices = first["y_true"], first["indices"]
                if drop_b:
                    keep = ~np.isin(b_values[indices], list(drop_b.get(fold, set())))
                    y_true, indices, averaged = y_true[keep], indices[keep], averaged[keep]
                    if indices.size == 0:
                        continue
                metrics, group_means = compute_copolymer_metrics(df, y_true, averaged, indices)
                averaged_group_means[(model, target, fold)] = group_means.assign(fold=fold)
                result = {"model": model, "target": target, "fold": fold, "n_test_rows": int(indices.size)}
                for metric in METRICS:
                    result[metric] = metrics[metric]
                    if not drop_b:
                        result[f"{metric}_seed_mean"] = float(subset[metric].mean())
                        result[f"{metric}_seed_sd"] = float(subset[metric].std(ddof=1))
                cell_rows.append(result)
    if not cell_rows:
        return pd.DataFrame(), {}
    cells = pd.DataFrame(cell_rows).merge(null_floors(), on=["target", "fold"], validate="many_to_one")
    cells["beats_null_floor"] = cells.group_mean_r2 > cells.null_group_mean_r2
    # Skill score against the null: 1 - MSE_model/MSE_null, computed from R2 because
    # both share the same denominator within a fold. 0 = no better than a predictor that
    # ignores the held-out monomer; 1 = perfect. Comparable across folds and targets in a
    # way raw R2 is not, because it removes each fold's own variance scale.
    cells["skill_group_mean"] = (cells.group_mean_r2 - cells.null_group_mean_r2) / (1.0 - cells.null_group_mean_r2)
    cells["skill_overall"] = (cells.overall_r2 - cells.null_overall_r2) / (1.0 - cells.null_overall_r2)
    cells["skill_vs_null"] = cells["skill_group_mean"]              # legacy alias
    cells["null_floor_headroom_used"] = cells["skill_group_mean"]   # legacy alias
    return cells, averaged_group_means


def build_pooled(cells: pd.DataFrame, detail_subset: pd.DataFrame, arrays: dict,
                 averaged_group_means: dict, folds: tuple[int, ...]) -> pd.DataFrame:
    pooled_rows = []
    for model in MODELS:
        for target in TARGETS:
            frames = [
                averaged_group_means[(model, target, fold)].assign(group=lambda v: v.fold.astype(str) + "||" + v.group)
                for fold in folds if (model, target, fold) in averaged_group_means
            ]
            if not frames:
                continue
            pooled_groups = pd.concat(frames, ignore_index=True)
            fold_stats = []
            for fold in folds:
                subset = detail_subset[(detail_subset.model == model) & (detail_subset.target == target) & (detail_subset.fold == fold)]
                if subset.empty:
                    continue
                payloads = [arrays[(model, target, fold, seed)] for seed in subset.seed.tolist()]
                prediction = np.mean(np.stack([item["y_pred"] for item in payloads]), axis=0)
                fold_stats.append({
                    "fold": fold,
                    "true_mean": payloads[0]["y_true"].mean(),
                    "pred_mean": prediction.mean(),
                    "bias": (prediction - payloads[0]["y_true"]).mean(),
                })
            if len(fold_stats) < 2:
                continue
            fold_stats = pd.DataFrame(fold_stats)
            slope, intercept, *_ = linregress(fold_stats.true_mean, fold_stats.pred_mean)
            subset_cells = cells[(cells.model == model) & (cells.target == target) & (cells.fold.isin(folds))]
            pooled_rows.append({
                "model": model,
                "target": target,
                "n_folds": len(fold_stats),
                "pooled_group_mean_r2": r2_score(pooled_groups.y_true, pooled_groups.y_pred),
                "fold_placement_r2": r2_score(fold_stats.true_mean, fold_stats.pred_mean),
                "fold_placement_slope": slope,
                "fold_placement_intercept": intercept,
                "fold_bias_sd": fold_stats.bias.std(ddof=0),
                "mean_within_fold_compression_ratio": subset_cells.compression_ratio.mean(),
            })
    return pd.DataFrame(pooled_rows)


def build_comparisons(cells: pd.DataFrame, folds: tuple[int, ...], label: str, holm: bool = True) -> pd.DataFrame:
    """Paired per-fold comparisons against hpg_hier, restricted to `folds`."""
    rows = []
    for target in TARGETS:
        reference = cells[(cells.model == "hpg_hier") & (cells.target == target) & (cells.fold.isin(folds))].set_index("fold")
        for model in MODELS[1:]:
            candidate = cells[(cells.model == model) & (cells.target == target) & (cells.fold.isin(folds))].set_index("fold")
            common = reference.index.intersection(candidate.index)
            if len(common) == 0:
                continue
            ref, cand = reference.loc[common], candidate.loc[common]
            for metric in COMPARISON_METRICS:
                differences = cand[metric] - ref[metric]
                better = differences < 0 if metric == "mae" else differences > 0
                worse = differences > 0 if metric == "mae" else differences < 0
                wins, losses = int(better.sum()), int(worse.sum())
                non_ties = wins + losses
                p_value = float(binomtest(wins, non_ties, 0.5).pvalue) if non_ties else 1.0
                row = {
                    "fold_group": label,
                    "n_folds": len(common),
                    "model": model,
                    "target": target,
                    "metric": metric,
                    "median_paired_difference": differences.median(),
                    "wins": wins,
                    "losses": losses,
                    "exact_sign_p": p_value,
                    "min_attainable_p": float(binomtest(len(common), len(common), 0.5).pvalue) if len(common) else np.nan,
                }
                if f"{metric}_seed_sd" in cand.columns:
                    seed_sd = np.maximum(cand[f"{metric}_seed_sd"], ref[f"{metric}_seed_sd"])
                    row["folds_smaller_than_measured_seed_sd"] = int((differences.abs() < seed_sd).sum())
                rows.append(row)
    comparisons = pd.DataFrame(rows)
    if holm and not comparisons.empty:
        comparisons["holm_p"] = holm_adjust(comparisons.exact_sign_p.tolist())
    return comparisons


# --------------------------------------------------------------------------- #

def main() -> None:
    partial = "--partial" in sys.argv
    df = pd.read_csv(DATA_PATH)

    inventory_rows = [
        {"model": model, "target": target, "fold": fold, "seed": seed,
         "available": prediction_path(model, target, fold, seed).is_file(),
         "sidecar": prediction_path(model, target, fold, seed).with_suffix(".config.json").is_file()}
        for model in MODELS for target in TARGETS for fold in FOLDS for seed in SEEDS
    ]
    inventory = pd.DataFrame(inventory_rows)
    inventory["complete"] = inventory.available & inventory.sidecar
    n_complete = int(inventory.complete.sum())

    try:
        composition, near_duplicates, cluster_stats = scaffold_structure()
        rdkit_note = ""
    except ImportError:
        composition, near_duplicates, cluster_stats = pd.DataFrame(), {}, {}
        rdkit_note = ("**RDKit unavailable — fold stratification and near-duplicate filtering were skipped.** "
                      "Results below pool structurally different folds and must not be read as a single population.")

    if n_complete < len(inventory) and not partial:
        OUTPUT_PATH.write_text("\n".join([
            "# R3 — clustered B-heldout regeneration",
            "",
            f"## Status: pending — {n_complete}/{len(inventory)} runs complete",
            "",
            "Re-run with `--partial` to analyse the subset that has landed.",
            "",
            "## Split structure (independent of results)",
            "",
            split_structure_section(composition, cluster_stats, rdkit_note),
            "",
            "## Missing cells",
            "",
            markdown(inventory[~inventory.complete]),
            "",
        ]))
        print(f"Wrote pending report: {OUTPUT_PATH}  ({n_complete}/{len(inventory)} runs)")
        return

    run_rows, arrays, split_hashes = [], {}, {}
    for row in inventory[inventory.complete].itertuples(index=False):
        path = prediction_path(row.model, row.target, row.fold, row.seed)
        metrics, _, payload = load_metrics(df, path)
        sidecar = json.loads(path.with_suffix(".config.json").read_text())
        if sidecar["epochs_actually_run"] <= 0 or sidecar["wall_time_seconds"] <= 60:
            raise AssertionError(f"Run does not look trained: {path}")
        final_metrics, _, _ = load_metrics(df, path, "y_pred_final")
        run_rows.append({
            "model": row.model, "target": row.target, "fold": row.fold, "seed": row.seed, **metrics,
            "final_mae": final_metrics["mae"], "final_minus_best_mae": final_metrics["mae"] - metrics["mae"],
            "epochs": sidecar["epochs_actually_run"], "best_epoch": sidecar["best_epoch"],
            "best_val_loss": sidecar["best_val_loss"], "wall_time_seconds": sidecar["wall_time_seconds"],
        })
        arrays[(row.model, row.target, row.fold, row.seed)] = payload
        split_hashes[(row.model, row.target, row.fold, row.seed)] = payload["split_hash"]

    detail = flag_undertrained(pd.DataFrame(run_rows))

    hash_rows = []
    for model in MODELS:
        for target in TARGETS:
            for fold in FOLDS:
                present = [split_hashes[k] for k in split_hashes if k[:3] == (model, target, fold)]
                if len(present) > 1:
                    hash_rows.append({"model": model, "target": target, "fold": fold,
                                      "n_seeds": len(present), "identical": len(set(present)) == 1})
    hash_check = pd.DataFrame(hash_rows)
    if not hash_check.empty and not hash_check.identical.all():
        raise AssertionError(f"Split hashes differ across seeds:\n{hash_check[~hash_check.identical]}")

    flag_counts = detail.groupby("model").undertrained.sum().astype(int).reset_index()
    flag_counts.columns = ["model", "flagged_runs"]

    cells, averaged_group_means = build_cells(detail, arrays, df)
    cells_filtered, _ = build_cells(detail, arrays, df, drop_b=near_duplicates) if near_duplicates else (pd.DataFrame(), {})

    if not composition.empty:
        group_s = tuple(composition.loc[composition.fold_group == "S_within_scaffold", "fold"].astype(int))
        group_d = tuple(composition.loc[composition.fold_group == "D_cross_scaffold", "fold"].astype(int))
        group_d_diverse = tuple(composition.loc[(composition.fold_group == "D_cross_scaffold") & (~composition.scaffold_homogeneous), "fold"].astype(int))
    else:
        group_s, group_d, group_d_diverse = (), (), ()

    comparison_blocks = []
    if group_s:
        comparison_blocks.append(("S — within-scaffold folds " + str(list(group_s)), build_comparisons(cells, group_s, "S_within_scaffold")))
    if group_d:
        comparison_blocks.append(("D — cross-scaffold folds " + str(list(group_d)), build_comparisons(cells, group_d, "D_cross_scaffold")))
    if group_d_diverse and set(group_d_diverse) != set(group_d):
        dropped = sorted(set(group_d) - set(group_d_diverse))
        comparison_blocks.append((
            f"D sensitivity — chemically diverse folds only {list(group_d_diverse)} (drops {dropped}, which sample ≤{HOMOGENEOUS_SCAFFOLD_MAX} scaffold(s))",
            build_comparisons(cells, group_d_diverse, "D_cross_scaffold_diverse_only"),
        ))
    comparison_blocks.append(("Pooled across all nine folds — SECONDARY ONLY", build_comparisons(cells, FOLDS, "pooled")))

    pooled_blocks = []
    if group_s:
        pooled_blocks.append(("S — within-scaffold", build_pooled(cells, detail, arrays, averaged_group_means, group_s)))
    if group_d:
        pooled_blocks.append(("D — cross-scaffold", build_pooled(cells, detail, arrays, averaged_group_means, group_d)))

    checkpoint_gap = detail.groupby("model", as_index=False).agg(
        runs=("final_minus_best_mae", "count"),
        final_minus_best_mae_mean=("final_minus_best_mae", "mean"),
        final_minus_best_mae_sd=("final_minus_best_mae", "std"),
    )

    stem = OUTPUT_PATH.with_suffix("")
    detail.to_csv(stem.with_name(stem.name + "_individual_runs.csv"), index=False)
    cells.to_csv(stem.with_name(stem.name + "_cells.csv"), index=False)
    if not cells_filtered.empty:
        cells_filtered.to_csv(stem.with_name(stem.name + "_cells_filtered.csv"), index=False)
    if not composition.empty:
        composition.to_csv(stem.with_name(stem.name + "_fold_composition.csv"), index=False)
    pd.concat([frame for _, frame in comparison_blocks], ignore_index=True).to_csv(
        stem.with_name(stem.name + "_comparisons.csv"), index=False)

    report = [
        "# R3 — clustered B-heldout regeneration",
        "",
        "**Convention:** all figures are the mean prediction of three seeds (42, 43, 44), applied "
        "identically to every model. Individual-seed SD is the error bar.",
        "",
        f"**Coverage:** {n_complete}/{len(inventory)} runs." + ("  *(partial analysis)*" if partial else ""),
        "",
        rdkit_note,
        "",
        "## Split structure — read before the results",
        "",
        split_structure_section(composition, cluster_stats, rdkit_note),
        "",
        "## Run-quality diagnostic",
        "",
        f"Each run is flagged **potentially undertrained** if `best_epoch < {UNDERTRAINED_BEST_EPOCH_THRESHOLD}`. "
        "This flag is reported per run as a diagnostic only. It is expressed in epochs, and epoch counts are "
        "not comparable across the batch-size boundary in this study: HPG models train at batch size 64 "
        "(~672 updates/epoch) while wDMPNN trains at batch size 512 (~84 updates/epoch), so the same epoch "
        "cutoff represents roughly 8x more gradient updates for HPG than for wDMPNN. No runs are excluded "
        "from any table in this report on the basis of this flag; every reported cell uses all three seeds.",
        "",
        markdown(flag_counts),
        "",
        "## Split-hash consistency across seeds",
        "",
        markdown(hash_check) if not hash_check.empty else "_Only one seed present per cell; nothing to compare._",
        "",
        "## Three-seed averaged cells",
        "",
        "`skill_group_mean` and `skill_overall` are skill scores against the fold's own null: "
        "1 - MSE_model/MSE_null. Zero means no better than a predictor that ignores the held-out "
        "monomer; one means perfect. Unlike raw R² these are comparable across folds and targets. "
        "`rmse` and `mae` are in eV and are the only absolute, chemically interpretable numbers here.",
        "",
        markdown(cells),
        "",
        "### Filtered — held-out B monomers with a training near-duplicate (Tanimoto ≥ 0.95) removed",
        "",
        markdown(cells_filtered) if not cells_filtered.empty else "_Not computed (RDKit unavailable)._",
        "",
        "## Paired per-fold comparisons against HPG-hier",
        "",
        "Signed differences are candidate minus HPG-hier; headlines are medians of paired per-fold "
        "differences. **Tests are run within fold group.** The pooled nine-fold test mixes two "
        "structurally different populations and is reported for reference only.",
        "",
    ]
    for label, frame in comparison_blocks:
        report += [f"### {label}", "", markdown(frame), ""]

    report += ["## Across-fold context, by fold group", ""]
    for label, frame in pooled_blocks:
        report += [f"### {label}", "", markdown(frame), ""]

    report += [
        "## Checkpoint gap (final model minus best checkpoint)",
        "",
        markdown(checkpoint_gap),
        "",
        "Positive values are what the model-selection bug would have cost had it not been fixed.",
        "",
    ]

    OUTPUT_PATH.write_text("\n".join(report))
    print(f"Wrote {OUTPUT_PATH}")


def split_structure_section(composition: pd.DataFrame, stats: dict, rdkit_note: str) -> str:
    if composition.empty:
        return rdkit_note or "_Scaffold structure unavailable._"
    lines = [
        "The clustered split is **capacity-balanced Murcko scaffold packing — it is not "
        "scaffold-disjoint.** Two scaffold families are large enough that the packer must split them "
        "across folds, so their members appear in both training and test.",
        "",
        f"- distinct Murcko scaffolds: **{stats['n_scaffolds']}** across **{stats['n_monomers']}** B monomers",
        f"- largest two clusters: **{stats['largest_cluster']}** and **{stats['second_cluster']}** members "
        f"(**{stats['top_two_share']:.1%}** of all B monomers)",
        f"- singleton scaffolds: **{stats['singletons']}**",
        "",
        "A strictly scaffold-disjoint nine-fold split is impossible with this monomer set: the largest "
        "family alone exceeds any balanced fold capacity. The consequence is that folds fall into two "
        "structurally different groups, derived below from the frozen split rather than assumed.",
        "",
        markdown(composition),
        "",
        f"Folds with more than {WITHIN_SCAFFOLD_SHARE_THRESHOLD:.0%} of held-out monomers sharing a scaffold "
        "with training are labelled **S (within-scaffold)** — they test generalisation to new substituents "
        "on familiar scaffolds. The remainder are **D (cross-scaffold)** — the held-out scaffolds are absent "
        "from training, so they test generalisation to genuinely new chemistry. These two groups answer "
        "different questions and are not exchangeable, so paired tests are run within group.",
        "",
        f"`scaffold_homogeneous` marks a fold that samples ≤ {HOMOGENEOUS_SCAFFOLD_MAX} distinct scaffold(s). "
        "Such a fold can still be cross-scaffold — its family's remaining members may sit in the validation "
        "fold rather than training — but it tests **one** chemistry where a diverse fold tests 26–29. Its "
        "result is a single chemical observation with correspondingly wide uncertainty, so group D is also "
        "reported with homogeneous folds dropped as a sensitivity check.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
