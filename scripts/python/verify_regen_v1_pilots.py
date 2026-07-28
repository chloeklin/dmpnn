"""Pilot verification for regen_v1 R1 and R3 before full submission.

Reads prediction NPZs and their .config.json sidecars under
predictions/regen_v1/ea_ip_lomo and predictions/regen_v1/ea_ip_lomo_b_clustered,
then writes analysis/model_diagnostics/_pilot_verification.md.

This script does not submit jobs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.metrics import compute_copolymer_metrics

DATA_PATH = ROOT / "data" / "ea_ip.csv"
DESIGN_AUDIT = ROOT / "analysis" / "model_diagnostics" / "_dataset_design_audit.md"


def load_null_floors() -> dict[tuple[str, int], float]:
    """Return per-fold B-blind group-mean R2 floors for the B-heldout clustered split."""
    floors: dict[tuple[str, int], float] = {}
    in_table = False
    for line in DESIGN_AUDIT.read_text().splitlines():
        if line.startswith("| split |"):
            in_table = True
            continue
        if in_table and not line.strip():
            break
        if in_table and line.startswith("| ---"):
            continue
        if in_table:
            cells = [cell.strip() for cell in line.split("|")]
            # cells[0] is empty because markdown tables start with "|".
            if len(cells) >= 6 and cells[1] == "B-heldout clustered" and cells[4] == "B-blind":
                target = cells[2]
                fold = int(cells[3])
                group_mean_r2 = float(cells[5])
                floors[(target, fold)] = group_mean_r2
    return floors


NULL_FLOORS = load_null_floors()
R1_DIR = ROOT / "predictions" / "regen_v1" / "ea_ip_lomo"
R3_DIR = ROOT / "predictions" / "regen_v1" / "ea_ip_lomo_b_clustered"
OLD_R1_DIR = ROOT / "predictions" / "ea_ip_lomo"
B_SPLIT_METADATA = ROOT / "metadata" / "splits" / "monomer_b_heldout_clustered.json"
OUTPUT_PATH = ROOT / "analysis" / "model_diagnostics" / "_pilot_verification.md"

R1_MODELS = ("hpg_hier", "wdmpnn", "hpg_hier_octamer", "hpg_hier_junction", "hpg_hier_junction1")


def md_table(rows: list[dict], columns: list[str] | None = None) -> str:
    if not rows:
        return "_No rows._"
    if columns is None:
        columns = list(rows[0].keys())
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for row in rows:
        cells = []
        for c in columns:
            v = row.get(c, "")
            if isinstance(v, float):
                v = f"{v:.6f}"
            elif v is None:
                v = ""
            cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load_npz(path: Path, require_final: bool = True):
    with np.load(path, allow_pickle=True) as d:
        result = {
            "y_true": d["y_true"].astype(float).ravel(),
            "y_pred": d["y_pred"].astype(float).ravel(),
            "test_indices": d["test_indices"].astype(int).ravel(),
            "n_train": int(d["n_train"]),
            "n_val": int(d["n_val"]),
            "n_test": int(d["n_test"]),
            "split_type": str(d["split_type"]),
            "model": str(d["model"]),
            "target": str(d["target"]),
            "fold": int(d["fold"]),
            "seed": int(d["seed"]),
            "prediction_scale": str(d["prediction_scale"]),
        }
        if "y_pred_final" in d.files:
            result["y_pred_final"] = d["y_pred_final"].astype(float).ravel()
        elif require_final:
            raise KeyError(f"y_pred_final missing in {path}")
        if "split_indices_sha256" in d.files:
            result["split_indices_sha256"] = str(d["split_indices_sha256"])
        return result


def load_sidecar(path: Path) -> dict:
    return json.loads(path.read_text())


def find_r1_pilot_files() -> list[Path]:
    files = []
    for model in R1_MODELS:
        for seed in (42, 43):
            p = R1_DIR / f"ea_ip__EA_vs_SHE_eV__{model}__monomer_heldout__fold0__s{seed}.npz"
            if p.is_file():
                files.append(p)
    return files


def find_r3_pilot_files() -> list[Path]:
    files = []
    for fold in (0, 1, 2):
        p = R3_DIR / f"ea_ip__EA_vs_SHE_eV__hpg_hier__monomer_b_heldout_clustered__fold{fold}__s42.npz"
        if p.is_file():
            files.append(p)
    p = R3_DIR / "ea_ip__EA_vs_SHE_eV__wdmpnn__monomer_b_heldout_clustered__fold0__s42.npz"
    if p.is_file():
        files.append(p)
    return files


def old_r1_counterpart(npz_path: Path) -> Path | None:
    """Find the pre-regen counterpart for an R1 pilot file, if it exists."""
    parts = npz_path.stem.split("__")
    # stem: ea_ip__EA_vs_SHE_eV__hpg_hier__monomer_heldout__fold0__s42
    model = parts[2]
    target = parts[1]  # EA_vs_SHE_eV
    fold = parts[4]
    # Old files use two target naming conventions; try the explicit one first.
    candidates = [
        OLD_R1_DIR / f"ea_ip__{target}__{model}__monomer_heldout__{fold}__s42.npz",
        OLD_R1_DIR / f"ea_ip__EA vs SHE_eV__{model}__monomer_heldout__{fold}.npz",
        OLD_R1_DIR / f"ea_ip__EA vs SHE (eV)__{model}__monomer_heldout__{fold}.npz",
        OLD_R1_DIR / f"ea_ip__{target}__{model}__monomer_heldout__{fold}.npz",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def safe_ratio(num: float, denom: float) -> float:
    return float(num / denom) if denom else float("nan")


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    r1_files = find_r1_pilot_files()
    r3_files = find_r3_pilot_files()

    report_lines: list[str] = [
        "# Regen v1 Pilot Verification",
        "",
        f"Scope: {len(r1_files)} R1 pilot runs (monomer_heldout) and {len(r3_files)} R3 pilot runs (monomer_b_heldout_clustered).",
        "",
        "**This report does not authorise submission of the remaining 260 R1 or 212 R3 jobs.**",
        "",
    ]

    issues: list[str] = []
    warnings_list: list[str] = []

    # ------------------------------------------------------------------
    # A. DID THEY ACTUALLY TRAIN
    # ------------------------------------------------------------------
    report_lines += ["## A. Did they actually train", ""]
    training_rows = []
    for p in sorted(r1_files + r3_files):
        npz = load_npz(p)
        sidecar = load_sidecar(p.with_suffix(".config.json"))
        env = sidecar.get("runtime_environment", {})
        row = {
            "split": npz["split_type"],
            "model": npz["model"],
            "target": npz["target"],
            "fold": npz["fold"],
            "seed": npz["seed"],
            "epochs_run": sidecar.get("epochs_actually_run"),
            "best_epoch": sidecar.get("best_epoch"),
            "best_val_loss": sidecar.get("best_val_loss"),
            "wall_time_s": sidecar.get("wall_time_seconds"),
            "wall_time_h": safe_ratio(sidecar.get("wall_time_seconds", 0), 3600),
            "accelerator": env.get("accelerator"),
            "device": env.get("device_name"),
            "git_sha": sidecar.get("git_commit", "")[:16] if sidecar.get("git_commit") else "",
            "pbs_job_id": sidecar.get("pbs_job_id"),
        }
        training_rows.append(row)
        epochs = sidecar.get("epochs_actually_run", 0)
        wall = sidecar.get("wall_time_seconds", 0)
        if epochs <= 1:
            issues.append(f"{p.name}: near-zero epochs ({epochs}); training likely skipped")
        if wall < 300:
            issues.append(f"{p.name}: implausibly short wall time ({wall:.0f}s); training likely skipped")

    report_lines.append(md_table(training_rows))
    report_lines.append("")
    if issues:
        report_lines.append("**Training concerns:**")
        for issue in issues:
            report_lines.append(f"- {issue}")
        report_lines.append("")
    else:
        report_lines.append("No runs flagged for near-zero epochs or implausibly short wall time.")
        report_lines.append("")

    # ------------------------------------------------------------------
    # B. IS THE NEW CODE PATH ACTUALLY BEING USED
    # ------------------------------------------------------------------
    report_lines += ["## B. Is the new code path actually being used", ""]
    code_path_rows = []
    for p in sorted(r1_files + r3_files):
        npz = load_npz(p)
        sidecar = load_sidecar(p.with_suffix(".config.json"))
        pred_ckpt = sidecar.get("prediction_checkpoint", {})
        final_ckpt = sidecar.get("final_prediction_checkpoint", {})
        pred_hash = pred_ckpt.get("sha256", "")
        final_hash = final_ckpt.get("sha256", "")
        yp = npz["y_pred"]
        ypf = npz["y_pred_final"]
        identical = np.array_equal(yp, ypf)
        both_present = (
            yp.size > 0 and ypf.size > 0 and
            yp.shape == ypf.shape and
            len(pred_hash) == 64 and len(final_hash) == 64
        )
        hashes_differ = pred_hash != final_hash
        row = {
            "split": npz["split_type"],
            "model": npz["model"],
            "fold": npz["fold"],
            "seed": npz["seed"],
            "y_pred_shape": str(yp.shape),
            "y_pred_final_shape": str(ypf.shape),
            "both_present": both_present,
            "hashes_differ": hashes_differ,
            "ypred_identical": identical,
        }
        code_path_rows.append(row)
        if not both_present:
            issues.append(f"{p.name}: y_pred/y_pred_final/checkpoint hashes incomplete")
        if not hashes_differ:
            issues.append(f"{p.name}: prediction and final checkpoint SHA-256 are identical; best==final fix not exercised")
        if identical:
            issues.append(f"{p.name}: y_pred and y_pred_final are element-wise identical")

    report_lines.append(md_table(code_path_rows))
    report_lines.append("")

    # B.4: compare new R1 y_pred to old counterpart
    r1_old_rows = []
    for p in sorted(r1_files):
        npz_new = load_npz(p)
        old_p = old_r1_counterpart(p)
        if old_p is None:
            r1_old_rows.append({
                "model": npz_new["model"],
                "seed": npz_new["seed"],
                "old_file": "NOT FOUND",
                "max_abs_diff": None,
                "identical": None,
            })
            issues.append(f"{p.name}: old counterpart not found")
            continue
        npz_old = load_npz(old_p, require_final=False)
        new_y = npz_new["y_pred"]
        old_y = npz_old["y_pred"]
        if new_y.shape != old_y.shape:
            diff = float("nan")
            identical = False
        else:
            diff = float(np.max(np.abs(new_y - old_y)))
            identical = bool(np.array_equal(new_y, old_y))
        r1_old_rows.append({
            "model": npz_new["model"],
            "seed": npz_new["seed"],
            "old_file": old_p.name,
            "max_abs_diff": diff,
            "identical": identical,
        })
        if identical:
            issues.append(f"{p.name}: y_pred identical to old counterpart {old_p.name}")

    report_lines.append("### R1 pilot vs pre-regen counterparts")
    report_lines.append(md_table(r1_old_rows))
    report_lines.append("")

    # ------------------------------------------------------------------
    # C. SPLIT INTEGRITY
    # ------------------------------------------------------------------
    report_lines += ["## C. Split integrity", ""]

    # C.1 split hash consistency across seeds
    hash_rows = []
    hash_groups: dict[tuple, list[str]] = {}
    for p in sorted(r1_files + r3_files):
        npz = load_npz(p)
        key = (npz["split_type"], npz["model"], npz["target"], npz["fold"])
        hash_groups.setdefault(key, []).append(npz["split_indices_sha256"])

    multi_seed_keys = []
    for key, hashes in hash_groups.items():
        unique = set(hashes)
        split_type, model, target, fold = key
        entry = {
            "split": split_type,
            "model": model,
            "fold": fold,
            "seeds": len(hashes),
            "unique_hashes": len(unique),
            "consistent": ("True" if len(unique) == 1 else "False") if len(hashes) > 1 else "n/a",
        }
        hash_rows.append(entry)
        if len(hashes) > 1 and len(unique) != 1:
            issues.append(f"Split hashes differ across seeds for {key}")
        if len(hashes) > 1:
            multi_seed_keys.append(key)

    report_lines.append("### C.1 Split hash consistency across seeds")
    report_lines.append(md_table(hash_rows))
    if not multi_seed_keys:
        report_lines.append("")
        report_lines.append("_No (model, target, fold) cell in the pilot spans more than one seed, so a vacuous cross-seed consistency check is not reported._")
    report_lines.append("")

    # C.2 / C.3 R3 split integrity
    report_lines.append("### C.2 R3 held-out B identity and fold sizes")
    with open(B_SPLIT_METADATA) as f:
        b_meta = json.load(f)
    meta_folds = {int(f["fold"]): f for f in b_meta["folds"]}

    r3_integrity_rows = []
    for p in sorted(r3_files):
        npz = load_npz(p)
        if npz["split_type"] != "monomer_b_heldout_clustered":
            continue
        fold = npz["fold"]
        meta = meta_folds[fold]

        # sizes
        sizes_ok = (
            npz["n_train"] == meta["n_train"] and
            npz["n_val"] == meta["n_val"] and
            npz["n_test"] == meta["n_test"]
        )
        if not sizes_ok:
            issues.append(
                f"R3 {npz['model']} fold {fold}: sizes mismatch "
                f"(npz {npz['n_train']}/{npz['n_val']}/{npz['n_test']} vs "
                f"meta {meta['n_train']}/{meta['n_val']}/{meta['n_test']})"
            )

        held = set(meta["held_out_monomer_B"])
        val_b = set(meta["validation_monomer_B"])
        train_b = set(meta["train_monomer_B"])
        disjoint = len(held & val_b) == 0 and len(held & train_b) == 0 and len(val_b & train_b) == 0
        if not disjoint:
            issues.append(f"R3 fold {fold}: held/val/train B monomer sets overlap")

        test_indices = npz["test_indices"]
        test_smiles_b = set(df.iloc[test_indices]["smiles_B"])
        test_b_in_held = test_smiles_b.issubset(held)
        if not test_b_in_held:
            issues.append(
                f"R3 {npz['model']} fold {fold}: test smiles_B not all in held_out_monomer_B "
                f"({len(test_smiles_b - held)} stray)"
            )

        r3_integrity_rows.append({
            "model": npz["model"],
            "fold": fold,
            "n_train": npz["n_train"],
            "n_val": npz["n_val"],
            "n_test": npz["n_test"],
            "sizes_match_meta": sizes_ok,
            "sets_disjoint": disjoint,
            "test_b_in_held": test_b_in_held,
        })

    report_lines.append(md_table(r3_integrity_rows))
    report_lines.append("")

    report_lines.append("### C.3 Frozen-split assertion execution")
    report_lines.append(
        "The frozen-split assertion is implemented in `scripts/python/frozen_splits.py`. "
        "Its runtime log line is not present in the downloaded artifacts (task logs were not pulled), "
        "so direct confirmation is **not available here**. Indirect evidence is that the split indices "
        "and held-out B identities reproduce the metadata exactly (see C.2)."
    )
    report_lines.append("")

    # ------------------------------------------------------------------
    # D. EARLY DIAGNOSTIC READS
    # ------------------------------------------------------------------
    report_lines += ["## D. Early diagnostic reads", ""]

    # D.1 final MAE - best MAE
    gap_rows = []
    for p in sorted(r1_files + r3_files):
        npz = load_npz(p)
        best_mae = mean_absolute_error(npz["y_true"], npz["y_pred"])
        final_mae = mean_absolute_error(npz["y_true"], npz["y_pred_final"])
        gap_rows.append({
            "split": npz["split_type"],
            "model": npz["model"],
            "fold": npz["fold"],
            "seed": npz["seed"],
            "best_mae": best_mae,
            "final_mae": final_mae,
            "final_minus_best_mae": final_mae - best_mae,
        })
    gap_df = pd.DataFrame(gap_rows)
    report_lines.append("### D.1 Final vs best checkpoint MAE gap (eV)")
    report_lines.append(md_table(gap_rows))
    report_lines.append("")
    mean_gap = gap_df["final_minus_best_mae"].mean()
    report_lines.append(f"**Mean final - best MAE across {len(gap_df)} pilots: {mean_gap:.6f} eV.**")
    report_lines.append("")
    for split_name, grp in gap_df.groupby("split"):
        report_lines.append(
            f"- {split_name}: mean gap {grp['final_minus_best_mae'].mean():.6f}, max {grp['final_minus_best_mae'].max():.6f}"
        )
    report_lines.append("")
    large_gaps = gap_df[gap_df["final_minus_best_mae"] > 0.05]
    if not large_gaps.empty:
        warnings_list.append(f"{len(large_gaps)} run(s) have final-minus-best MAE gap > 0.05 eV; largest is {large_gaps['final_minus_best_mae'].max():.3f} eV")
    report_lines.append("")

    # D.2 best epochs
    epoch_rows = []
    for p in sorted(r1_files + r3_files):
        npz = load_npz(p)
        sidecar = load_sidecar(p.with_suffix(".config.json"))
        epoch_rows.append({
            "split": npz["split_type"],
            "model": npz["model"],
            "fold": npz["fold"],
            "seed": npz["seed"],
            "best_epoch": sidecar.get("best_epoch"),
            "patience": sidecar.get("resolved_config", {}).get("patience"),
        })
    report_lines.append("### D.2 Best epochs")
    report_lines.append(md_table(epoch_rows))
    if epoch_rows:
        best_epochs = [r["best_epoch"] for r in epoch_rows]
        report_lines.append(
            f"Best epochs range {min(best_epochs)}–{max(best_epochs)}, mean {sum(best_epochs)/len(best_epochs):.1f}. "
            "Patience is 15; clustering around early epochs is noted but does not block the pilot."
        )
        early = [r for r in epoch_rows if r["best_epoch"] is not None and r["best_epoch"] <= 5]
        if early:
            warnings_list.append(f"{len(early)} run(s) reached best epoch ≤ 5 (minimum early-stopping plateau)")
    report_lines.append("")

    # D.3 R1 pilot vs old counterparts: metrics side-by-side
    report_lines.append("### D.3 R1 pilot vs pre-regen counterparts")
    comparison_rows = []
    for p in sorted(r1_files):
        npz_new = load_npz(p)
        old_p = old_r1_counterpart(p)
        if old_p is None:
            continue
        npz_old = load_npz(old_p, require_final=False)
        metrics_new, _ = compute_copolymer_metrics(df, npz_new["y_true"], npz_new["y_pred"], npz_new["test_indices"])
        metrics_old, _ = compute_copolymer_metrics(df, npz_old["y_true"], npz_old["y_pred"], npz_old["test_indices"])
        comparison_rows.append({
            "model": npz_new["model"],
            "seed": npz_new["seed"],
            "old_file": old_p.name,
            "new_group_mean_r2": metrics_new["group_mean_r2"],
            "old_group_mean_r2": metrics_old["group_mean_r2"],
            "delta_group_mean_r2": metrics_new["group_mean_r2"] - metrics_old["group_mean_r2"],
            "new_overall_r2": metrics_new["overall_r2"],
            "old_overall_r2": metrics_old["overall_r2"],
            "new_mae": metrics_new["mae"],
            "old_mae": metrics_old["mae"],
            "new_delta_r2": metrics_new["delta_r2"],
            "old_delta_r2": metrics_old["delta_r2"],
            "new_ordering": metrics_new["ordering"],
            "old_ordering": metrics_old["ordering"],
        })
    report_lines.append(md_table(comparison_rows))
    report_lines.append("")
    report_lines.append(
        "_This is a sanity check on direction and magnitude only; single-seed pilot runs are not interpreted as results._"
    )
    report_lines.append("")
    if comparison_rows:
        comp_df = pd.DataFrame(comparison_rows)
        degraded = comp_df[comp_df["new_group_mean_r2"] < comp_df["old_group_mean_r2"] - 0.05]
        if not degraded.empty:
            warnings_list.append(
                f"{len(degraded)} R1 pilot run(s) have new group-mean R² > 0.05 below the pre-regen counterpart"
            )
    report_lines.append("")

    # D.4 R3 vs per-fold null floor
    report_lines.append("### D.4 R3 group-mean R2 vs per-fold B-blind null floor")
    r3_metric_rows = []
    for p in sorted(r3_files):
        npz = load_npz(p)
        metrics, _ = compute_copolymer_metrics(df, npz["y_true"], npz["y_pred"], npz["test_indices"])
        target = npz["target"]
        # Map filename tokens like "EA_vs_SHE_eV" to the short target names used in the audit table.
        target_key = target.split("_")[0]
        null_floor = NULL_FLOORS.get((target_key, npz["fold"]))
        above_floor = (
            metrics["group_mean_r2"] > null_floor
            if null_floor is not None else None
        )
        r3_metric_rows.append({
            "model": npz["model"],
            "fold": npz["fold"],
            "seed": npz["seed"],
            "group_mean_r2": metrics["group_mean_r2"],
            "overall_r2": metrics["overall_r2"],
            "mae": metrics["mae"],
            "delta_r2": metrics["delta_r2"],
            "null_floor": null_floor,
            "above_floor": above_floor,
        })
    report_lines.append(md_table(r3_metric_rows))
    report_lines.append("")
    below_floor = [r for r in r3_metric_rows if r["above_floor"] is False]
    if below_floor:
        warnings_list.append(
            f"{len(below_floor)} R3 pilot cell(s) fall below the per-fold B-blind null floor"
        )
    else:
        report_lines.append("All R3 pilot group-mean R² values are above their per-fold B-blind null floor.")
    report_lines += [
        "### D.5 PBS task logs",
        "",
        "R3 pilot task logs must be downloaded alongside the NPZs to confirm the frozen-split assertion executed. "
        "Use `scripts/shell/download_regen_v1_artifacts.sh` after the pilot finishes, then grep the logs for markers such as "
        "`Frozen monomer_b_heldout split assertions passed for all folds`, `differs from frozen metadata`, `B-identity leakage`, or `frozen_protocol`.",
        "",
    ]

    # ------------------------------------------------------------------
    # E. BUDGET
    # ------------------------------------------------------------------
    report_lines += ["## E. Budget", ""]
    wall_times = [r["wall_time_s"] for r in training_rows]
    mean_wall = sum(wall_times) / len(wall_times)
    max_wall = max(wall_times)
    report_lines.append(f"- Mean wall time per pilot job: {mean_wall/3600:.2f} h")
    report_lines.append(f"- Max wall time per pilot job: {max_wall/3600:.2f} h")
    report_lines.append(f"- Jobs remaining: R1 = 260, R3 = 212")
    report_lines.append(
        f"- Revised GPU-hour estimate (mean walltime basis): "
        f"R1 {(260 * mean_wall / 3600):.0f} h, R3 {(212 * mean_wall / 3600):.0f} h, "
        f"total {((260 + 212) * mean_wall / 3600):.0f} h"
    )
    report_lines.append(
        f"- Revised GPU-hour estimate (max walltime basis, conservative): "
        f"R1 {(260 * max_wall / 3600):.0f} h, R3 {(212 * max_wall / 3600):.0f} h, "
        f"total {((260 + 212) * max_wall / 3600):.0f} h"
    )
    report_lines.append("")

    # ------------------------------------------------------------------
    # F. VERDICT
    # ------------------------------------------------------------------
    report_lines += ["## F. Verdict", ""]

    r1_go = True
    r3_go = True
    r1_blockers = []
    r3_blockers = []

    # R1 checks
    r1_code = [r for r in code_path_rows if r["split"] == "monomer_heldout"]
    if not r1_code or not all(r["both_present"] and r["hashes_differ"] and not r["ypred_identical"] for r in r1_code):
        r1_go = False
        r1_blockers.append("best/final checkpoint code path not exercised")
    r1_training = [r for r in training_rows if r["split"] == "monomer_heldout"]
    if any(r["epochs_run"] <= 1 or r["wall_time_s"] < 300 for r in r1_training):
        r1_go = False
        r1_blockers.append("at least one R1 pilot run looks untrained")
    r1_old_same = [r for r in r1_old_rows if r.get("identical")]
    if r1_old_same:
        r1_go = False
        r1_blockers.append("regenerated R1 y_pred identical to pre-regen counterpart")

    # R3 checks
    r3_code = [r for r in code_path_rows if r["split"] == "monomer_b_heldout_clustered"]
    if not r3_code or not all(r["both_present"] and r["hashes_differ"] and not r["ypred_identical"] for r in r3_code):
        r3_go = False
        r3_blockers.append("best/final checkpoint code path not exercised")
    r3_training = [r for r in training_rows if r["split"] == "monomer_b_heldout_clustered"]
    if any(r["epochs_run"] <= 1 or r["wall_time_s"] < 300 for r in r3_training):
        r3_go = False
        r3_blockers.append("at least one R3 pilot run looks untrained")
    r3_integrity = [r for r in r3_integrity_rows]
    if not all(r["sizes_match_meta"] and r["sets_disjoint"] and r["test_b_in_held"] for r in r3_integrity):
        r3_go = False
        r3_blockers.append("R3 split integrity check failed")
    if below_floor:
        warnings_list.append("R3 group-mean R2 below per-fold B-blind null floor")

    report_lines.append(f"### R1 remaining 260 jobs: {'**GO**' if r1_go else '**NO-GO**'}")
    if r1_blockers:
        report_lines.append("- Blockers:")
        for b in r1_blockers:
            report_lines.append(f"  - {b}")
    else:
        report_lines.append("- No blocking issues detected in the pilot.")
    report_lines.append("")

    report_lines.append(f"### R3 remaining 212 jobs: {'**GO**' if r3_go else '**NO-GO**'}")
    if r3_blockers:
        report_lines.append("- Blockers:")
        for b in r3_blockers:
            report_lines.append(f"  - {b}")
    else:
        report_lines.append("- No blocking issues detected in the pilot.")
    report_lines.append("")

    if warnings_list:
        report_lines.append("### Warnings")
        for w in warnings_list:
            report_lines.append(f"- {w}")
        report_lines.append("")

    if issues:
        report_lines.append("### Outstanding issues logged during verification")
        for issue in sorted(set(issues)):
            report_lines.append(f"- {issue}")
        report_lines.append("")

    OUTPUT_PATH.write_text("\n".join(report_lines))
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
