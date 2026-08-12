"""Check octamer position-embedding ablation pilot (noposemb) vs baseline.

Frozen protocol (HANDOFF_2026-08-05 §4): every reported cell metric is computed
from the three-seed prediction average (average y_pred across seeds 42/43/44,
then compute the metric once), never from the mean of three per-seed metrics.
Mean-of-metrics and prediction-level-mean metrics are NOT the same number for a
nonlinear metric like R^2, and on this pilot they disagree in sign: mean-of-metrics
across folds 0 and 4 gives delta_r2 diff = +0.0282 (driven almost entirely by one
failed baseline run, fold 4 / seed 43, delta_r2 = -0.2150), while the correct
prediction-level cells give -0.0003 (fold 0) and +0.0150 (fold 4).

Per-seed metrics are still computed and reported, but only in a diagnostic table
labelled as such, alongside the per-cell across-seed SD.
"""
import json, warnings, numpy as np, pandas as pd, sys, importlib.util
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluation.metrics import compute_copolymer_metrics

_spec = importlib.util.spec_from_file_location(
    "figstyle", ROOT / "29-07-2026 supervisor_update" / "figstyle.py")
figstyle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(figstyle)

df = pd.read_csv(ROOT / "data" / "ea_ip.csv")
TMAP = {"EA_vs_SHE_eV": "EA vs SHE (eV)", "IP_vs_SHE_eV": "IP vs SHE (eV)"}

SEEDS = (42, 43, 44)
METRICS = ["group_mean_r2", "delta_r2", "ordering", "overall_r2", "mae", "rmse"]


def metrics_for(y_true, y_pred, idx):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m, _ = compute_copolymer_metrics(df, y_true, y_pred, idx)
    return m


def load_raw(path):
    """Return (y_true, y_pred, test_indices) or None if the file is missing."""
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    yt = d["y_true"].astype(np.float64)
    yp = d["y_pred"].astype(np.float64)
    idx = d["test_indices"].astype(int)
    return yt, yp, idx


def fname(target, model, split, fold, seed, suffix=""):
    return f"ea_ip__{target}__{model}__{split}__fold{fold}__s{seed}{suffix}.npz"


OUT = ROOT / "analysis" / "model_diagnostics"

# ---------------------------------------------------------------------------
# Load per-seed raw predictions for every (target, fold, cond) cell.
# ---------------------------------------------------------------------------
per_seed_rows = []
cell_payloads = {}  # (target, fold, cond) -> list of (seed, y_true, y_pred, idx)

for target in ["EA_vs_SHE_eV"]:
    for fold in [0, 4]:
        for cond, root, suffix in [
            ("baseline", ROOT / "predictions" / "regen_v1" / "ea_ip_lomo", ""),
            ("noposemb", ROOT / "predictions" / "octamer_posemb" / "ea_ip_lomo", "__noposemb"),
        ]:
            loaded = []
            for seed in SEEDS:
                raw = load_raw(root / fname(target, "hpg_hier_octamer", "monomer_heldout", fold, seed, suffix))
                if raw is not None:
                    loaded.append((seed,) + raw)
            cell_payloads[(target, fold, cond)] = loaded
            for seed, yt, yp, idx in loaded:
                m = metrics_for(yt, yp, idx)
                per_seed_rows.append({"target": target, "fold": fold, "seed": seed, "cond": cond, **m})

per_seed = pd.DataFrame(per_seed_rows)
per_seed.to_csv(OUT / "_check_octamer_posemb_pilot_per_seed_diagnostic.csv", index=False)
print("Per-seed metrics (DIAGNOSTIC ONLY — not the reported cell metric):")
print(per_seed[["target", "fold", "seed", "cond"] + METRICS].to_string())

# ---------------------------------------------------------------------------
# Cell metrics: average predictions across seeds first, then compute the
# metric once. This is the only number that should be reported as "the"
# metric for a cell.
# ---------------------------------------------------------------------------
cell_rows = []
for (target, fold, cond), loaded in cell_payloads.items():
    n_seeds = len(loaded)
    assert n_seeds == 3, (
        f"Expected exactly 3 seeds contributing to cell "
        f"(target={target}, fold={fold}, cond={cond}), got {n_seeds}. "
        "Refusing to report a cell metric built from an incomplete seed set."
    )
    seeds, y_trues, y_preds, idxs = zip(*loaded)
    first_yt, first_idx = y_trues[0], idxs[0]
    for yt, idx in zip(y_trues[1:], idxs[1:]):
        assert np.array_equal(yt, first_yt) and np.array_equal(idx, first_idx), (
            f"Rows differ across seeds for (target={target}, fold={fold}, cond={cond})"
        )
    y_pred_avg = np.mean(np.stack(y_preds), axis=0)
    m = metrics_for(first_yt, y_pred_avg, first_idx)

    seed_metrics = per_seed[
        (per_seed.target == target) & (per_seed.fold == fold) & (per_seed.cond == cond)
    ]
    row = {"target": target, "fold": fold, "cond": cond, "n_seeds": n_seeds, **m}
    for metric in METRICS:
        row[f"{metric}_seed_sd"] = float(seed_metrics[metric].std(ddof=1))
    cell_rows.append(row)

cells = pd.DataFrame(cell_rows)
cells.to_csv(OUT / "_check_octamer_posemb_pilot_metrics.csv", index=False)
print("\nCell metrics (three-seed prediction average, computed once — this is the reported number):")
print(cells[["target", "fold", "cond", "n_seeds"] + METRICS].to_string())

# ---------------------------------------------------------------------------
# Summary: paired cell-level differences (noposemb minus baseline), per fold.
# This replaces the old mean-of-per-seed-metrics summary, which silently
# averaged over per-seed noise (including one failed run) rather than
# reflecting the frozen three-seed-prediction-average protocol.
# ---------------------------------------------------------------------------
summary = []
if len(cells):
    for target, g in cells.groupby("target"):
        for fold, gg in g.groupby("fold"):
            baseline = gg[gg.cond == "baseline"]
            noposemb = gg[gg.cond == "noposemb"]
            if baseline.empty or noposemb.empty:
                continue
            for m in METRICS:
                baseline_val = float(baseline[m].iloc[0])
                noposemb_val = float(noposemb[m].iloc[0])
                summary.append({
                    "target": target,
                    "fold": fold,
                    "metric": m,
                    "baseline_cell": baseline_val,
                    "noposemb_cell": noposemb_val,
                    "diff": noposemb_val - baseline_val,
                })
    sm = pd.DataFrame(summary)
    sm.to_csv(OUT / "_check_octamer_posemb_pilot_summary.csv", index=False)
    print("\nSummary (per-fold, three-seed-prediction-average cells; noposemb minus baseline):")
    print(sm.to_string())

    # -----------------------------------------------------------------
    # Metric-direction finding: overall_r2 and delta_r2 moved in opposite
    # directions on both pilot folds. This is a finding about the metrics
    # themselves (overall_r2 rewards well-calibrated bulk predictions;
    # delta_r2 rewards recovering architecture-driven within-group
    # structure, and the two need not move together), not a finding about
    # the ablation.
    # -----------------------------------------------------------------
    print("\nMetric-direction check (overall_r2 vs delta_r2, noposemb minus baseline, per fold):")
    for fold in sorted(cells.fold.unique()):
        row = sm[(sm.fold == fold) & (sm.metric.isin(["overall_r2", "delta_r2"]))]
        if len(row) == 2:
            o = float(row[row.metric == "overall_r2"]["diff"].iloc[0])
            d = float(row[row.metric == "delta_r2"]["diff"].iloc[0])
            opposite = (o > 0) != (d > 0)
            print(f"  fold {fold}: overall_r2 diff={o:+.4f}, delta_r2 diff={d:+.4f}, "
                  f"opposite sign={opposite}")

    # Quick plot — three-seed-averaged cell values, not per-seed metrics.
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.5))
    plot_metrics = ["group_mean_r2", "delta_r2", "overall_r2", "mae", "rmse", "ordering"]
    for ax, m in zip(axes.flat, plot_metrics):
        sub = cells[cells.target == "EA_vs_SHE_eV"].pivot_table(index="fold", columns="cond", values=m, aggfunc="first")
        for fold, row in sub.iterrows():
            ax.plot([0, 1], [row["baseline"], row["noposemb"]], marker="o", color="#888", lw=0.8, ms=5)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["baseline", "noposemb"])
        ax.set_ylabel(m); ax.set_title(f"{m} — EA")
    fig.suptitle("octamer position embeddings off pilot (folds 0,4) — three-seed prediction average", fontsize=10)
    plt.tight_layout()
    fig.savefig(OUT / "_check_octamer_posemb_pilot_plot.png", dpi=200)
    print("\nWrote", OUT / "_check_octamer_posemb_pilot_plot.png")
