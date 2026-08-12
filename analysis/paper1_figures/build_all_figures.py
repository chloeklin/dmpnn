"""
build_all_figures.py — Six paper figures for polymer evaluation methodology.

Usage (from project root):
    .venv/bin/python analysis/paper1_figures/build_all_figures.py
"""
from __future__ import annotations
import importlib.util, json, sys, warnings
from pathlib import Path
from textwrap import dedent
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluation.metrics import compute_copolymer_metrics

# Use the same A-blind null implementation and LOMAO split generator that produced
# analysis/model_diagnostics/_groupmean_metric_floor.md.
sys.path.insert(0, str(ROOT / "scripts" / "python"))
from utils import generate_a_held_out_splits
from aggregate_lomo_seeds import null_floor as _lomo_null_floor

_spec = importlib.util.spec_from_file_location(
    "figstyle", ROOT / "29-07-2026 supervisor_update" / "figstyle.py")
figstyle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(figstyle)

OUTDIR   = ROOT / "analysis" / "paper1_figures"
OUTDIR.mkdir(parents=True, exist_ok=True)
SEEDS    = [42, 43, 44]
TMAP     = {"EA": "EA vs SHE (eV)", "IP": "IP vs SHE (eV)"}
TSAFE    = {"EA": "EA_vs_SHE_eV",   "IP": "IP_vs_SHE_eV"}
PRED_R1  = ROOT / "predictions" / "regen_v1" / "ea_ip_lomo"
PRED_R3  = ROOT / "predictions" / "regen_v1" / "ea_ip_lomo_b_clustered"
SPLIT_R1 = "monomer_heldout"
SPLIT_R3 = "monomer_b_heldout_clustered"
MNAMES   = {"octamer": "hpg_hier_octamer", "wdmpnn": "wdmpnn"}

# ── Utilities ─────────────────────────────────────────────────────────────────

def load_npz(pred_dir, model, tsafe, split, fold, seed):
    p = pred_dir / f"ea_ip__{tsafe}__{model}__{split}__fold{fold}__s{seed}.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    return {"y_true": d["y_true"].astype(np.float64),
            "y_pred": d["y_pred"].astype(np.float64),
            "idx":    d["test_indices"].astype(int)}

def avg_seeds(pred_dir, model, tsafe, split, fold):
    parts = [load_npz(pred_dir, model, tsafe, split, fold, s) for s in SEEDS]
    if any(p is None for p in parts):
        return None
    y_pred = np.mean([p["y_pred"] for p in parts], axis=0)
    return parts[0]["y_true"], y_pred, parts[0]["idx"]

def metrics(df, yt, yp, idx):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m, _ = compute_copolymer_metrics(df, yt, yp, idx)
    return m

def apply_style():
    figstyle.apply_style("paper")
    plt.rcParams["savefig.dpi"] = 200

def save(fig, name):
    return figstyle.save(fig, OUTDIR, name, formats=("png", "pdf"))

def write_manifest(name, txt):
    p = OUTDIR / f"{name}_manifest.md"
    p.write_text(txt.strip())
    print(f"  manifest → {p.name}")

# ── F1 — Variance decomposition ───────────────────────────────────────────────

def build_f1(df):
    print("F1 — variance decomposition")

    def fr2(tcol, keys):
        y = df[tcol].values
        p = df.groupby(keys)[tcol].transform("mean").values
        return float(r2_score(y, p))

    rows = []
    for tn, tc in TMAP.items():
        rA   = fr2(tc, ["smiles_A"])
        rAB  = fr2(tc, ["smiles_A","smiles_B"])
        rABf = fr2(tc, ["smiles_A","smiles_B","fracA"])
        rFUL = fr2(tc, ["smiles_A","smiles_B","fracA","poly_type"])
        rows.append({"target":tn, "r2_A":rA, "r2_AB":rAB, "r2_ABf":rABf, "r2_full":rFUL,
                     "A":rA, "B_given_A":rAB-rA, "comp_given_AB":rABf-rAB,
                     "arch":rFUL-rABf, "residual":1.0-rFUL,
                     "arch_pct_total":(rFUL-rABf)*100,
                     "arch_pct_post_AB":((rFUL-rABf)/(1-rAB)*100) if (1-rAB)>1e-10 else np.nan,
                     "n":len(df)})
    csv = pd.DataFrame(rows)
    csv.to_csv(OUTDIR/"f1_variance_decomposition.csv", index=False)

    apply_style()
    C = figstyle.COLORS
    comp_cols  = ["A","B_given_A","comp_given_AB","arch","residual"]
    comp_lbls  = ["Monomer A","Monomer B | A","Composition | A,B","Architecture | A,B,comp","Residual"]
    comp_clrs  = [C[0], C[2], C[3], C[1], "#888888"]

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    for ax, tn in zip(axes, ["EA","IP"]):
        row  = csv[csv.target==tn].iloc[0]
        vals = [float(row[c]) for c in comp_cols]
        bot  = 0.0
        for col, val, lbl in zip(comp_clrs, vals, comp_lbls):
            ax.bar(0, val, bottom=bot, color=col, label=lbl, width=0.5, linewidth=0)
            bot += val
        ai   = comp_cols.index("arch")
        abot = sum(vals[:ai])
        ax.annotate(
            f"arch: {row.arch_pct_total:.1f}% of total\n({row.arch_pct_post_AB:.1f}% of post-AB residual)",
            xy=(0.28, abot + vals[ai]/2),
            # xytext was at y = abot + 4*arch ~= 1.03 against a ylim of 1.05, so the
            # two-line label overflowed the axes and landed on the panel title. The
            # bar occupies x in [-0.25, 0.25]; the right half of the panel is empty.
            xytext=(0.62, 0.58),
            fontsize=7.5, arrowprops=dict(arrowstyle="-", color="#666", lw=0.7),
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'), zorder=20)
        ax.set_xlim(-0.5, 1.5); ax.set_ylim(0, 1.05); ax.set_xticks([])
        ax.set_ylabel("Fraction of total variance")
        ax.set_title(f"{tn}: {TMAP[tn]}")
    # "lower left" sat on top of the bar; the right half of the panel is empty.
    leg1 = axes[0].legend(loc="center right", fontsize=7, frameon=True)
    leg1.get_frame().set_facecolor('white'); leg1.get_frame().set_edgecolor('none'); leg1.set_zorder(20)
    fig.suptitle("Variance decomposition — EA and IP copolymer targets", fontsize=9, y=1.01)
    plt.tight_layout()
    save(fig, "f1_variance_decomposition")

    ea, ip = csv[csv.target=="EA"].iloc[0], csv[csv.target=="IP"].iloc[0]
    write_manifest("f1", dedent(f"""
    # F1 — Variance decomposition

    ## Data source
    `data/ea_ip.csv` ({int(ea.n)} rows). No model predictions.

    ## Metric
    Sequential R² via `sklearn.metrics.r2_score` on group-transform means
    (same as `audit_b_heldout_design.factor_variance`).

    ## Cumulative R² values
    | Keys | EA | IP |
    |---|---|---|
    | smiles_A | {ea.r2_A:.6f} | {ip.r2_A:.6f} |
    | +smiles_B | {ea.r2_AB:.6f} | {ip.r2_AB:.6f} |
    | +fracA | {ea.r2_ABf:.6f} | {ip.r2_ABf:.6f} |
    | +poly_type | {ea.r2_full:.6f} | {ip.r2_full:.6f} |

    ## Component shares
    | Component | EA | IP |
    |---|---|---|
    | A | {ea.A:.6f} | {ip.A:.6f} |
    | B\\|A | {ea.B_given_A:.6f} | {ip.B_given_A:.6f} |
    | comp\\|A,B | {ea.comp_given_AB:.6f} | {ip.comp_given_AB:.6f} |
    | arch\\|A,B,comp | {ea.arch:.6f} | {ip.arch:.6f} |
    | residual | {ea.residual:.6f} | {ip.residual:.6f} |

    ## Architecture annotations
    - EA: {ea.arch_pct_total:.2f}% of total; {ea.arch_pct_post_AB:.1f}% of post-AB residual
    - IP: {ip.arch_pct_total:.2f}% of total; {ip.arch_pct_post_AB:.1f}% of post-AB residual

    ## Cells: 1 whole-dataset computation per target (2 targets).
    """))

# ── F2 — Worked example ───────────────────────────────────────────────────────

def build_f2(df):
    """F2: paired two-model worked example (selected example).

    Selects one group where octamer and wDMPNN tie on chemistry placement but
    disagree completely on architecture ordering (octamer=1.0, wDMPNN=0.0).
    Pool: all 9 folds × both targets × both models. Figure rendered for EA.

    Marginals note (§3 — 10 August): the conjunction of ordering==0.0 and
    gm_err in the best decile is rare partly by construction (the decile
    condition selects 10% of groups by definition); the ordering-failure rate
    alone describes how often the model fails to rank architecture. The
    expected_if_indep value is preserved for each model × target × fold-0 cell.
    """
    import matplotlib.lines as mlines
    print("F2 — worked example (paired two-model, selected example)")
    grpk = ["smiles_A", "smiles_B", "fracA"]
    FOLDS = list(range(9))
    TARGETS_F2 = [("EA", "EA_vs_SHE_eV"), ("IP", "IP_vs_SHE_eV")]
    MODELS_F2  = [("octamer", "hpg_hier_octamer"), ("wdmpnn", "wdmpnn")]

    # ── Load all predictions ─────────────────────────────────────────────────
    missing = []
    raw = {}
    for tshort, tsafe in TARGETS_F2:
        for fold in FOLDS:
            for mkey, mname in MODELS_F2:
                r = avg_seeds(PRED_R1, mname, tsafe, SPLIT_R1, fold)
                if r is None:
                    for seed in SEEDS:
                        p = PRED_R1 / f"ea_ip__{tsafe}__{mname}__{SPLIT_R1}__fold{fold}__s{seed}.npz"
                        if not p.exists():
                            missing.append(str(p.relative_to(ROOT)))
                else:
                    raw[(tshort, fold, mkey)] = r

    if missing:
        miss_txt = "\n".join(f"- {f}" for f in sorted(set(missing)))
        write_manifest("f2", f"# F2 — Worked example\n\n## Missing files\n{miss_txt}")
        return

    # ── Per-group stats helper ───────────────────────────────────────────────
    def grp_stats(yt_arr, yp_arr, idx_arr):
        frame = df.iloc[idx_arr].reset_index(drop=True)[grpk + ["poly_type"]].copy()
        frame["y_true"] = yt_arr
        frame["y_pred"] = yp_arr
        sub05 = frame[frame.fracA == 0.5].copy()
        cnt = sub05.groupby(grpk).poly_type.nunique()
        three_ids = cnt[cnt == 3].index
        sub = sub05[sub05.set_index(grpk).index.isin(three_ids)].copy()
        out = {}
        for gid, gdf in sub.groupby(grpk):
            gdf = gdf.sort_values("y_true").reset_index(drop=True)
            gmt_g = float(gdf.y_true.mean())
            gmp_g = float(gdf.y_pred.mean())
            yt_g, yp_g = gdf.y_true.values, gdf.y_pred.values
            pairs = [(i, j) for i in range(3) for j in range(i + 1, 3) if yt_g[i] != yt_g[j]]
            oa = float(np.mean([
                0.5 if yp_g[i] == yp_g[j]
                else float((yt_g[i] - yt_g[j]) * (yp_g[i] - yp_g[j]) > 0)
                for i, j in pairs
            ])) if pairs else np.nan
            _ts = float(yt_g.max() - yt_g.min())
            _ps = float(yp_g.max() - yp_g.min())
            out[gid] = {
                "gm_err": abs(gmt_g - gmp_g), "ordering": oa,
                "spread": _ts, "spread_pred": _ps,
                "ratio": _ps / _ts if _ts > 0.0 else np.nan,
                "gdf": gdf,
            }
        return out, len(three_ids)

    # ── Build eligible pool; assert §4 counts; collect marginals ─────────────
    EXPECTED_POOL = {
        "EA": [0, 10, 12, 2, 6, 12, 14, 1, 4],
        "IP": [0, 19,  5, 1, 1,  0, 25, 2, 1],
    }
    eligible = {"EA": [], "IP": []}
    all_stats = {}
    fold_ord_rates      = {t: {m: [] for m, _ in MODELS_F2} for t, _ in TARGETS_F2}
    fold_mean_ord       = {t: {m: [] for m, _ in MODELS_F2} for t, _ in TARGETS_F2}
    fold_spread_ratios  = {t: {m: [] for m, _ in MODELS_F2} for t, _ in TARGETS_F2}
    fold_collapse_rates = {t: {m: [] for m, _ in MODELS_F2} for t, _ in TARGETS_F2}

    for fold in FOLDS:
        for tshort, _ in TARGETS_F2:
            oct_s, n_oct = grp_stats(*raw[(tshort, fold, "octamer")])
            wdm_s, n_wdm = grp_stats(*raw[(tshort, fold, "wdmpnn")])
            if n_oct != 682:
                raise AssertionError(
                    f"682-group assertion failed: {tshort} fold {fold} octamer got {n_oct}")
            if n_wdm != 682:
                raise AssertionError(
                    f"682-group assertion failed: {tshort} fold {fold} wdmpnn got {n_wdm}")
            all_stats[(tshort, fold, "octamer")] = oct_s
            all_stats[(tshort, fold, "wdmpnn")]  = wdm_s

            for mkey, mstats in [("octamer", oct_s), ("wdmpnn", wdm_s)]:
                ords = [s["ordering"] for s in mstats.values() if not np.isnan(s["ordering"])]
                fold_ord_rates[tshort][mkey].append(
                    100.0 * sum(v == 0.0 for v in ords) / len(ords) if ords else np.nan)
                fold_mean_ord[tshort][mkey].append(float(np.mean(ords)) if ords else np.nan)
                ratios_f = [s["ratio"] for s in mstats.values()
                            if not np.isnan(s.get("ratio", np.nan))]
                fold_spread_ratios[tshort][mkey].append(
                    float(np.median(ratios_f)) if ratios_f else np.nan)
                fold_collapse_rates[tshort][mkey].append(
                    100.0 * sum(r < 0.25 for r in ratios_f) / len(ratios_f) if ratios_f else np.nan)

            n_elig = 0
            for gid in set(oct_s.keys()) & set(wdm_s.keys()):
                o, w = oct_s[gid], wdm_s[gid]
                if (not np.isnan(o["ordering"]) and not np.isnan(w["ordering"])
                        and abs(o["gm_err"] - w["gm_err"]) <= 0.01
                        and o["ordering"] == 1.0 and w["ordering"] == 0.0):
                    n_elig += 1
                    eligible[tshort].append({
                        "fold": fold, "gid": gid,
                        "gm_err_octamer": o["gm_err"], "gm_err_wdmpnn": w["gm_err"],
                        "spread": o["spread"],
                        "gdf_octamer": o["gdf"], "gdf_wdmpnn": w["gdf"],
                    })
            exp = EXPECTED_POOL[tshort][fold]
            if n_elig != exp:
                raise AssertionError(
                    f"F2 §4 pool assertion failed: {tshort} fold {fold} = {n_elig} eligible, "
                    f"expected {exp}. Do not adjust criterion — stop and reconcile.")

    ea_counts = [sum(1 for e in eligible["EA"] if e["fold"] == f) for f in FOLDS]
    ip_counts = [sum(1 for e in eligible["IP"] if e["fold"] == f) for f in FOLDS]
    print(f"  EA eligible pool: {ea_counts}  total={len(eligible['EA'])}")
    print(f"  IP eligible pool: {ip_counts}  total={len(eligible['IP'])}")

    # ── Fold-0 marginals assertion (§4 table) ─────────────────────────────────
    FOLD0_SPEC = {
        ("EA", "octamer"): {"n_ord0": 69, "pct_ord0": 10.1, "mean_ordering": 0.814, "decile_thr": 0.01806, "joint": 1},
        ("EA", "wdmpnn"):  {"n_ord0": 43, "pct_ord0":  6.3, "mean_ordering": 0.858, "decile_thr": 0.09580, "joint": 7},
        ("IP", "octamer"): {"n_ord0": 27, "pct_ord0":  4.0, "mean_ordering": 0.886, "decile_thr": 0.04317, "joint": 6},
        ("IP", "wdmpnn"):  {"n_ord0": 42, "pct_ord0":  6.2, "mean_ordering": 0.822, "decile_thr": 0.10909, "joint": 10},
    }
    fold0_computed = {}
    disc_f0 = []
    for tshort in ["EA", "IP"]:
        for mkey in ["octamer", "wdmpnn"]:
            mstats = all_stats[(tshort, 0, mkey)]
            ords    = [s["ordering"] for s in mstats.values() if not np.isnan(s["ordering"])]
            gm_errs = [s["gm_err"]   for s in mstats.values()]
            n_ord0   = sum(v == 0.0 for v in ords)
            pct_ord0 = 100.0 * n_ord0 / len(ords)
            mean_ord = float(np.mean(ords))
            n_decile = len(ords)   # will be overwritten
            dthr     = float(np.quantile(gm_errs, 0.10))
            n_decile = sum(g <= dthr for g in gm_errs)
            joint    = sum(
                s["ordering"] == 0.0 and s["gm_err"] <= dthr
                for s in mstats.values() if not np.isnan(s["ordering"]))
            # §3 marginals: expected_if_indep preserved
            expected_if_indep = (n_ord0 / len(ords)) * 0.10 * len(gm_errs)
            fold0_computed[(tshort, mkey)] = {
                "n_ord0": n_ord0, "pct_ord0": pct_ord0, "mean_ordering": mean_ord,
                "decile_thr": dthr, "n_decile": n_decile, "joint": joint,
                "expected_if_indep": expected_if_indep,
            }
            sp = FOLD0_SPEC[(tshort, mkey)]
            if n_ord0 != sp["n_ord0"]:
                disc_f0.append(f"{tshort}·{mkey} n_ord0: got {n_ord0}, expected {sp['n_ord0']}")
            if round(pct_ord0, 1) != sp["pct_ord0"]:
                disc_f0.append(f"{tshort}·{mkey} pct_ord0: got {pct_ord0:.1f}, expected {sp['pct_ord0']}")
            if abs(mean_ord - sp["mean_ordering"]) > 5e-4:
                disc_f0.append(f"{tshort}·{mkey} mean_ordering: got {mean_ord:.3f}, expected {sp['mean_ordering']}")
            if abs(dthr - sp["decile_thr"]) > 5e-4:
                disc_f0.append(f"{tshort}·{mkey} decile_thr: got {dthr:.5f}, expected {sp['decile_thr']:.5f}")
            if joint != sp["joint"]:
                disc_f0.append(f"{tshort}·{mkey} joint: got {joint}, expected {sp['joint']}")
    if disc_f0:
        raise AssertionError("F2 fold-0 marginals (§4) disagree:\n" + "\n".join(disc_f0))
    print("  Fold-0 marginals OK")

    # ── Median ordering-failure rate assertion (§4) ──────────────────────────
    MEDIAN_FAIL_SPEC = {
        ("EA", "octamer"): 6.5, ("IP", "octamer"): 6.6,
        ("EA", "wdmpnn"):  8.8, ("IP", "wdmpnn"):  8.1,
    }
    median_fail = {}
    disc_med = []
    for tshort in ["EA", "IP"]:
        for mkey in ["octamer", "wdmpnn"]:
            rates = [r for r in fold_ord_rates[tshort][mkey] if not np.isnan(r)]
            med = float(np.median(rates))
            median_fail[(tshort, mkey)] = med
            exp = MEDIAN_FAIL_SPEC[(tshort, mkey)]
            if abs(med - exp) > 0.2:
                disc_med.append(
                    f"Median fail rate {tshort}·{mkey}: got {med:.1f}%, expected {exp}%")
    if disc_med:
        raise AssertionError("F2 median ordering-failure rates (§4) disagree:\n" + "\n".join(disc_med))
    print("  Median ordering-failure rates OK")

    # ── §2 per-fold arch_spread_ratio median assertions ────────────────────────────
    SPREAD_RATIO_SPEC = {
        ("EA", "octamer"): [0.649, 0.747, 0.929, 0.689, 0.811, 0.833, 1.559, 0.713, 0.768],
        ("EA", "wdmpnn"):  [0.775, 0.247, 0.378, 0.526, 0.909, 0.582, 0.610, 1.390, 0.616],
        ("IP", "octamer"): [0.570, 0.947, 0.927, 0.904, 1.162, 0.738, 0.967, 1.118, 1.034],
        ("IP", "wdmpnn"):  [0.467, 1.484, 1.074, 1.219, 0.968, 0.265, 0.350, 0.229, 0.551],
    }
    disc_sr = []
    for (tshort_k, mkey_k), spec_vals in SPREAD_RATIO_SPEC.items():
        computed = fold_spread_ratios[tshort_k][mkey_k]
        for fold_i, (c, s) in enumerate(zip(computed, spec_vals)):
            if abs(c - s) > 0.001:
                disc_sr.append(f"{tshort_k}·{mkey_k} fold {fold_i}: got {c:.3f}, expected {s:.3f}")
    if disc_sr:
        raise AssertionError("F2 §2 arch_spread_ratio medians disagree:\n" + "\n".join(disc_sr))
    print("  §2 per-fold spread ratio medians OK")

    # ── §2 collapse-rate spot checks ─────────────────────────────────────────────
    COLLAPSE_SPEC = [
        ("EA", "wdmpnn", 1, 50.0),
        ("IP", "wdmpnn", 5, 44.3),
        ("IP", "wdmpnn", 7, 52.3),
    ]
    disc_cr = []
    for tshort_k, mkey_k, fold_i, spec_pct in COLLAPSE_SPEC:
        computed = fold_collapse_rates[tshort_k][mkey_k][fold_i]
        if abs(computed - spec_pct) > 0.2:
            disc_cr.append(f"{tshort_k}·{mkey_k} fold {fold_i}: got {computed:.1f}%, expected {spec_pct:.1f}%")
    if disc_cr:
        raise AssertionError("F2 §2 collapse-rate spot checks disagree:\n" + "\n".join(disc_cr))
    print("  §2 collapse-rate spot checks OK")

    # ── §2 octamer collapse-rate bounds (6.3–18.5% EA, 6.6–16.0% IP) ────────────
    disc_oc = []
    for tshort_k, lo, hi in [("EA", 6.3, 18.5), ("IP", 6.6, 16.0)]:
        for fold_i, rate in enumerate(fold_collapse_rates[tshort_k]["octamer"]):
            rate_r = round(rate, 1)
            if not (lo <= rate_r <= hi):
                disc_oc.append(f"octamer {tshort_k} fold {fold_i}: {rate_r:.1f}% outside [{lo},{hi}]%")
    if disc_oc:
        raise AssertionError("F2 §2 octamer collapse-rate bounds failed:\n" + "\n".join(disc_oc))
    print("  §2 octamer collapse-rate bounds OK")

    # ── Select group: largest true architecture spread in EA pool ────────────────
    ea_pool = eligible["EA"]
    if not ea_pool:
        write_manifest("f2", "# F2 — Worked example\n\nNo eligible groups found in EA pool.")
        return
    best_e = max(ea_pool, key=lambda e: e["spread"])
    sorted_spreads = sorted((e["spread"] for e in ea_pool), reverse=True)
    spread_rank = sorted_spreads.index(best_e["spread"]) + 1
    best_fold = best_e["fold"]
    best_gid  = best_e["gid"]
    gdf_oct   = best_e["gdf_octamer"]
    gdf_wdm   = best_e["gdf_wdmpnn"]
    gmt       = float(gdf_oct.y_true.mean())
    gmp_oct   = float(gdf_oct.y_pred.mean())
    gmp_wdm   = float(gdf_wdm.y_pred.mean())
    print(f"  Selected: fold {best_fold}, gid={best_gid}, spread={best_e['spread']:.4f} "
          f"(rank {spread_rank}/{len(ea_pool)})")

    # ── §2 selected-group spread assertions ────────────────────────────────────
    true_spread = float(gdf_oct.y_true.max() - gdf_oct.y_true.min())
    spread_oct  = float(gdf_oct.y_pred.max() - gdf_oct.y_pred.min())
    spread_wdm  = float(gdf_wdm.y_pred.max() - gdf_wdm.y_pred.min())
    ratio_oct   = spread_oct / true_spread if true_spread > 0.0 else np.nan
    ratio_wdm   = spread_wdm / true_spread if true_spread > 0.0 else np.nan
    disc_f2s = []
    if abs(true_spread - 0.33250) > 1e-4:
        disc_f2s.append(f"true spread: got {true_spread:.5f}, expected 0.33250")
    if abs(spread_oct - 0.32904) > 1e-4:
        disc_f2s.append(f"octamer pred spread: got {spread_oct:.5f}, expected 0.32904")
    if abs(spread_wdm - 0.02400) > 1e-4:
        disc_f2s.append(f"wDMPNN pred spread: got {spread_wdm:.5f}, expected 0.02400")
    if disc_f2s:
        raise AssertionError("§2 F2 selected-group spread:\n" + "\n".join(disc_f2s))
    print(f"  §2 selected-group spread OK: oct={ratio_oct:.4f}, wDMPNN={ratio_wdm:.4f}")

    # ── CSV — model column, no NaN ────────────────────────────────────────────────────
    csv_rows = []
    for _, r in gdf_oct.iterrows():
        wrow = gdf_wdm[gdf_wdm.poly_type == r.poly_type].iloc[0]
        csv_rows.append({
            "smiles_A": best_gid[0], "smiles_B": best_gid[1], "fracA": best_gid[2],
            "fold": best_fold, "poly_type": r.poly_type,
            "y_true": r.y_true, "delta_true": r.y_true - gmt,
            "model": "hpg_hier_octamer", "y_pred": r.y_pred,
            "delta_pred": r.y_pred - gmp_oct,
            "arch_spread_ratio_predavg": ratio_oct,
        })
        csv_rows.append({
            "smiles_A": best_gid[0], "smiles_B": best_gid[1], "fracA": best_gid[2],
            "fold": best_fold, "poly_type": r.poly_type,
            "y_true": r.y_true, "delta_true": r.y_true - gmt,
            "model": "wdmpnn", "y_pred": float(wrow.y_pred),
            "delta_pred": float(wrow.y_pred) - gmp_wdm,
            "arch_spread_ratio_predavg": ratio_wdm,
        })
    pd.DataFrame(csv_rows).to_csv(OUTDIR / "f2_worked_example.csv", index=False)

    # ── Plot ──────────────────────────────────────────────────────────────────
    apply_style()
    C = figstyle.COLORS
    arch_c = {"block": C[5], "random": C[1], "alternating": C[2]}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.8))
    for _, r in gdf_oct.iterrows():
        wrow = gdf_wdm[gdf_wdm.poly_type == r.poly_type].iloc[0]
        yp_w = float(wrow.y_pred)
        c = arch_c.get(r.poly_type, "#555555")
        ax1.scatter(0, r.y_true,  color=c, marker="o", s=60, zorder=5)
        ax1.scatter(1, r.y_pred,  color=c, marker="^", s=60, zorder=5)
        ax1.scatter(2, yp_w,      color=c, marker="s", s=55, zorder=5)
        ax1.plot([0, 1], [r.y_true, r.y_pred], color=c, lw=0.9, alpha=0.65, ls="-")
        ax1.plot([0, 2], [r.y_true, yp_w],     color=c, lw=0.9, alpha=0.65, ls=":")
        ax2.scatter(0, r.y_true - gmt,      color=c, marker="o", s=60, zorder=5)
        ax2.scatter(1, r.y_pred - gmp_oct,  color=c, marker="^", s=60, zorder=5)
        ax2.scatter(2, yp_w      - gmp_wdm, color=c, marker="s", s=55, zorder=5)
        ax2.plot([0, 1], [r.y_true - gmt, r.y_pred - gmp_oct], color=c, lw=0.9, alpha=0.65, ls="-")
        ax2.plot([0, 2], [r.y_true - gmt, yp_w      - gmp_wdm], color=c, lw=0.9, alpha=0.65, ls=":")

    ax1.axhline(gmt,     color="#444", lw=1.0, ls="--")
    ax1.axhline(gmp_oct, color="#777", lw=0.8, ls=":")
    ax1.axhline(gmp_wdm, color="#aaa", lw=0.8, ls="-.")
    ax2.axhline(0.0, color="#444", lw=1.0, ls="--")

    for ax, ttl in zip([ax1, ax2], ["Chemistry placement", "Architecture recovery"]):
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["True", "Octamer", "wDMPNN"])
        ax.set_ylabel("EA (eV)")
        ax.set_title(ttl)

    # zorder must exceed the scatter's zorder=5, otherwise markers draw over the
    # text and the white bbox cannot rescue it (this was the actual collision --
    # repositioning alone did not fix it).
    _ann_bbox = dict(facecolor="white", alpha=0.85, edgecolor="none")
    ax1.text(0.03, 0.03,
             f"gm err: oct={best_e['gm_err_octamer']:.4f} / wDMPNN={best_e['gm_err_wdmpnn']:.4f} eV",
             transform=ax1.transAxes, fontsize=7.0, va="bottom", ha="left",
             bbox=_ann_bbox, zorder=20)
    ax2.text(0.97, 0.97,
             (f"ordering: oct=1.0  wDMPNN=0.0\n"
              f"spread ratio (predavg): oct={ratio_oct:.2f}  wDMPNN={ratio_wdm:.2f}"),
             transform=ax2.transAxes, fontsize=7.0, va="top", ha="right",
             bbox=_ann_bbox, zorder=20)

    arch_handles = [mpatches.Patch(color=arch_c[a], label=a) for a in ["block", "random", "alternating"]]
    model_handles = [
        mlines.Line2D([0], [0], marker="o", color="#555", ms=6, ls="none", label="true"),
        mlines.Line2D([0], [0], marker="^", color="#555", ms=6, ls="none", label="octamer (ordering=1.0)"),
        mlines.Line2D([0], [0], marker="s", color="#555", ms=6, ls="none", label="wDMPNN (ordering=0.0)"),
    ]
    fig.legend(handles=arch_handles + model_handles, loc="upper center", ncol=3,
               fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle(
        f"selected example — octamer vs wDMPNN · R1 fold {best_fold} · EA · "
        f"spread rank {spread_rank} of {len(ea_pool)} eligible",
        fontsize=8, y=1.11)
    plt.tight_layout()
    save(fig, "f2_worked_example")

    # ── Manifest ──────────────────────────────────────────────────────────────
    def mrow_f2(tshort, mkey):
        c = fold0_computed[(tshort, mkey)]
        return (f"| {tshort} · {mkey} | {c['n_ord0']} ({c['pct_ord0']:.1f}%) | "
                f"{c['mean_ordering']:.3f} | {c['decile_thr']:.5f} | {c['joint']} | "
                f"{c['expected_if_indep']:.2f} |")

    f4_ea = {m: {"fail": fold_ord_rates["EA"][m][4], "mean": fold_mean_ord["EA"][m][4]}
             for m in ["octamer", "wdmpnn"]}

    def sr_row(tshort, mkey):
        vals = fold_spread_ratios[tshort][mkey]
        return f"| {tshort} \u00b7 {mkey} | " + " | ".join(f"{v:.3f}" for v in vals) + " |"

    def cr_row(tshort, mkey):
        vals = fold_collapse_rates[tshort][mkey]
        return f"| {tshort} \u00b7 {mkey} | " + " | ".join(f"{v:.1f}%" for v in vals) + " |"

    write_manifest("f2", dedent(f"""
    # F2 — Worked example (selected example)

    ## Prediction files loaded
    `{PRED_R1.relative_to(ROOT)}/ea_ip__[EA|IP]_vs_SHE_eV__[hpg_hier_octamer|wdmpnn]__{SPLIT_R1}__fold[0-8]__s[42/43/44].npz`
    Seeds 42,43,44 averaged at the prediction level; metric computed once on averaged predictions.
    Both models, both targets (EA, IP), all 9 folds used for pool construction. Figure rendered for EA.

    ## Selection criterion (selected example)
    Candidate pool: fracA==0.5 groups with all 3 poly_type values present in the test fold (682 per fold, asserted).
    Eligibility: `|gm_err_octamer - gm_err_wdmpnn| <= 0.01` (chemistry placement tied)
    AND `ordering_octamer == 1.0` AND `ordering_wdmpnn == 0.0`.
    Among eligible group-folds: selected group with largest true architecture spread (max - min y_true).

    ## Selected group
    - smiles_A: `{best_gid[0]}`
    - smiles_B: `{best_gid[1]}`
    - fracA: {best_gid[2]}
    - fold: {best_fold}
    - true spread: {best_e["spread"]:.5f} eV  (rank {spread_rank} of {len(ea_pool)} in EA eligible pool)
    - gm_err octamer: {best_e["gm_err_octamer"]:.5f} eV
    - gm_err wDMPNN:  {best_e["gm_err_wdmpnn"]:.5f} eV

    ## Per-fold eligible-pool counts
    | target | f0 | f1 | f2 | f3 | f4 | f5 | f6 | f7 | f8 | total |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | EA | {" | ".join(str(c) for c in ea_counts)} | {sum(ea_counts)} |
    | IP | {" | ".join(str(c) for c in ip_counts)} | {sum(ip_counts)} |

    ## Fold-0 marginals (single-model; fold 0 only)
    | target · model | ordering==0.0 | mean ordering | gm_err decile thr (eV) | joint w/ decile | expected if independent |
    | --- | --- | --- | --- | --- | --- |
    {mrow_f2("EA", "octamer")}
    {mrow_f2("EA", "wdmpnn")}
    {mrow_f2("IP", "octamer")}
    {mrow_f2("IP", "wdmpnn")}

    **Do not quote the conjunction (joint) alone as the failure rate.** The decile condition
    selects 10% of groups by definition (§3 marginals note). The ordering-failure rate is the
    relevant quantity describing how often a model fails to rank architecture.

    ## Median ordering-failure rate across all 9 folds
    | model | EA | IP |
    | --- | --- | --- |
    | octamer | {median_fail[("EA","octamer")]:.1f}% | {median_fail[("IP","octamer")]:.1f}% |
    | wDMPNN  | {median_fail[("EA","wdmpnn")]:.1f}% | {median_fail[("IP","wdmpnn")]:.1f}% |

    EA fold 4 is an outlier: octamer {f4_ea["octamer"]["fail"]:.1f}% failure / mean ordering {f4_ea["octamer"]["mean"]:.3f},
    wDMPNN {f4_ea["wdmpnn"]["fail"]:.1f}% / {f4_ea["wdmpnn"]["mean"]:.3f} — both near chance. Fold not excluded.

    ## Missing files
    None

    ## Architecture-spread recovery (fracA=0.5, arch3 groups)

    ### Selected-group arch_spread_ratio_predavg
    `arch_spread_ratio_predavg` = pred_spread / true_spread, computed from the **three-seed
    prediction average** (seeds 42/43/44 averaged at the prediction level before scoring).
    Distinct from the per-run `arch_spread_ratio_arch3` that appears in
    `_regen_v1_results_individual_runs.csv`, which is computed from a single seed's predictions.
    Those per-run values are not interchangeable with these and must not be quoted as one number.

    | model | true spread (eV) | pred spread (eV) | arch_spread_ratio_predavg |
    | --- | --- | --- | --- |
    | hpg_hier_octamer | {true_spread:.5f} | {spread_oct:.5f} | {ratio_oct:.4f} |
    | wdmpnn | {true_spread:.5f} | {spread_wdm:.5f} | {ratio_wdm:.4f} |

    wDMPNN does not rank the three architectures in the wrong order so much as predict
    nearly the same value for all three — it recovers {ratio_wdm * 100:.0f}% of the true
    architecture range on this group, so its ordering is the sign of residual noise.

    **Strata note (two-sided):** All spread statistics below are for fracA=0.5 groups with
    exactly 3 poly_types present in the test fold. A range over 3 points and a range over
    2 points are not the same quantity; the two strata are never pooled.

    ### Per-fold median arch_spread_ratio (fracA=0.5, arch3, folds 0-8)
    | target · model | f0 | f1 | f2 | f3 | f4 | f5 | f6 | f7 | f8 |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    {sr_row("EA", "octamer")}
    {sr_row("EA", "wdmpnn")}
    {sr_row("IP", "octamer")}
    {sr_row("IP", "wdmpnn")}

    ### Per-fold collapse rate (ratio < 0.25, fracA=0.5, arch3, folds 0-8)
    | target · model | f0 | f1 | f2 | f3 | f4 | f5 | f6 | f7 | f8 |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    {cr_row("EA", "octamer")}
    {cr_row("EA", "wdmpnn")}
    {cr_row("IP", "octamer")}
    {cr_row("IP", "wdmpnn")}

    Octamer collapse rates are asserted within 6.3-18.5% on EA and 6.6-16.0% on IP.
    wDMPNN spot checks: EA fold 1 = {fold_collapse_rates["EA"]["wdmpnn"][1]:.1f}%,
    IP fold 5 = {fold_collapse_rates["IP"]["wdmpnn"][5]:.1f}%,
    IP fold 7 = {fold_collapse_rates["IP"]["wdmpnn"][7]:.1f}%.

    ## Cells: 1 group × 2 models × 3 poly_types × 2 panels = 12 plotted series endpoints.
    """))

# ── F3 — Null floor ───────────────────────────────────────────────────────────

# Cache LOMAO splits once; matches the fold order in metadata/splits/monomer_heldout.json
_LOMAO_TRAIN, _LOMAO_VAL, _LOMAO_TEST, _LOMAO_HELD = None, None, None, None

def _lomao_splits(df):
    global _LOMAO_TRAIN, _LOMAO_VAL, _LOMAO_TEST, _LOMAO_HELD
    if _LOMAO_TRAIN is None:
        _LOMAO_TRAIN, _LOMAO_VAL, _LOMAO_TEST, _LOMAO_HELD = generate_a_held_out_splits(
            df["smiles_A"].astype(str).values, len(df), seed=42, n_splits=9,
            protocol="leave_one_A_out")
        # Verify that generated test folds equal the canonical split file.
        meta = json.loads((ROOT/"metadata"/"splits"/"monomer_heldout.json").read_text())
        for i, rec in enumerate(meta["folds"]):
            expected = np.asarray(rec["global_test_indices"], dtype=int)
            if not np.array_equal(np.sort(_LOMAO_TEST[i]), np.sort(expected)):
                raise AssertionError(f"LOMAO split fold {i} does not match monomer_heldout.json")
    return _LOMAO_TRAIN, _LOMAO_VAL, _LOMAO_TEST, _LOMAO_HELD

def build_f3(df):
    print("F3 — null floor")
    lomao_train, lomao_val, lomao_test, _ = _lomao_splits(df)
    records, missing = [], []

    for fold in range(9):
        # Null: use the same implementation that produced _groupmean_metric_floor.md.
        # Training excludes the held-out A monomer (test) and the A monomer reserved for
        # validation; the runner derives the latter with random seed split_seed+fold.
        for tshort, tcol in [("EA", "EA vs SHE (eV)"), ("IP", "IP vs SHE (eV)")]:
            null_gm = _lomo_null_floor(df, tshort, np.asarray(lomao_train[fold], dtype=int),
                                       np.asarray(lomao_test[fold], dtype=int))
            records.append({"fold": fold, "target": tshort, "model": "null_A_blind",
                            "group_mean_r2": null_gm})
        for mk, mn in MNAMES.items():
            for tshort, ts in [("EA", "EA_vs_SHE_eV"), ("IP", "IP_vs_SHE_eV")]:
                r = avg_seeds(PRED_R1, mn, ts, SPLIT_R1, fold)
                if r is None:
                    missing.append(f"{mn}/{fold}/{ts}")
                    records.append({"fold": fold, "target": tshort, "model": mk, "group_mean_r2": np.nan})
                else:
                    yt, yp, idx = r
                    records.append({"fold": fold, "target": tshort, "model": mk,
                                    "group_mean_r2": metrics(df, yt, yp, idx)["group_mean_r2"]})

    csv = pd.DataFrame(records)
    csv.to_csv(OUTDIR/"f3_null_floor.csv", index=False)

    # Hard assertion against documented _groupmean_metric_floor.md medians.
    for tshort, expected_median in [("EA", 0.67571), ("IP", -0.03401)]:
        vals = csv[(csv.model=="null_A_blind") & (csv.target==tshort)].group_mean_r2.values
        med = float(np.median(vals))
        if abs(med - expected_median) > 5e-4:
            raise AssertionError(
                f"F3 null_A_blind median for {tshort} = {med:.5f}, expected {expected_median:.5f} "
                f"(_groupmean_metric_floor.md). F3 must not be used until reconciled.")
        print(f"  {tshort} null_A_blind median = {med:.5f} (expected {expected_median:.5f}) OK")

    apply_style()
    C = figstyle.COLORS
    mcol  = {"null_A_blind":"#333333","octamer":C[5],"wdmpnn":C[1]}
    mmrk  = {"null_A_blind":"D","octamer":"o","wdmpnn":"s"}
    mlbl  = {"null_A_blind":"A-blind null","octamer":"HPG-octamer","wdmpnn":"wDMPNN"}
    folds = sorted(csv.fold.unique())

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5), sharey=False)
    for ax, tn in zip(axes, ["EA","IP"]):
        sub = csv[csv.target==tn]
        for mk in ["null_A_blind","octamer","wdmpnn"]:
            ms = sub[sub.model==mk].sort_values("fold")
            ax.plot(folds, ms.group_mean_r2.values, color=mcol[mk], marker=mmrk[mk],
                    lw=1.2, ms=5, label=mlbl[mk])
        nv = sub[sub.model=="null_A_blind"].sort_values("fold").group_mean_r2.values
        nm = float(np.nanmedian(nv))
        ax.axhline(nm, color=mcol["null_A_blind"], lw=0.8, ls="--", alpha=0.6)
        ax.annotate(f"null median={nm:.3f}", xy=(folds[-1]-0.3, nm),
                    xytext=(-4,5), textcoords="offset points", ha="right", fontsize=7, color=mcol["null_A_blind"],
                    bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'), zorder=20)
        all_v = sub.group_mean_r2.dropna().values
        ax.set_ylim(max(np.nanmin(all_v)-0.05, -0.5), min(1.02, np.nanmax(all_v)+0.05))
        ax.set_xlim(-0.5, len(folds)-0.5); ax.set_xticks(folds)
        ax.set_xlabel("Fold"); ax.set_ylabel("Group-mean R\u00b2"); ax.set_title(f"{tn}: {TMAP[tn]}")
        leg3 = ax.legend(fontsize=7.5, frameon=True)
        leg3.get_frame().set_facecolor('white'); leg3.get_frame().set_edgecolor('none'); leg3.set_zorder(20)
        ylo, yhi = ax.get_ylim()
        for mk in ["null_A_blind", "octamer", "wdmpnn"]:
            for _, sr in sub[sub.model == mk].sort_values("fold").iterrows():
                v = sr.group_mean_r2
                if not np.isnan(v) and v < ylo:
                    fi = int(sr.fold)
                    ax.scatter([fi], [ylo], marker="v", color=mcol[mk], s=36,
                               zorder=10, clip_on=False)
                    ax.text(fi, ylo + (yhi - ylo) * 0.015, f"{v:.2f}",
                            ha="center", va="bottom", fontsize=5.5, color=mcol[mk],
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'), zorder=20)
        for fi in folds:
            fn = sub[(sub.model=="null_A_blind")&(sub.fold==fi)].group_mean_r2.values
            fm = sub[(sub.model!="null_A_blind")&(sub.fold==fi)].group_mean_r2.dropna().values
            if len(fn) and len(fm) and fn[0]>np.nanmin(fm):
                ax.axvspan(fi-0.4, fi+0.4, color=C[4], alpha=0.18, zorder=0)

    fig.suptitle("A-split (monomer_heldout): A-blind null vs trained models — group-mean R\u00b2", fontsize=9)
    plt.tight_layout()
    save(fig, "f3_null_floor")

    miss_str = "\n".join(f"- {f}" for f in missing) if missing else "None"
    val_summary = {i: len(lomao_val[i]) for i in range(9)}
    held_summary = {i: _LOMAO_HELD[i] for i in range(9)}
    write_manifest("f3", dedent(f"""
    # F3 — Null floor

    ## Prediction source
    `{PRED_R1.relative_to(ROOT)}/ea_ip__[target]__[model]__{SPLIT_R1}__fold[0-8]__s[42/43/44].npz`
    Seeds 42,43,44 averaged. Models: hpg_hier_octamer, wdmpnn. Split: {SPLIT_R1} (9 folds).

    ## Null predictor
    A-blind group-mean lookup by (smiles_B, fracA, poly_type), then by (smiles_B, poly_type),
    then by global training mean. This is the exact `null_floor()` function from
    `scripts/python/aggregate_lomo_seeds.py`, which produced `analysis/model_diagnostics/_groupmean_metric_floor.md`.

    ## Training-set construction
    Training indices for fold k are generated by `scripts/python/utils.py::generate_a_held_out_splits`
    (`protocol='leave_one_A_out'`, `seed=42`, `n_splits=9`). This is the same generator the runner uses:
    - Test fold = one held-out monomer A (4774 rows).
    - Validation fold = one randomly chosen A monomer from the remaining 8 (4774 rows), selected with
      `np.random.default_rng(seed + k)`.
    - Training fold = the remaining 7 A monomers (33418 rows).
    The generated test folds were checked against `metadata/splits/monomer_heldout.json` and match.
    Held-out A per fold: {held_summary}

    ## Null model label
    `null_A_blind` (changed from `null` to avoid NaN grouping in downstream CSV).

    ## Metric function
    `aggregate_lomo_seeds.null_floor` uses the same matched-group grouping as `compute_copolymer_metrics`.

    ## Missing files
    {miss_str}

    ## Cells: {len(folds)} folds × 2 targets × 3 series = {len(folds)*2*3} values.
    """))

# ── F4 — Split design ─────────────────────────────────────────────────────────

def build_f4(df):
    print("F4 — split design")
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        has_rdkit = True
    except ImportError:
        has_rdkit = False

    comp_csv = ROOT/"analysis"/"model_diagnostics"/"_octamer_k1_r3_results_fold_composition.csv"
    fc = pd.read_csv(comp_csv)
    s_folds = sorted(fc[fc.fold_group=="S_within_scaffold"].fold.tolist())
    d_folds = sorted(fc[fc.fold_group=="D_cross_scaffold"].fold.tolist())
    fc[["fold","same_scaffold_share","fold_group"]].to_csv(OUTDIR/"f4_fold_scaffold_share.csv", index=False)

    family_sizes = None
    top2_share   = np.nan
    if has_rdkit:
        b_smis = df["smiles_B"].dropna().unique().tolist()
        scafs  = {}
        for smi in b_smis:
            mol = Chem.MolFromSmiles(smi)
            s   = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else ""
            scafs[smi] = s or f"ACYCLIC::{smi}"
        fs = pd.Series(scafs).value_counts().sort_values(ascending=False)
        family_sizes = fs
        top2_share   = fs.iloc[:2].sum() / len(b_smis)
        pd.DataFrame({"scaffold":fs.index,"n_monomers":fs.values}).to_csv(
            OUTDIR/"f4_scaffold_families.csv", index=False)

    apply_style()
    C  = figstyle.COLORS
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.5))

    if has_rdkit and family_sizes is not None:
        sizes = family_sizes.values
        ax1.hist(sizes, bins=range(1, int(max(sizes))+2), color=C[5], edgecolor="white", lw=0.3)
        ax1.set_xlabel("Family size (# B monomers per Murcko scaffold)")
        ax1.set_ylabel("Number of scaffold families")
        ax1.set_title(f"B-monomer scaffold distribution (n={len(b_smis)})")
        t1, t2 = int(family_sizes.iloc[0]), int(family_sizes.iloc[1])
        ax1.annotate(f"Top-2 families: {t1} + {t2} monomers\n({top2_share*100:.1f}% of all B monomers)",
                     xy=(0.45,0.78), xycoords="axes fraction", fontsize=8)
    else:
        ax1.text(0.5, 0.5, "rdkit unavailable", ha="center", va="center", transform=ax1.transAxes)

    for _, row in fc.iterrows():
        c = C[2] if "S_within" in row.fold_group else C[1]
        ax2.bar(row.fold, row.same_scaffold_share, color=c, alpha=0.75, width=0.6)
    sp = mpatches.Patch(color=C[2], alpha=0.75, label=f"S folds {s_folds}")
    dp = mpatches.Patch(color=C[1], alpha=0.75, label=f"D folds {d_folds}")
    ax2.set_xlabel("Fold"); ax2.set_ylabel("Fraction of held-out B monomers\nwith same-scaffold relative in training")
    ax2.set_title("B-split: scaffold overlap by fold")
    ax2.set_xticks(sorted(fc.fold.unique())); ax2.set_ylim(0,1.1)
    ax2.legend(handles=[sp,dp], fontsize=8, frameon=False)

    fig.suptitle("Split design: Murcko scaffolds — monomer_b_heldout_clustered", fontsize=9)
    plt.tight_layout()
    save(fig, "f4_split_design")

    write_manifest("f4", dedent(f"""
    # F4 — Split design

    ## Left panel
    Unique smiles_B from `data/ea_ip.csv` ({int(df.smiles_B.nunique())} B monomers).
    Murcko scaffold via `rdkit.Chem.Scaffolds.MurckoScaffold.MurckoScaffoldSmiles(includeChirality=False)`.
    rdkit available: {has_rdkit}.

    ## Right panel
    `analysis/model_diagnostics/_octamer_k1_r3_results_fold_composition.csv`
    Column `same_scaffold_share` = fraction of held-out B monomers whose Murcko scaffold
    appears among training B monomers. S folds: {s_folds}. D folds: {d_folds}.

    ## Cells (right): {len(fc)} folds × 1 value each.
    """))

# ── F5 — Noise floor ─────────────────────────────────────────────────────────

_RPT = pd.DataFrame([
    {"fold":0,"repeat":1,"group_mean_r2":0.96202091,"delta_r2":0.76442090,"overall_r2":0.95798297,"mae":0.08380947},
    {"fold":0,"repeat":2,"group_mean_r2":0.98242197,"delta_r2":0.82993524,"overall_r2":0.98019690,"mae":0.05459051},
    {"fold":0,"repeat":3,"group_mean_r2":0.98550585,"delta_r2":0.78295383,"overall_r2":0.98320852,"mae":0.05210783},
    {"fold":1,"repeat":1,"group_mean_r2":0.78978591,"delta_r2":0.73827110,"overall_r2":0.78749832,"mae":0.14560965},
    {"fold":1,"repeat":2,"group_mean_r2":0.44970180,"delta_r2":0.65505979,"overall_r2":0.44059472,"mae":0.22635591},
    {"fold":1,"repeat":3,"group_mean_r2":0.97815400,"delta_r2":0.82057898,"overall_r2":0.97359665,"mae":0.04459828},
])
_SUM = pd.DataFrame([
    {"fold":0,"gm_r2_sd":0.01276228,"dr2_sd":0.03377093,"or2_sd":0.01377714,"mae_sd":0.01763002},
    {"fold":1,"gm_r2_sd":0.26783125,"dr2_sd":0.08276001,"or2_sd":0.27051363,"mae_sd":0.09106691},
])

def build_f5(df):
    print("F5 — noise floor")
    seed_rows = []
    for mk, mn in MNAMES.items():
        for tn, ts in TSAFE.items():
            for fold in range(9):
                per = []
                for s in SEEDS:
                    d = load_npz(PRED_R1, mn, ts, SPLIT_R1, fold, s)
                    if d is None: continue
                    per.append(metrics(df, d["y_true"], d["y_pred"], d["idx"])["delta_r2"])
                if len(per)==3:
                    seed_rows.append({"model":mk,"target":tn,"fold":fold,
                                      "delta_r2_sd":float(np.std(per,ddof=1)),
                                      "delta_r2_mean":float(np.mean(per))})
    sd_df = pd.DataFrame(seed_rows)
    sd_df.to_csv(OUTDIR/"f5_seed_delta_r2_sd.csv", index=False)
    _RPT.to_csv(OUTDIR/"f5_noise_floor_repeats.csv", index=False)

    apply_style()
    C = figstyle.COLORS
    fc = {0:C[5], 1:C[1]}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.5))

    ax1b = ax1.twinx()
    for fi in [0,1]:
        sub = _RPT[_RPT.fold==fi].sort_values("repeat")
        xs  = sub.repeat.values + (fi-0.5)*0.15
        ax1.plot(xs, sub.group_mean_r2.values, color=fc[fi], marker="o", lw=1.0, ms=6,
                 label=f"gm-R² fold {fi}")
        ax1b.plot(xs, sub.mae.values, color=fc[fi], marker="s", lw=1.0, ms=6, ls="--",
                  label=f"MAE fold {fi}")
    # The SD values used to be annotated inside the axes. ax1b is a twinx of ax1,
    # and a twin axes is drawn entirely above its parent, so an artist zorder on
    # ax1 can never lift text above an ax1b line -- bbox+zorder could not fix it.
    # Moving the values below the axes removes the collision by construction.
    _sd_txt = "     ".join(
        f"fold {int(r.fold)}:  gm-R² SD {r.gm_r2_sd:.3f}   MAE SD {r.mae_sd:.3f} eV"
        for _, r in _SUM.sort_values("fold").iterrows())
    ax1.set_xticks([1,2,3])
    ax1.set_xlabel("Repeat\n" + _sd_txt, fontsize=7.5, linespacing=1.9)
    ax1.set_ylabel("Group-mean R²"); ax1b.set_ylabel("MAE (eV)")
    ax1.set_title("Repeat study: HPG-octamer A-split EA (seed 42)")
    h1,l1 = ax1.get_legend_handles_labels(); h2,l2 = ax1b.get_legend_handles_labels()
    # "upper left" sat under the gm-R² fold 0 trace, which runs flat near 0.98.
    leg5 = ax1.legend(h1+h2, l1+l2, fontsize=7, frameon=True, loc="center left")
    leg5.get_frame().set_facecolor('white'); leg5.get_frame().set_edgecolor('none'); leg5.set_zorder(20)

    if len(sd_df):
        pos, xlbls, xpos = 0, [], []
        for tn in ["EA","IP"]:
            for mk in ["octamer","wdmpnn"]:
                sds = sd_df[(sd_df.model==mk)&(sd_df.target==tn)].delta_r2_sd.dropna().values
                c   = C[5] if tn=="EA" else C[1]
                ax2.scatter([pos]*len(sds), sds, color=c, alpha=0.65, s=28, zorder=5)
                if len(sds):
                    ax2.plot([pos-0.3,pos+0.3],[np.median(sds)]*2, color="#333", lw=1.8, zorder=6)
                xpos.append(pos); xlbls.append(f"{mk}\n{tn}")
                pos += 1
            pos += 0.4
        ax2.set_xticks(xpos); ax2.set_xticklabels(xlbls, fontsize=7)
        ax2.set_ylabel("Across-seed SD of delta_r2 (3 seeds)")
        ax2.set_title("Seed variability: delta_r2 (R1, all 9 folds)")
        om = float(np.nanmedian(sd_df.delta_r2_sd)); ax2.axhline(0, color="#bbb", lw=0.5)
        ax2.axhline(om, color="#333", lw=0.8, ls="--")
        ax2.annotate(f"overall median={om:.4f}", xy=(0.02,0.93), xycoords="axes fraction", fontsize=7,
                     bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'), zorder=20)

    fig.suptitle("Noise floor: repeat study (left) and cross-seed delta_r2 SD (right)", fontsize=8.5)
    plt.tight_layout()
    save(fig, "f5_noise_floor")

    write_manifest("f5", dedent(f"""
    # F5 — Noise floor

    ## Left panel
    Hard-coded from `analysis/model_diagnostics/_noise_floor_results.md`.
    6 runs: HPG-octamer, A-split EA, seed 42, 2 folds × 3 repeats, V100 GPU.

    ## Right panel
    `{PRED_R1.relative_to(ROOT)}/ea_ip__[target]__[model]__{SPLIT_R1}__fold[0-8]__s[42/43/44].npz`
    Per-seed delta_r2 via `compute_copolymer_metrics`, then sample SD (ddof=1) across 3 seeds.
    Cells available: {len(sd_df)} of {9*2*2} expected.
    Overall median SD: {float(np.nanmedian(sd_df.delta_r2_sd)) if len(sd_df) else 'N/A':.4f}
    """))

# ── F6 — Demonstration paired-difference ─────────────────────────────────────

def build_f6(df):
    print("F6 — demonstration")
    fc_csv = ROOT/"analysis"/"model_diagnostics"/"_octamer_k1_r3_results_fold_composition.csv"
    fc     = pd.read_csv(fc_csv)
    d_folds = sorted(fc[fc.fold_group=="D_cross_scaffold"].fold.tolist())
    print(f"  D folds: {d_folds}")

    METS = ["overall_r2","mae","rmse","group_mean_r2","delta_r2"]
    rows, missing = [], []

    for fold in d_folds:
        for tn, ts in TSAFE.items():
            row = {"fold":fold,"target":tn}
            for mk, mn in MNAMES.items():
                r = avg_seeds(PRED_R3, mn, ts, SPLIT_R3, fold)
                if r is None:
                    missing.append(f"{mn}/{fold}/{ts}")
                    for m in METS: row[f"{mk}_{m}"] = np.nan
                else:
                    yt, yp, idx = r
                    ms = metrics(df, yt, yp, idx)
                    for m in METS: row[f"{mk}_{m}"] = ms[m]
            rows.append(row)

    full = pd.DataFrame(rows)
    for m in METS:
        full[f"diff_{m}"] = full[f"octamer_{m}"] - full[f"wdmpnn_{m}"]
    full.to_csv(OUTDIR/"f6_paired_differences.csv", index=False)

    # Summary + expected-value check
    EXPECTED = {
        "EA":{"overall_r2":(0.0034,3),"mae":(-0.0003,3),"rmse":(-0.0044,3),
              "group_mean_r2":(0.0022,3),"delta_r2":(0.2597,5)},
        "IP":{"overall_r2":(0.0098,3),"mae":(-0.0096,3),"rmse":(-0.0117,3),
              "group_mean_r2":(0.0072,3),"delta_r2":(0.1518,5)},
    }
    summary_rows, discrepancies = [], []
    print("  Computed vs expected:")
    for tn in ["EA","IP"]:
        sub = full[full.target==tn]
        for m in METS:
            diffs = sub[f"diff_{m}"].dropna().values
            wins  = int(np.sum(diffs<0)) if m in ["mae","rmse"] else int(np.sum(diffs>0))
            med   = float(np.nanmedian(diffs)) if len(diffs) else np.nan
            em, ew = EXPECTED[tn][m]
            ok = abs(med-em)<0.0005 and wins==ew if not np.isnan(med) else False
            tag = "OK" if ok else "DISCREPANCY"
            print(f"    {tn} {m}: median={med:.4f} wins={wins}/{len(diffs)} | exp {em} {ew}/5 → {tag}")
            if not ok:
                discrepancies.append(f"- {tn} {m}: got median={med:.4f} wins={wins} vs expected median={em} wins={ew}")
            summary_rows.append({"target":tn,"metric":m,"median_diff":med,
                                  "n_folds":len(diffs),"wins":wins,
                                  "expected_median":em,"expected_wins":ew,"match":ok})
    pd.DataFrame(summary_rows).to_csv(OUTDIR/"f6_summary.csv", index=False)

    # — Plot ——————————————————————————————————————
    apply_style()
    C = figstyle.COLORS
    mlbl = {"overall_r2":"overall R²","mae":"MAE","rmse":"RMSE",
            "group_mean_r2":"group-mean R²","delta_r2":"delta R²"}
    tc   = {"EA":C[5],"IP":C[1]}
    tmk  = {"EA":"o","IP":"s"}

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for ax, tn in zip(axes, ["EA","IP"]):
        ax.axhline(0, color="#555", lw=0.9, ls="--", zorder=0)
        sub = full[full.target==tn]
        for pi, m in enumerate(METS):
            diffs = sub[f"diff_{m}"].dropna().values
            wins  = int(np.sum(diffs<0)) if m in ["mae","rmse"] else int(np.sum(diffs>0))
            med   = float(np.nanmedian(diffs)) if len(diffs) else np.nan
            c = tc[tn]
            for dv in diffs:
                ax.scatter(pi, dv, color=c, alpha=0.55, s=38, marker=tmk[tn], zorder=5)
            if not np.isnan(med):
                ax.plot([pi-0.3,pi+0.3],[med,med], color=c, lw=2.2, zorder=6)
            ax.annotate(f"{wins}/{len(diffs)}",
                        xy=(pi, 0), xycoords=("data", "axes fraction"),
                        xytext=(0, -14), textcoords="offset points",
                        ha="center", fontsize=7, color=c)
        ax.set_xticks(range(len(METS)))
        ax.set_xticklabels([mlbl[m] for m in METS], rotation=20, ha="right", fontsize=7.5)
        ax.set_ylabel("octamer − wDMPNN (paired per fold)")
        ax.set_title(f"{tn}: D folds {d_folds}")

    fig.suptitle(f"B-split D folds: octamer vs wDMPNN — paired differences (3 seeds averaged)",
                 fontsize=9)
    plt.tight_layout()
    save(fig, "f6_demonstration")

    disc_str = "\n".join(discrepancies) if discrepancies else "None — all values match expected."
    miss_str = "\n".join(f"- {f}" for f in missing) if missing else "None"
    write_manifest("f6", dedent(f"""
    # F6 — Demonstration: paired differences

    ## Prediction source
    `{PRED_R3.relative_to(ROOT)}/ea_ip__[target]__[model]__{SPLIT_R3}__fold[4-8]__s[42/43/44].npz`
    Seeds 42,43,44 averaged at prediction level.
    Models: hpg_hier_octamer (octamer), wdmpnn.
    D folds: {d_folds} (cross-scaffold fold group).

    ## Metric function
    `compute_copolymer_metrics` → overall_r2, mae, rmse, group_mean_r2, delta_r2.
    Paired difference = octamer − wDMPNN per fold. Win = octamer better:
    higher for R² metrics, lower for MAE/RMSE.

    ## Missing files
    {miss_str}

    ## Expected vs computed
    {disc_str}

    ## Cells: {len(d_folds)} D folds × 2 targets × 5 metrics = {len(d_folds)*2*5} values.
    """))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading data/ea_ip.csv ...")
    df = pd.read_csv(ROOT / "data" / "ea_ip.csv")
    print(f"  {len(df)} rows, columns: {list(df.columns)}")

    build_f1(df)
    build_f2(df)
    build_f3(df)
    build_f4(df)
    build_f5(df)
    build_f6(df)
    print(f"\nAll figures written to {OUTDIR.relative_to(ROOT)}/")

if __name__ == "__main__":
    main()
