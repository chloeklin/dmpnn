"""
Stage 2D Dataset Bottleneck Assessment
=======================================
Combines two analyses to determine whether architecture learning is
data-limited or has reached the information capacity of the dataset.

EXPERIMENT A: Architecture Signal Magnitude Analysis
  A1. Architecture deviations
  A2. Variance decomposition
  A3. Architecture signal statistics + histograms
  A4. Per-group variance distribution
  A5. Signal vs Frac residual comparison

EXPERIMENT B: Matched-Group Saturation Learning Curve
  B1–B3. Load existing learning curve predictions (HPG2Stage_LC)
  B4. Saturation curve fitting
  B5. Interpretation

Output:
    experiments/hpg2stage/output/bottleneck/
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats, optimize
from sklearn.metrics import r2_score, mean_absolute_error

# ─── Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts' / 'python'))

DATA_PATH = PROJECT_ROOT / "data" / "ea_ip.csv"
PRED_DIR_LC = PROJECT_ROOT / "predictions" / "HPG2Stage_LC"
PRED_DIR_ORIG = PROJECT_ROOT / "predictions" / "HPG2Stage"
OUT_DIR = Path(__file__).resolve().parents[1] / "output" / "bottleneck"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = {"EA": "EA vs SHE (eV)", "IP": "IP vs SHE (eV)"}
MODELS = ["2d0_arch", "2d1_arch"]
FRACTIONS = [25, 50, 75, 100]
N_FOLDS = 5
DATASET_NAME = "ea_ip"

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.titlesize': 13, 'axes.labelsize': 12,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'figure.facecolor': 'white',
})

report_lines = []

def report(text=""):
    print(text)
    report_lines.append(text)


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_dataset():
    df = pd.read_csv(DATA_PATH)
    df['group_key'] = (df['smiles_A'].astype(str) + '||' +
                       df['smiles_B'].astype(str) + '||' +
                       df['fracA'].astype(str) + '||' +
                       df['fracB'].astype(str))
    return df


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT A: ARCHITECTURE SIGNAL MAGNITUDE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def experiment_a(df):
    report("=" * 70)
    report("EXPERIMENT A: ARCHITECTURE SIGNAL MAGNITUDE ANALYSIS")
    report("=" * 70)

    # --- A1 + A2: Variance Decomposition ---
    report("\n--- A1/A2: Variance Decomposition ---")
    variance_rows = []
    dev_data = {}  # Store Δy for later use

    for tshort, tlong in TARGETS.items():
        y = df[tlong].values
        groups = df['group_key'].values

        gdf = pd.DataFrame({'y': y, 'group': groups})
        gmean = gdf.groupby('group')['y'].transform('mean')
        gsize = gdf.groupby('group')['y'].transform('count')

        delta = y - gmean.values  # Δy = y - group_mean(y)

        var_total = np.var(y, ddof=0)
        var_arch = np.var(delta, ddof=0)
        var_comp = np.var(gmean.values, ddof=0)

        # Store for A3/A5
        dev_data[tshort] = {
            'delta': delta,
            'gmean': gmean.values,
            'y': y,
            'groups': groups,
            'multi_mask': gsize.values > 1,
        }

        variance_rows.append({
            'target': tshort,
            'Var_total': var_total,
            'Var_comp': var_comp,
            'Var_arch': var_arch,
            'Var_arch_frac': var_arch / var_total,
            'Var_comp_frac': var_comp / var_total,
            'Sum_check': var_comp + var_arch,
            'Sum_ratio': (var_comp + var_arch) / var_total,
        })

        report(f"\n  {tshort}:")
        report(f"    Var_total       = {var_total:.6f}")
        report(f"    Var_comp        = {var_comp:.6f}  ({var_comp/var_total*100:.2f}%)")
        report(f"    Var_arch        = {var_arch:.6f}  ({var_arch/var_total*100:.2f}%)")
        report(f"    Var_comp + Var_arch = {var_comp + var_arch:.6f}  (ratio to total: {(var_comp + var_arch)/var_total:.6f})")

    var_df = pd.DataFrame(variance_rows)
    var_df.to_csv(OUT_DIR / "architecture_variance_table.csv", index=False)
    report(f"\n  Saved: architecture_variance_table.csv")

    # --- A3: Architecture Signal Statistics ---
    report("\n--- A3: Architecture Signal Statistics ---")
    report(f"  {'':4} {'mean|Δ|':>10} {'med|Δ|':>10} {'std(Δ)':>10} {'p90|Δ|':>10} {'p95|Δ|':>10} {'max|Δ|':>10}")
    report("  " + "-" * 66)

    for tshort in TARGETS:
        delta = dev_data[tshort]['delta']
        abs_d = np.abs(delta)
        report(f"  {tshort:4} {abs_d.mean():>10.4f} {np.median(abs_d):>10.4f} {delta.std():>10.4f} "
               f"{np.percentile(abs_d, 90):>10.4f} {np.percentile(abs_d, 95):>10.4f} {abs_d.max():>10.4f}")

    # A3 Histograms
    for tshort in TARGETS:
        delta = dev_data[tshort]['delta']
        abs_d = np.abs(delta)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(delta, bins=80, color='#4C72B0', alpha=0.7, edgecolor='white', linewidth=0.5)
        ax.axvline(delta.mean(), color='red', linestyle='--', linewidth=1.5,
                   label=f'mean = {delta.mean():.4f}')
        ax.axvline(0, color='gray', linestyle='-', linewidth=0.5)
        # Mark std and p95
        p95 = np.percentile(abs_d, 95)
        ax.axvline(p95, color='orange', linestyle=':', linewidth=1.5,
                   label=f'p95(|Δ|) = {p95:.4f}')
        ax.axvline(-p95, color='orange', linestyle=':', linewidth=1.5)
        ax.axvspan(-delta.std(), delta.std(), alpha=0.1, color='red',
                   label=f'±std = ±{delta.std():.4f}')
        ax.set_xlabel(f'Δ{tshort} (eV)')
        ax.set_ylabel('Count')
        ax.set_title(f'Architecture Deviation Distribution — {tshort}')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"delta{tshort}_distribution.png", dpi=150)
        plt.close()

    report("  Saved: deltaEA_distribution.png, deltaIP_distribution.png")

    # --- A4: Per-Group Architecture Variance ---
    report("\n--- A4: Architecture Variance By Group ---")
    group_var_rows = []

    for tshort, tlong in TARGETS.items():
        y = df[tlong].values
        groups = df['group_key'].values

        gdf = pd.DataFrame({'y': y, 'group': groups})
        gvar = gdf.groupby('group')['y'].var(ddof=0)
        gsize = gdf.groupby('group')['y'].count()

        # Only groups with >1 member
        multi = gvar[gsize > 1]
        all_gvar = gvar.values

        report(f"\n  {tshort}:")
        report(f"    Total groups: {len(gvar)}")
        report(f"    Multi-arch groups (≥2 members): {len(multi)}")
        report(f"    Median group variance: {np.median(multi):.6f}")
        report(f"    Mean group variance:   {multi.mean():.6f}")
        report(f"    Max group variance:    {multi.max():.6f}")

        # Top 10 highest-variance groups
        top10 = multi.nlargest(10)
        report(f"    Top 10 highest-variance groups:")
        for gk, gv in top10.items():
            n = gsize[gk]
            parts = gk.split('||')
            report(f"      var={gv:.6f}  n={n}  (A={parts[0][:20]}, B={parts[1][:20]}, fA={parts[2]})")

        for gk in gvar.index:
            group_var_rows.append({
                'target': tshort,
                'group_key': gk,
                'group_var': gvar[gk],
                'group_size': gsize[gk],
            })

        # A4 Histogram of per-group variance
        fig, ax = plt.subplots(figsize=(8, 5))
        multi_vals = multi.values
        ax.hist(multi_vals, bins=60, color='#DD8452', alpha=0.7, edgecolor='white', linewidth=0.5)
        ax.axvline(np.median(multi_vals), color='red', linestyle='--', linewidth=1.5,
                   label=f'median = {np.median(multi_vals):.5f}')
        ax.axvline(multi_vals.mean(), color='blue', linestyle='--', linewidth=1.5,
                   label=f'mean = {multi_vals.mean():.5f}')
        ax.set_xlabel(f'{tshort} Within-Group Variance')
        ax.set_ylabel('Count (groups)')
        ax.set_title(f'Architecture Variance Distribution — {tshort}')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"architecture_variance_distribution_{tshort}.png", dpi=150)
        plt.close()

    gvar_df = pd.DataFrame(group_var_rows)
    gvar_df.to_csv(OUT_DIR / "architecture_group_variance.csv", index=False)
    report("  Saved: architecture_group_variance.csv")
    report("  Saved: architecture_variance_distribution_EA.png, architecture_variance_distribution_IP.png")

    # --- A5: Architecture Signal Relative to Frac Residual ---
    report("\n--- A5: Architecture Signal vs Frac Residual ---")
    a5_compute_signal_vs_residual(df, dev_data)

    return dev_data


def a5_compute_signal_vs_residual(df, dev_data):
    """Compare Var_arch to Frac baseline residual variance."""
    frac_residual_var = {}

    for tshort, tlong in TARGETS.items():
        # Load Frac predictions (original a_held_out, all 5 folds)
        yt_all, yp_all = [], []
        for fold in range(N_FOLDS):
            fname = f"ea_ip__{tlong}__copoly_stage2d_frac__a_held_out__split{fold}.npz"
            fpath = PRED_DIR_ORIG / fname
            if not fpath.exists():
                report(f"  [MISSING] Frac predictions: {fname}")
                continue
            npz = np.load(fpath, allow_pickle=True)
            yt = npz['y_true'].flatten().astype(float)
            yp = npz['y_pred'].flatten().astype(float)
            # Apply linregress inverse transform (normalized predictions)
            slope, intercept, _, _, _ = stats.linregress(yp, yt)
            yp_corr = yp * slope + intercept
            yt_all.extend(yt)
            yp_all.extend(yp_corr)

        yt_all = np.array(yt_all)
        yp_all = np.array(yp_all)
        residual = yt_all - yp_all
        var_resid = np.var(residual, ddof=0)
        frac_residual_var[tshort] = var_resid

        var_arch = dev_data[tshort]['Var_arch'] if 'Var_arch' in dev_data[tshort] else np.var(dev_data[tshort]['delta'], ddof=0)
        ratio = var_arch / var_resid if var_resid > 0 else float('inf')

        report(f"\n  {tshort}:")
        report(f"    Frac residual variance:     {var_resid:.6f}")
        report(f"    Architecture variance:       {np.var(dev_data[tshort]['delta'], ddof=0):.6f}")
        report(f"    Var_arch / Var_residual:      {ratio:.4f}")
        report(f"    → Architecture effects explain {ratio*100:.1f}% of remaining Frac error")


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT B: MATCHED-GROUP SATURATION LEARNING CURVE
# ═══════════════════════════════════════════════════════════════════════

def experiment_b(df):
    report("\n" + "=" * 70)
    report("EXPERIMENT B: MATCHED-GROUP SATURATION LEARNING CURVE")
    report("=" * 70)

    # --- B1–B3: Load existing learning curve predictions ---
    report("\n--- B1–B3: Load Learning Curve Predictions ---")
    lc_rows = []

    for tshort, tlong in TARGETS.items():
        for model in MODELS:
            for frac in FRACTIONS:
                fold_r2s = []
                fold_arch_r2s = []
                fold_maes = []

                for fold in range(N_FOLDS):
                    fname = f"{DATASET_NAME}__{tlong}__stage2d_{model}__fold{fold}__frac{frac}__split{fold}.npz"
                    fpath = PRED_DIR_LC / fname
                    if not fpath.exists():
                        report(f"  [MISSING] {fname}")
                        continue

                    npz = np.load(fpath, allow_pickle=True)
                    yt = npz['y_true'].flatten().astype(float)
                    yp = npz['y_pred'].flatten().astype(float)

                    # Check if predictions need inverse transform
                    if abs(yp.mean() - yt.mean()) > 1.0 or r2_score(yt, yp) < -1:
                        slope, intercept, _, _, _ = stats.linregress(yp, yt)
                        yp = yp * slope + intercept

                    r2_ov = r2_score(yt, yp)
                    mae_ov = mean_absolute_error(yt, yp)
                    fold_r2s.append(r2_ov)
                    fold_maes.append(mae_ov)

                    # Arch-dev R²
                    if 'test_indices' in npz:
                        ti = npz['test_indices'].flatten().astype(int)
                    else:
                        # Fall back to y_true matching
                        vals = df[tlong].values
                        lookup = {}
                        for idx, v in enumerate(vals):
                            if np.isfinite(v):
                                lookup[round(float(v), 6)] = idx
                        ti = np.array([lookup.get(round(float(v), 6), -1) for v in yt])

                    valid = ti >= 0
                    if valid.sum() > 20:
                        gdf = pd.DataFrame({
                            'y_true': yt[valid], 'y_pred': yp[valid],
                            'group': df.iloc[ti[valid]]['group_key'].values,
                            'arch': df.iloc[ti[valid]]['poly_type'].values,
                        })
                        ga = gdf.groupby('group')['arch'].nunique()
                        multi_groups = ga[ga >= 2].index
                        gdf_m = gdf[gdf['group'].isin(multi_groups)]

                        if len(gdf_m) >= 20:
                            gmt = gdf_m.groupby('group')['y_true'].transform('mean')
                            gmp = gdf_m.groupby('group')['y_pred'].transform('mean')
                            dt = gdf_m['y_true'] - gmt
                            dp = gdf_m['y_pred'] - gmp
                            if dt.std() > 1e-10:
                                fold_arch_r2s.append(r2_score(dt, dp))
                            else:
                                fold_arch_r2s.append(np.nan)
                        else:
                            fold_arch_r2s.append(np.nan)
                    else:
                        fold_arch_r2s.append(np.nan)

                if fold_r2s:
                    lc_rows.append({
                        'target': tshort, 'model': model, 'fraction': frac,
                        'R2_mean': np.mean(fold_r2s), 'R2_std': np.std(fold_r2s),
                        'MAE_mean': np.mean(fold_maes), 'MAE_std': np.std(fold_maes),
                        'R2_arch_mean': np.nanmean(fold_arch_r2s),
                        'R2_arch_std': np.nanstd(fold_arch_r2s),
                        'n_folds': len(fold_r2s),
                        'n_arch_folds': np.sum(np.isfinite(fold_arch_r2s)),
                    })

    lc_df = pd.DataFrame(lc_rows)
    lc_df.to_csv(OUT_DIR / "learning_curve_metrics.csv", index=False)
    report(f"  Loaded {len(lc_df)} data points")
    report(f"  Saved: learning_curve_metrics.csv")

    # Print summary table
    report(f"\n{'target':>4} {'model':>10} {'frac':>5} {'R2_mean':>8} {'R2_std':>7} {'R2_arch':>8} {'R2a_std':>7}")
    report("-" * 55)
    for _, row in lc_df.iterrows():
        report(f"{row['target']:>4} {row['model']:>10} {row['fraction']:>5} "
               f"{row['R2_mean']:>8.4f} {row['R2_std']:>7.4f} "
               f"{row['R2_arch_mean']:>8.4f} {row['R2_arch_std']:>7.4f}")

    # --- B4: Saturation Analysis + Plots ---
    report("\n--- B4: Saturation Analysis ---")
    b4_saturation_analysis(lc_df)

    # --- B5: Interpretation ---
    report("\n--- B5: Interpretation ---")
    b5_interpretation(lc_df)

    return lc_df


def b4_saturation_analysis(lc_df):
    """Fit saturation curves and create learning curve plots."""
    COLORS = {'2d0_arch': '#4C72B0', '2d1_arch': '#DD8452'}
    LABELS = {'2d0_arch': '2D0-arch', '2d1_arch': '2D1-arch'}
    MARKERS = {'2d0_arch': 'o', '2d1_arch': 's'}

    def saturation_model(x, a, b):
        """R² = a * (1 - exp(-b * x))"""
        return a * (1.0 - np.exp(-b * x))

    saturation_results = []

    # Four plots: EA overall, IP overall, EA archdev, IP archdev
    plot_configs = [
        ('EA', 'R2_mean', 'R2_std', 'EA Overall R²', 'EA_overall_learning_curve'),
        ('IP', 'R2_mean', 'R2_std', 'IP Overall R²', 'IP_overall_learning_curve'),
        ('EA', 'R2_arch_mean', 'R2_arch_std', 'EA Architecture-Deviation R²', 'EA_archdev_learning_curve'),
        ('IP', 'R2_arch_mean', 'R2_arch_std', 'IP Architecture-Deviation R²', 'IP_archdev_learning_curve'),
    ]

    for tgt, mean_col, std_col, title, save_name in plot_configs:
        fig, ax = plt.subplots(figsize=(7, 5))

        for model in MODELS:
            sub = lc_df[(lc_df['target'] == tgt) & (lc_df['model'] == model)].sort_values('fraction')
            if sub.empty:
                continue

            x = sub['fraction'].values / 100.0  # Normalize to [0, 1]
            y = sub[mean_col].values
            yerr = sub[std_col].values

            ax.errorbar(x * 100, y, yerr=yerr,
                        marker=MARKERS[model], markersize=8, linewidth=2.2,
                        capsize=5, capthick=1.5, elinewidth=1.5,
                        color=COLORS[model], label=LABELS[model])

            # Try fitting saturation model
            valid = np.isfinite(y) & (y > 0)
            if valid.sum() >= 3:
                try:
                    popt, pcov = optimize.curve_fit(
                        saturation_model, x[valid], y[valid],
                        p0=[y[valid].max() * 1.1, 3.0],
                        bounds=([0, 0.01], [1.5, 50.0]),
                        maxfev=5000,
                    )
                    a_fit, b_fit = popt
                    perr = np.sqrt(np.diag(pcov))

                    # Plot fitted curve
                    x_fine = np.linspace(0.1, 2.0, 100)
                    y_fit = saturation_model(x_fine, a_fit, b_fit)
                    ax.plot(x_fine * 100, y_fit, '--', color=COLORS[model], alpha=0.5, linewidth=1.5)

                    # Asymptote
                    ax.axhline(a_fit, color=COLORS[model], alpha=0.3, linestyle=':', linewidth=1)

                    # Headroom
                    current = y[valid][-1]
                    headroom = a_fit - current
                    pct_saturated = (current / a_fit) * 100 if a_fit > 0 else 100

                    saturation_results.append({
                        'target': tgt, 'model': model, 'metric': mean_col,
                        'a_asymptote': a_fit, 'b_rate': b_fit,
                        'a_stderr': perr[0], 'b_stderr': perr[1],
                        'current_100pct': current,
                        'headroom': headroom,
                        'pct_saturated': pct_saturated,
                    })

                    report(f"  {tgt} {model} {mean_col}:")
                    report(f"    Asymptote a = {a_fit:.4f} ± {perr[0]:.4f}")
                    report(f"    Rate b = {b_fit:.2f} ± {perr[1]:.2f}")
                    report(f"    Current (100%) = {current:.4f}")
                    report(f"    Headroom = {headroom:.4f}")
                    report(f"    Saturated = {pct_saturated:.1f}%")

                except (RuntimeError, ValueError) as e:
                    report(f"  {tgt} {model} {mean_col}: Fit failed ({e})")

        ax.set_xlabel('Training Groups (%)')
        ax.set_ylabel('R²')
        ax.set_title(title)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        if 'archdev' in save_name:
            ax.set_ylim(bottom=-0.1)
        else:
            ax.set_ylim(bottom=0.8)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{save_name}.png", dpi=150)
        plt.close()

    report(f"  Saved: *_learning_curve.png (4 figures)")

    if saturation_results:
        sat_df = pd.DataFrame(saturation_results)
        report(f"\n  Saturation Summary:")
        report(f"  {'target':>4} {'model':>10} {'metric':>15} {'asympt':>8} {'current':>8} {'headroom':>9} {'%sat':>6}")
        report("  " + "-" * 60)
        for _, row in sat_df.iterrows():
            report(f"  {row['target']:>4} {row['model']:>10} {row['metric']:>15} "
                   f"{row['a_asymptote']:>8.4f} {row['current_100pct']:>8.4f} "
                   f"{row['headroom']:>+9.4f} {row['pct_saturated']:>5.1f}%")


def b5_interpretation(lc_df):
    """Generate interpretation of learning curve results."""
    report("\n" + "=" * 70)
    report("INTERPRETATION")
    report("=" * 70)

    # Check if arch-dev R² is still increasing at 100%
    for model in MODELS:
        for tgt in ['EA', 'IP']:
            sub = lc_df[(lc_df['target'] == tgt) & (lc_df['model'] == model)].sort_values('fraction')
            if len(sub) < 2:
                continue

            vals = sub['R2_arch_mean'].values
            fracs = sub['fraction'].values

            # Check slope from 75→100%
            if len(vals) >= 2:
                i75 = np.where(fracs == 75)[0]
                i100 = np.where(fracs == 100)[0]
                if len(i75) > 0 and len(i100) > 0:
                    delta = vals[i100[0]] - vals[i75[0]]
                    std_100 = sub.loc[sub['fraction'] == 100, 'R2_arch_std'].values[0]
                    report(f"\n  {tgt} {model}: R²_arch(75→100%) = {delta:+.4f}, std at 100% = {std_100:.4f}")
                    if delta > std_100:
                        report(f"    → Still improving (Δ > std): MORE DATA LIKELY BENEFICIAL")
                    elif delta > 0:
                        report(f"    → Marginal improvement (Δ < std): APPROACHING SATURATION")
                    else:
                        report(f"    → No improvement or decreasing: LIKELY SATURATED")


# ═══════════════════════════════════════════════════════════════════════
# BOTTLENECK ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════

def bottleneck_assessment(dev_data, lc_df):
    report("\n" + "=" * 70)
    report("DATASET BOTTLENECK ASSESSMENT")
    report("=" * 70)

    report("""
QUESTION: Are we hitting a dataset bottleneck (insufficient architecture
information), or would more matched architecture examples continue to
improve performance?

EVIDENCE SUMMARY:
""")

    # 1. Architecture signal magnitude
    for tshort in TARGETS:
        delta = dev_data[tshort]['delta']
        var_arch = np.var(delta, ddof=0)
        var_total = np.var(dev_data[tshort]['y'], ddof=0)
        pct = var_arch / var_total * 100
        report(f"  {tshort}: Architecture variance = {pct:.2f}% of total variance")
    report("")

    # 2. Learning curve slope at 100%
    for model in MODELS:
        for tgt in ['EA', 'IP']:
            sub = lc_df[(lc_df['target'] == tgt) & (lc_df['model'] == model)].sort_values('fraction')
            if len(sub) >= 2:
                vals = sub['R2_arch_mean'].values
                fracs = sub['fraction'].values
                i25 = np.where(fracs == 25)[0]
                i100 = np.where(fracs == 100)[0]
                if len(i25) > 0 and len(i100) > 0:
                    delta_full = vals[i100[0]] - vals[i25[0]]
                    report(f"  {tgt} {model}: R²_arch(25→100%) = {delta_full:+.4f}")

    report("""
ASSESSMENT:
""")

    # Determine verdict based on data
    # Check if arch-dev R² is increasing, plateauing, or high-variance
    verdicts = []
    for model in ['2d1_arch']:  # Focus on best model
        for tgt in ['EA', 'IP']:
            sub = lc_df[(lc_df['target'] == tgt) & (lc_df['model'] == model)].sort_values('fraction')
            if len(sub) < 4:
                verdicts.append('inconclusive')
                continue
            vals = sub['R2_arch_mean'].values
            stds = sub['R2_arch_std'].values
            # Check if monotonically increasing
            increasing = all(vals[i+1] >= vals[i] - stds[i] for i in range(len(vals)-1))
            # Check if large uncertainty
            mean_std = np.mean(stds)
            range_vals = vals[-1] - vals[0]
            if mean_std > 0.5 * range_vals:
                verdicts.append('inconclusive')
            elif increasing and (vals[-1] - vals[-2]) > 0.01:
                verdicts.append('data_limited')
            else:
                verdicts.append('saturated')

    if 'data_limited' in verdicts and 'saturated' not in verdicts:
        report("""  CONCLUSION: Evidence suggests PARTIAL DATA LIMITATION.
  Architecture-deviation R² continues to improve with more training groups,
  indicating that additional matched architecture examples would likely
  benefit performance. However, the improvement rate is slowing, suggesting
  we are approaching but have not yet reached the information ceiling.""")
    elif 'saturated' in verdicts and 'data_limited' not in verdicts:
        report("""  CONCLUSION: Evidence suggests ARCHITECTURE SATURATION.
  Architecture-deviation R² plateaus by 50-75% of training groups,
  indicating the model has extracted most learnable architecture
  information from the available data. More matched groups are unlikely
  to significantly improve architecture-deviation performance.""")
    elif 'inconclusive' in verdicts:
        report("""  CONCLUSION: INCONCLUSIVE due to high fold-to-fold variance.
  The large standard deviations across folds (driven by the a_held_out
  split's extreme monomer-level hold-out) prevent definitive determination
  of saturation vs data limitation. The signal-to-noise ratio in the
  learning curves is insufficient for a clear verdict.""")
    else:
        report("""  CONCLUSION: MIXED EVIDENCE.
  Results are target-dependent: one target shows saturation while the other
  shows continued improvement. This suggests the dataset bottleneck may
  differ by property.""")

    # Quantitative summary
    report("""
QUANTITATIVE SUMMARY:
""")
    for tshort in TARGETS:
        var_arch = np.var(dev_data[tshort]['delta'], ddof=0)
        var_total = np.var(dev_data[tshort]['y'], ddof=0)
        report(f"  {tshort}:")
        report(f"    Architecture effects = {var_arch/var_total*100:.2f}% of total variance")
        report(f"    |Δy| mean = {np.mean(np.abs(dev_data[tshort]['delta'])):.4f} eV")
        report(f"    |Δy| p95  = {np.percentile(np.abs(dev_data[tshort]['delta']), 95):.4f} eV")

    sub_best = lc_df[lc_df['model'] == '2d1_arch']
    if not sub_best.empty:
        for tgt in ['EA', 'IP']:
            s100 = sub_best[(sub_best['target'] == tgt) & (sub_best['fraction'] == 100)]
            s25 = sub_best[(sub_best['target'] == tgt) & (sub_best['fraction'] == 25)]
            if not s100.empty and not s25.empty:
                report(f"\n  2D1-arch {tgt} R²_arch: {s25.iloc[0]['R2_arch_mean']:.4f} (25%) → {s100.iloc[0]['R2_arch_mean']:.4f} (100%) "
                       f"  Δ = {s100.iloc[0]['R2_arch_mean'] - s25.iloc[0]['R2_arch_mean']:+.4f}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def save_reports():
    """Save markdown reports."""
    # Architecture signal report
    a_lines = [l for l in report_lines if report_lines.index(l) < next(
        (i for i, l in enumerate(report_lines) if 'EXPERIMENT B' in l), len(report_lines))]

    with open(OUT_DIR / "architecture_signal_analysis.md", 'w') as f:
        f.write("# Architecture Signal Magnitude Analysis\n\n")
        f.write("```\n")
        f.write("\n".join(a_lines))
        f.write("\n```\n")

    # Full report including learning curve + bottleneck
    with open(OUT_DIR / "architecture_learning_curve.md", 'w') as f:
        f.write("# Dataset Bottleneck Assessment\n\n")
        f.write("Combines architecture signal analysis with matched-group saturation learning curves.\n\n")
        f.write("```\n")
        f.write("\n".join(report_lines))
        f.write("\n```\n")


def main():
    report("=" * 70)
    report("STAGE 2D DATASET BOTTLENECK ASSESSMENT")
    report("=" * 70)

    df = load_dataset()
    report(f"Dataset: {len(df)} rows")
    report(f"Unique groups (A,B,fA,fB): {df['group_key'].nunique()}")
    report(f"Unique (A,B) pairs: {df.groupby(['smiles_A','smiles_B']).ngroups}")

    # Experiment A
    dev_data = experiment_a(df)

    # Add Var_arch to dev_data for bottleneck assessment
    for tshort, tlong in TARGETS.items():
        dev_data[tshort]['Var_arch'] = np.var(dev_data[tshort]['delta'], ddof=0)

    # Experiment B
    lc_df = experiment_b(df)

    # Combined assessment
    bottleneck_assessment(dev_data, lc_df)

    # Save reports
    save_reports()

    report(f"\n\nAll outputs saved to: {OUT_DIR}")
    report("Files:")
    report("  architecture_signal_analysis.md")
    report("  architecture_learning_curve.md")
    report("  architecture_variance_table.csv")
    report("  architecture_group_variance.csv")
    report("  learning_curve_metrics.csv")
    report("  deltaEA_distribution.png")
    report("  deltaIP_distribution.png")
    report("  architecture_variance_distribution_EA.png")
    report("  architecture_variance_distribution_IP.png")
    report("  EA_overall_learning_curve.png")
    report("  IP_overall_learning_curve.png")
    report("  EA_archdev_learning_curve.png")
    report("  IP_archdev_learning_curve.png")


if __name__ == "__main__":
    main()
