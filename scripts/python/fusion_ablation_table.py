#!/usr/bin/env python3
"""
Fusion Ablation Manuscript Table Generator
============================================
Reads existing per-fold metrics and significance-test CSVs.
Computes actual median R² (not median Δ), fold-win counts, and Wilcoxon p-values.
Outputs CSV, Markdown, LaTeX, and notes files.

Input:
  output/fusion_ablation/fusion_per_fold_metrics.csv
  output/fusion_ablation/fusion_significance_tests.csv

Output:
  output/fusion_ablation/fusion_ablation_table_values.csv
  output/fusion_ablation/fusion_ablation_table_values.md
  output/fusion_ablation/fusion_ablation_table_latex.tex
  output/fusion_ablation/fusion_ablation_table_notes.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / 'output' / 'fusion_ablation'

df_fold = pd.read_csv(OUT_DIR / 'fusion_per_fold_metrics.csv')
df_stat = pd.read_csv(OUT_DIR / 'fusion_significance_tests.csv')

MODEL_ORDER = ['Additive (2D1)', 'FiLM', 'NLMix', 'FiLM+NLMix']
NON_BASE    = ['FiLM', 'NLMix', 'FiLM+NLMix']

# ── Map metric column names ────────────────────────────────────────────────────
# (target, fold_col, stat_label)
METRICS = [
    ('EA', 'R2',       'EA Overall R²',    'EA_Overall'),
    ('IP', 'R2',       'IP Overall R²',    'IP_Overall'),
    ('EA', 'ArchDev_R2', 'EA ArchDev R²',  'EA_ArchDev'),
    ('IP', 'ArchDev_R2', 'IP ArchDev R²',  'IP_ArchDev'),
]


# ── 1. Actual median R² per model per metric ──────────────────────────────────
def get_median_r2(model, target, col):
    sub = df_fold[(df_fold['Model'] == model) & (df_fold['Target'] == target)]
    vals = sub[col].dropna().values
    return float(np.median(vals)) if len(vals) > 0 else np.nan


# ── 2. Fold-win counts (variant vs Additive) ──────────────────────────────────
def fold_win_count(variant, target, col):
    """Count folds where variant > Additive on given metric (higher is better for R²)."""
    add_sub = df_fold[(df_fold['Model'] == 'Additive (2D1)') & (df_fold['Target'] == target)
                      ].set_index('Fold')[col]
    var_sub = df_fold[(df_fold['Model'] == variant) & (df_fold['Target'] == target)
                      ].set_index('Fold')[col]
    common  = add_sub.index.intersection(var_sub.index)
    wins    = int(((var_sub.loc[common]) > (add_sub.loc[common])).sum())
    return wins, len(common)


# ── 3. Wilcoxon p-values from significance-tests CSV ─────────────────────────
def get_wilcoxon_p(variant, stat_label):
    comp = f"{variant} vs Additive"
    row  = df_stat[(df_stat['Metric'] == stat_label) & (df_stat['Comparison'] == comp)]
    if row.empty:
        return np.nan
    return float(row['Wilcoxon_pvalue'].values[0])


# ── Build master rows ─────────────────────────────────────────────────────────
records = []
for model in MODEL_ORDER:
    row = {'Fusion': model}
    for target, col, stat_label, prefix in METRICS:
        med = get_median_r2(model, target, col)
        row[f'{prefix}_median_R2'] = round(med, 6)

        if model == 'Additive (2D1)':
            row[f'{prefix}_wins_vs_Additive'] = '--'
            row[f'{prefix}_p_vs_Additive']    = '--'
        else:
            wins, n = fold_win_count(model, target, col)
            p        = get_wilcoxon_p(model, stat_label)
            row[f'{prefix}_wins_vs_Additive'] = f"{wins}/{n}"
            row[f'{prefix}_p_vs_Additive']    = round(p, 6) if not np.isnan(p) else 'N/A'
    records.append(row)

df_out = pd.DataFrame(records)

# ── Output 1: CSV ─────────────────────────────────────────────────────────────
col_order = ['Fusion']
for _, _, _, prefix in METRICS:
    col_order += [f'{prefix}_median_R2', f'{prefix}_wins_vs_Additive', f'{prefix}_p_vs_Additive']
df_out[col_order].to_csv(OUT_DIR / 'fusion_ablation_table_values.csv', index=False)
print("Saved: fusion_ablation_table_values.csv")


# ── Output 2: Markdown ────────────────────────────────────────────────────────
# Find best median R² per metric column (for bolding)
best_med = {}
for _, _, _, prefix in METRICS:
    vals = {r['Fusion']: r[f'{prefix}_median_R2'] for r in records}
    best_med[prefix] = max(vals, key=lambda k: vals[k])

def fmt_med(val, prefix, model):
    s = f"{val:.3f}"
    if model == best_med[prefix]:
        s = f"**{s}**"
    return s

def fmt_p(val):
    if val == '--':
        return '--'
    if isinstance(val, str):
        return val
    s = f"{val:.4f}"
    if float(val) < 0.05:
        s += '\\*'
    return s

md_lines = [
    "# Fusion Ablation — Manuscript Table Values\n\n",
    "> Median R² is the actual median across 9 LOMO folds (not median Δ).\n",
    "> Wins = number of folds where the variant outperformed Additive (2D1).\n",
    "> p = paired Wilcoxon signed-rank test vs Additive. \\* = p < 0.05.\n",
    "> **Bold** = best median R² in that column.\n\n",
    "| Model | EA R² med | EA wins | EA p | IP R² med | IP wins | IP p | "
    "EA ArchDev R² med | EA ArchDev wins | EA ArchDev p | "
    "IP ArchDev R² med | IP ArchDev wins | IP ArchDev p |\n",
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n",
]

for r in records:
    model = r['Fusion']
    cells = [model]
    for _, _, _, prefix in METRICS:
        cells.append(fmt_med(r[f'{prefix}_median_R2'], prefix, model))
        cells.append(r[f'{prefix}_wins_vs_Additive'])
        cells.append(fmt_p(r[f'{prefix}_p_vs_Additive']))
    md_lines.append("| " + " | ".join(str(c) for c in cells) + " |\n")

with open(OUT_DIR / 'fusion_ablation_table_values.md', 'w') as f:
    f.writelines(md_lines)
print("Saved: fusion_ablation_table_values.md")


# ── Output 3: LaTeX ───────────────────────────────────────────────────────────
def latex_med(val, prefix, model):
    s = f"{val:.3f}"
    if model == best_med[prefix]:
        s = f"\\textbf{{{s}}}"
    return s

def latex_p(val):
    if val == '--':
        return '--'
    if isinstance(val, str):
        return val
    s = f"{float(val):.4f}"
    if float(val) < 0.05:
        s += '$^*$'
    return s

latex_lines = [
    "% Fusion Ablation Manuscript Table\n",
    "% Generated from fusion_per_fold_metrics.csv and fusion_significance_tests.csv\n",
    "% Median R² = actual fold median. * = Wilcoxon p < 0.05 vs Additive (2D1).\n",
    "% Bold = best in column.\n\n",
    "\\begin{table*}[t]\n",
    "\\centering\n",
    "\\caption{Summary of fusion ablation under leave-one-monomer-out evaluation. "
    "Median $R^2$ is reported across the nine folds. "
    "``Wins'' denotes the number of folds in which each variant outperformed "
    "the additive formulation. Statistical significance was assessed using a "
    "paired Wilcoxon signed-rank test against Additive (Stage~2D1).}\n",
    "\\label{tab:fusion_ablation}\n",
    "\\resizebox{\\textwidth}{!}{%\n",
    "\\begin{tabular}{l"
    " c c c"   # EA overall
    " c c c"   # IP overall
    " c c c"   # EA ArchDev
    " c c c"   # IP ArchDev
    "}\n",
    "\\toprule\n",
    # Two-row header
    " & \\multicolumn{3}{c}{\\textbf{Overall EA $R^2$}}"
    " & \\multicolumn{3}{c}{\\textbf{Overall IP $R^2$}}"
    " & \\multicolumn{3}{c}{\\textbf{EA Arch-Dev $R^2$}}"
    " & \\multicolumn{3}{c}{\\textbf{IP Arch-Dev $R^2$}} \\\\\n",
    "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\\cmidrule(lr){11-13}\n",
    "\\textbf{Model}"
    " & Median & Wins & $p$"
    " & Median & Wins & $p$"
    " & Median & Wins & $p$"
    " & Median & Wins & $p$ \\\\\n",
    "\\midrule\n",
]

for r in records:
    model = r['Fusion']
    display = model.replace('Additive (2D1)', 'Additive (Stage 2D1)').replace('+', '$+$')
    cells = [display]
    for _, _, _, prefix in METRICS:
        cells.append(latex_med(r[f'{prefix}_median_R2'], prefix, model))
        cells.append(r[f'{prefix}_wins_vs_Additive'])
        cells.append(latex_p(r[f'{prefix}_p_vs_Additive']))
    latex_lines.append(" & ".join(str(c) for c in cells) + " \\\\\n")

latex_lines += [
    "\\bottomrule\n",
    "\\end{tabular}%\n",
    "}\n",
    "\\end{table*}\n",
]

with open(OUT_DIR / 'fusion_ablation_table_latex.tex', 'w') as f:
    f.writelines(latex_lines)
print("Saved: fusion_ablation_table_latex.tex")


# ── Output 4: Notes ───────────────────────────────────────────────────────────
# Check if excluding fold 6 changes any direction
df_no6 = df_fold[df_fold['Fold'] != 6]

direction_changes = []
for target, col, stat_label, prefix in METRICS:
    for variant in NON_BASE:
        add_sub = df_fold[(df_fold['Model'] == 'Additive (2D1)') & (df_fold['Target'] == target)
                          ].set_index('Fold')[col]
        var_sub = df_fold[(df_fold['Model'] == variant) & (df_fold['Target'] == target)
                          ].set_index('Fold')[col]
        common_all = add_sub.index.intersection(var_sub.index)
        diff_all   = (var_sub.loc[common_all] - add_sub.loc[common_all]).values
        sign_all   = np.sign(diff_all.mean())

        add_no6 = add_sub[add_sub.index != 6]
        var_no6 = var_sub[var_sub.index != 6]
        common_no6 = add_no6.index.intersection(var_no6.index)
        diff_no6   = (var_no6.loc[common_no6] - add_no6.loc[common_no6]).values
        sign_no6   = np.sign(diff_no6.mean())

        if sign_all != sign_no6:
            direction_changes.append(f"  - {stat_label} / {variant}: "
                                     f"direction reverses (all: Δ={diff_all.mean():+.4f}, "
                                     f"excl. fold 6: Δ={diff_no6.mean():+.4f})")

# Verify median R² are not median Δ (sanity check)
add_ea_med = get_median_r2('Additive (2D1)', 'EA', 'R2')
film_ea_med = get_median_r2('FiLM', 'EA', 'R2')
film_ea_stat_median_delta = df_stat[
    (df_stat['Metric'] == 'EA Overall R²') &
    (df_stat['Comparison'] == 'FiLM vs Additive')]['Median_delta'].values[0]

notes_lines = [
    "# Fusion Ablation — Table Verification Notes\n\n",

    "## 1. Median R² values are actual fold medians\n\n",
    "Values in `fusion_ablation_table_values.csv` are computed as "
    "`np.median(per_fold_R2_values)` from `fusion_per_fold_metrics.csv`.\n",
    "They are **not** median differences (Δ) from `fusion_significance_tests.csv`.\n\n",
    "Cross-check:\n",
    f"  - Additive (2D1) EA Overall R²: median = {add_ea_med:.4f}  "
    f"(median of 9 fold R² values)\n",
    f"  - FiLM EA Overall R²:           median = {film_ea_med:.4f}  "
    f"(median of 9 fold R² values)\n",
    f"  - FiLM vs Additive EA Overall median Δ from significance tests: "
    f"{film_ea_stat_median_delta:+.4f}  (this is a different quantity)\n",
    f"  - Difference: {film_ea_med:.4f} − {add_ea_med:.4f} = "
    f"{film_ea_med - add_ea_med:+.4f}  ≠ median Δ above (due to nonlinearity of median)\n\n",

    "## 2. p-values are paired Wilcoxon signed-rank tests\n\n",
    "Taken directly from `fusion_significance_tests.csv` column `Wilcoxon_pvalue`.\n",
    "Nine folds were treated as paired observations.\n",
    "Comparison in each row: variant fold_i − Additive fold_i.\n\n",

    "## 3. Robustness: effect of excluding fold 6\n\n",
    f"Fold 6 held-out monomer (OB(O)c1ccc(B(O)O)c2nsnc12) causes catastrophic failure "
    f"in ALL models (EA R² ≈ −12 to −17).\n\n",
]

if direction_changes:
    notes_lines += [
        "The following comparisons **change direction** when fold 6 is excluded:\n\n",
    ]
    for ch in direction_changes:
        notes_lines.append(ch + "\n")
    notes_lines.append("\n")
else:
    notes_lines.append(
        "**No comparison changes direction after excluding fold 6.**\n"
        "The sign of the mean difference is identical for all 12 model–metric pairs "
        "with and without fold 6. The conclusions in the table are robust to this outlier.\n\n"
    )

notes_lines += [
    "## 4. Win counts\n\n",
    "Win counts = number of the 9 folds where variant R² > Additive R² "
    "(strict inequality). Computed directly from `fusion_per_fold_metrics.csv`.\n\n",

    "## 5. Best-in-column bolding\n\n",
    "Bold values in the Markdown and LaTeX tables indicate the model with "
    "the highest median R² in each column.\n\n",
]

for _, _, _, prefix in METRICS:
    notes_lines.append(f"  - {prefix.replace('_', ' ')}: best = {best_med[prefix]}\n")

with open(OUT_DIR / 'fusion_ablation_table_notes.md', 'w') as f:
    f.writelines(notes_lines)
print("Saved: fusion_ablation_table_notes.md")


# ── Console summary ───────────────────────────────────────────────────────────
print()
print("=" * 70)
print("TABLE VALUES (median R² | wins/9 | Wilcoxon p)")
print("=" * 70)
header = f"{'Model':<22} {'EA med':>7} {'EA wins':>8} {'EA p':>8}  "
header += f"{'IP med':>7} {'IP wins':>8} {'IP p':>8}  "
header += f"{'EA AD med':>9} {'EA AD wins':>10} {'EA AD p':>8}  "
header += f"{'IP AD med':>9} {'IP AD wins':>10} {'IP AD p':>8}"
print(header)
print("-" * len(header))
for r in records:
    model = r['Fusion']
    def fv(prefix):
        med  = r[f'{prefix}_median_R2']
        wins = r[f'{prefix}_wins_vs_Additive']
        p    = r[f'{prefix}_p_vs_Additive']
        p_s  = f"{float(p):.4f}*" if p != '--' and not isinstance(p, str) and float(p) < 0.05 \
               else (f"{float(p):.4f}" if p != '--' and not isinstance(p, str) else '--')
        return f"{med:>7.3f} {str(wins):>8} {p_s:>9}"
    line = f"{model:<22} " + "  ".join(fv(pfx) for _, _, _, pfx in METRICS)
    print(line)
