from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import wasserstein_distance
from sklearn.metrics import mean_absolute_error, r2_score

from analysis.diagnostics.data_loading import get_global_indices, load_all_meta, load_dataset, load_predictions_single
from analysis.diagnostics.grouping import add_group_means, build_fold_df, filter_matched_groups

OUT = ROOT / "analysis" / "model_diagnostics" / "report_figures" / "all"
MODELS = ["hpg_hier", "wdmpnn", "chemarch"]
LABELS = {"hpg_hier": "HPG-hier", "wdmpnn": "wDMPNN", "chemarch": "ChemArch"}
COLORS = {"hpg_hier": "#E8A33D", "wdmpnn": "#12314E", "chemarch": "#1C7293"}
TARGETS = ["EA", "IP"]
SPLITS = ["group_disjoint", "pair_disjoint", "monomer_heldout"]
SPLIT_LABELS = {"group_disjoint": "GD", "pair_disjoint": "PD", "monomer_heldout": "LOMO"}
N_FOLDS = {"group_disjoint": 5, "pair_disjoint": 5, "monomer_heldout": 9}
FOLD_NAMES = ["spiro-bifluorene", "dibenzothiophene sulfone", "difluorobenzene diboronic acid", "DTT fused trithiophene", "pyrene diboronic acid", "bithiophene diboronic acid", "benzothiadiazole diboronic acid", "benzene-1,4-diboronic acid", "carbazole"]
FIGURES: list[tuple[str, str]] = []


def save(fig: plt.Figure, filename: str, description: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    FIGURES.append((filename, description))


def metric_row(model: str, target: str, split: str, fold: int, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "model": model,
        "target": target,
        "split": split,
        "fold": fold,
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "target_sd": float(np.std(y_true)),
    }


def load_predictions(meta: dict[str, list], df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    data, rows = {}, []
    for model in MODELS:
        for target in TARGETS:
            for split in SPLITS:
                for fold in range(N_FOLDS[split]):
                    pred = load_predictions_single(model, target, split, fold, meta[split], seed=42)
                    if pred is None:
                        raise FileNotFoundError(f"Missing prediction: {model} {target} {split} fold {fold}")
                    fdf = add_group_means(build_fold_df(df, pred["y_true"], pred["y_pred"], pred["global_idx"]))
                    data[(model, target, split, fold)] = fdf
                    rows.append(metric_row(model, target, split, fold, pred["y_true"], pred["y_pred"]))
    return data, pd.DataFrame(rows)


def lomo_overall_plot(metrics: pd.DataFrame, metric: str, ylabel: str, filename: str, ylim: tuple[float, float] | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    width = 0.22
    for ax, target in zip(axes, TARGETS):
        sub = metrics[(metrics.target == target) & (metrics.split == "monomer_heldout")]
        for mi, model in enumerate(MODELS):
            values = sub[sub.model == model].sort_values("fold")[metric].to_numpy()
            shown = np.clip(values, *ylim) if ylim else values
            x = np.arange(9) + (mi - 1) * width
            ax.bar(x, shown, width, color=COLORS[model], label=LABELS[model])
            if ylim:
                for xi, val in zip(x, values):
                    if val < ylim[0] or val > ylim[1]:
                        ax.annotate(f"{val:.2f}", (xi, np.clip(val, *ylim)), xytext=(0, -14 if val < ylim[0] else 4), textcoords="offset points", ha="center", fontsize=7, rotation=90, color=COLORS[model])
        ax.axhline(0, color="#555555", lw=0.8)
        ax.set(title=f"{target}: LOMO per-fold {ylabel} | seed 42, single seed", xlabel="LOMO fold / held-out monomer")
        ax.set_xticks(range(9))
        ax.set_xticklabels([f"{i}\n{name}" for i, name in enumerate(FOLD_NAMES)], fontsize=7)
        ax.grid(axis="y", alpha=0.25)
        if ylim:
            ax.set_ylim(*ylim)
    axes[0].set_ylabel(ylabel)
    axes[1].legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    save(fig, filename, f"Per-fold LOMO {ylabel} by held-out monomer for all three models; bars are direct fold values and off-scale values are annotated.")


def fold1_parity(data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    group_rows, delta_rows = [], []
    for ax, model in zip(axes, ["hpg_hier", "wdmpnn"]):
        fdf = data[(model, "EA", "monomer_heldout", 1)]
        x, y = fdf.y_true.to_numpy(), fdf.y_pred.to_numpy()
        lo = min(x.min(), y.min()) - 0.08
        hi = max(x.max(), y.max()) + 0.08
        ax.scatter(x, y, s=5, alpha=0.18, color=COLORS[model], rasterized=True)
        ax.plot([lo, hi], [lo, hi], "--", color="#333333", lw=0.9)
        bias = float(np.mean(y - x))
        slope = float(np.polyfit(x, y, 1)[0])
        ax.text(0.04, 0.96, f"R²={r2_score(x, y):.3f}\nMAE={mean_absolute_error(x, y):.3f} eV\nbias={bias:+.3f} eV\nslope={slope:.3f}", transform=ax.transAxes, va="top", fontsize=8, bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
        ax.set(title=f"{LABELS[model]} | EA LOMO fold 1\ndibenzothiophene sulfone | seed 42", xlim=(lo, hi), ylim=(lo, hi), aspect="equal", xlabel="True EA (eV)")
    for model in MODELS:
        matched = filter_matched_groups(data[(model, "EA", "monomer_heldout", 1)])
        grouped = matched.groupby("group_key").agg(y_true=("y_true", "mean"), y_pred=("y_pred", "mean"))
        group_rows.append({"model": model, "r2": r2_score(grouped.y_true, grouped.y_pred), "mae": mean_absolute_error(grouped.y_true, grouped.y_pred), "rmse": np.sqrt(np.mean((grouped.y_true - grouped.y_pred) ** 2)), "n": len(grouped)})
        delta_rows.append({"model": model, "r2": r2_score(matched.delta_true, matched.delta_pred), "mae": mean_absolute_error(matched.delta_true, matched.delta_pred), "rmse": np.sqrt(np.mean((matched.delta_true - matched.delta_pred) ** 2)), "n": len(matched)})
    axes[0].set_ylabel("Predicted EA (eV)")
    fig.suptitle("EA fold-1 parity comparison | seed 42, single seed", fontweight="bold")
    fig.tight_layout()
    save(fig, "15_ea_lomo_fold1_parity_hpg_vs_wdmpnn.png", "EA LOMO fold-1 overall parity: HPG-hier versus wDMPNN, with R², MAE, bias, and slope.")
    return pd.DataFrame(group_rows), pd.DataFrame(delta_rows)


def decomposition_figure(group: pd.DataFrame, delta: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=False)
    x = np.arange(3)
    for ax, frame, title in zip(axes, [group, delta], ["Group-mean chemistry baseline", "Architecture deviation"]):
        for i, model in enumerate(MODELS):
            if model not in set(frame.model):
                continue
            row = frame[frame.model == model].iloc[0]
            ax.bar(i, row.mae, color=COLORS[model], label=LABELS[model])
            ax.text(i, row.mae, f"R²={row.r2:.3f}\nMAE={row.mae:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set(xticks=x, xticklabels=[LABELS[m] for m in MODELS], ylabel="MAE (eV)", title=title)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("EA LOMO fold 1 error decomposition | seed 42, single seed", fontweight="bold")
    fig.tight_layout()
    save(fig, "16_ea_lomo_fold1_group_vs_deviation.png", "EA fold-1 group-mean and architecture-deviation MAE with corresponding R² for all models.")


def target_shift(df: pd.DataFrame, meta: dict[str, list]) -> pd.DataFrame:
    test_idx = get_global_indices(1, meta["monomer_heldout"])
    train_idx = np.array(sorted(set(range(len(df))) - set(test_idx.tolist())), dtype=int)
    rows = []
    for target, col in [("EA", "EA vs SHE (eV)"), ("IP", "IP vs SHE (eV)")]:
        train = df.iloc[train_idx][col].dropna().to_numpy(float)
        test = df.iloc[test_idx][col].dropna().to_numpy(float)
        rows.append({"target": target, "train_mean": train.mean(), "heldout_mean": test.mean(), "mean_shift": test.mean() - train.mean(), "train_sd": train.std(), "heldout_sd": test.std(), "std_ratio": test.std() / train.std(), "wasserstein": wasserstein_distance(train, test), "train_n": len(train), "heldout_n": len(test)})
    return pd.DataFrame(rows)


def target_shift_figure(df: pd.DataFrame, meta: dict[str, list], shift: pd.DataFrame) -> None:
    test_idx = get_global_indices(1, meta["monomer_heldout"])
    train_idx = np.array(sorted(set(range(len(df))) - set(test_idx.tolist())), dtype=int)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, target, col in zip(axes, TARGETS, ["EA vs SHE (eV)", "IP vs SHE (eV)"]):
        train = df.iloc[train_idx][col].dropna().to_numpy(float)
        test = df.iloc[test_idx][col].dropna().to_numpy(float)
        bins = np.linspace(min(train.min(), test.min()), max(train.max(), test.max()), 45)
        ax.hist(train, bins=bins, density=True, alpha=0.55, color="#888888", label="train")
        ax.hist(test, bins=bins, density=True, alpha=0.55, color=COLORS["hpg_hier"], label="held-out fold 1")
        row = shift[shift.target == target].iloc[0]
        ax.text(0.04, 0.96, f"mean shift={row.mean_shift:+.3f} eV\nstd ratio={row.std_ratio:.3f}\nW₁={row.wasserstein:.3f} eV", transform=ax.transAxes, va="top", fontsize=8, bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
        ax.set(title=f"{target}: fold-1 target shift | seed 42", xlabel=f"{target} (eV)", ylabel="Density")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "17_lomo_fold1_target_shift_EA_IP.png", "Fold-1 target distributions: training versus held-out dibenzothiophene sulfone, with mean shift, standard-deviation ratio, and Wasserstein distance.")


def ensemble_metrics(data: dict) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        for split in SPLITS:
            for fold in range(N_FOLDS[split]):
                h, w = data[("hpg_hier", target, split, fold)], data[("wdmpnn", target, split, fold)]
                if not np.allclose(h.y_true, w.y_true):
                    raise ValueError(f"Mismatched truths for {target} {split} fold {fold}")
                rows.append(metric_row("ensemble_mean", target, split, fold, h.y_true.to_numpy(), (h.y_pred.to_numpy() + w.y_pred.to_numpy()) / 2))
    return pd.DataFrame(rows)


def wdmpnn_input_sanity(df: pd.DataFrame, meta: dict[str, list]) -> str:
    test_idx = get_global_indices(1, meta["monomer_heldout"])
    inputs = df.iloc[test_idx].WDMPNN_Input.dropna().unique()
    fragment_counts, port_counts, invalid = set(), set(), []
    for value in inputs:
        fragments = value.split("|")[0].split(".")
        fragment_counts.add(len(fragments))
        ports = []
        for fragment in fragments:
            mol = Chem.MolFromSmiles(fragment)
            if mol is None:
                invalid.append("RDKit parse failure")
                continue
            ports.extend(re.findall(r"\[\*:(\d+)\]", fragment))
        port_counts.add(len(ports))
        if len(fragments) != 2 or len(ports) != 4 or len(set(ports)) != 4:
            invalid.append(f"fragments={len(fragments)}, ports={ports}")
    return f"Fold-1 WDMPNN_Input sanity: {len(inputs)} unique inputs; fragment counts={sorted(fragment_counts)}; port counts={sorted(port_counts)}; RDKit fragment parse failures/port violations={len(invalid)}."


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in frame.iterrows():
        values = []
        for col in columns:
            val = row[col]
            values.append(f"{val:.{digits}f}" if isinstance(val, (float, np.floating)) else str(val))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def extend_readme(metrics: pd.DataFrame, fold1: pd.DataFrame, shift: pd.DataFrame, group: pd.DataFrame, delta: pd.DataFrame, ensemble: pd.DataFrame, sanity: str) -> None:
    path = OUT / "README.md"
    prior = path.read_text() if path.exists() else "# Seed-42 Diagnostics Figure Index\n\n"
    prior = prior.split("## Added: LOMO overall breakdown and EA fold-1 deep dive", 1)[0].rstrip() + "\n"
    fold1 = fold1.copy()
    if set(fold1.model).issubset(set(LABELS)):
        fold1["model"] = fold1.model.map(LABELS)
    ensemble_summary = ensemble.groupby(["target", "split"])[["r2", "mae"]].agg(["median", "mean"]).reset_index()
    ensemble_summary.columns = ["target", "split", "r2_median", "r2_mean", "mae_median", "mae_mean"]
    individual = metrics[metrics.model.isin(["hpg_hier", "wdmpnn"])].groupby(["model", "target", "split"])[["r2", "mae"]].agg(["median", "mean"]).reset_index()
    individual.columns = ["model", "target", "split", "r2_median", "r2_mean", "mae_median", "mae_mean"]
    individual["model"] = individual.model.map(LABELS)
    text = [prior.rstrip(), "", "## Added: LOMO overall breakdown and EA fold-1 deep dive", "", "### New figures", ""]
    text.extend([f"- `{name}` — {desc}" for name, desc in FIGURES])
    text.extend(["", "### EA/IP fold-1 (dibenzothiophene sulfone): overall test-set metrics", "", markdown_table(fold1, ["target", "model", "r2", "mae", "rmse", "target_sd"]), "", "Interpret MAE against `target_sd`: an R² drop can be amplified by a narrow held-out target distribution; MAE and SD are both eV.", "", "### Fold-1 target-distribution shift", "", markdown_table(shift, ["target", "train_mean", "heldout_mean", "mean_shift", "train_sd", "heldout_sd", "std_ratio", "wasserstein"]), "", "### EA fold-1 decomposition", "", "#### Group means", "", markdown_table(group, ["model", "r2", "mae", "rmse", "n"]), "", "#### Architecture deviations", "", markdown_table(delta, ["model", "r2", "mae", "rmse", "n"]), "", "### WDMPNN input sanity", "", f"- {sanity}", "", "### HPG-hier + wDMPNN arithmetic-mean ensemble", "", "#### Individual models", "", markdown_table(individual, ["target", "split", "model", "r2_median", "r2_mean", "mae_median", "mae_mean"]), "", "#### Ensemble", "", markdown_table(ensemble_summary, ["target", "split", "r2_median", "r2_mean", "mae_median", "mae_mean"]), ""])
    path.write_text("\n".join(text))


def main() -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    df, meta = load_dataset(), load_all_meta()
    data, metrics = load_predictions(meta, df)
    lomo_overall_plot(metrics, "r2", "Overall R²", "13_lomo_overall_r2.png", ylim=(-0.5, 1.05))
    lomo_overall_plot(metrics, "mae", "Overall MAE (eV)", "14_lomo_overall_mae.png")
    lomo_overall_plot(metrics, "rmse", "Overall RMSE (eV)", "14b_lomo_overall_rmse.png")
    group, delta = fold1_parity(data)
    all_fold1 = metrics[(metrics.split == "monomer_heldout") & (metrics.fold == 1)].copy()
    all_fold1["model"] = all_fold1.model.map(LABELS)
    decomposition_figure(group, delta)
    shift = target_shift(df, meta)
    target_shift_figure(df, meta, shift)
    ensemble = ensemble_metrics(data)
    sanity = wdmpnn_input_sanity(df, meta)
    extend_readme(metrics, all_fold1, shift, group, delta, ensemble, sanity)
    print(markdown_table(all_fold1, ["target", "model", "r2", "mae", "rmse", "target_sd"]))
    print(markdown_table(shift, ["target", "mean_shift", "std_ratio", "wasserstein"]))
    print(markdown_table(ensemble.groupby(["target", "split"])[["r2", "mae"]].agg(["median", "mean"]).reset_index().set_axis(["target", "split", "r2_median", "r2_mean", "mae_median", "mae_mean"], axis=1), ["target", "split", "r2_median", "r2_mean", "mae_median", "mae_mean"]))
    print(sanity)
    print(f"Wrote {len(FIGURES)} figures to {OUT}")


if __name__ == "__main__":
    main()
