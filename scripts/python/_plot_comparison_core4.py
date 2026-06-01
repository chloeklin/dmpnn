import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

out_dir = Path("pre_2d_diagnostics/feature_conditioned_transfer")
summary_df = pd.read_csv(out_dir / "transfer_metrics.csv")

KEEP = ["arch_only", "arch_frac", "arch_chem", "arch_chem_frac"]
COLORS = {
    "arch_only":      "#90caf9",
    "arch_frac":      "#42a5f5",
    "arch_chem":      "#ffcc80",
    "arch_chem_frac": "#a5d6a7",
}

sub_all = summary_df[summary_df["feature_set"].isin(KEEP)]

for target in sorted(sub_all["target"].unique()):
    sub = sub_all[sub_all["target"] == target]

    best_rows = []
    for fs in KEEP:
        fs_sub = sub[sub["feature_set"] == fs]
        if fs_sub.empty:
            continue
        best_row = fs_sub.loc[fs_sub["mean_R2"].idxmax()].copy()
        best_rows.append(best_row)

    best_df = pd.DataFrame(best_rows).set_index("feature_set")
    best_df = best_df.reindex(
        best_df["mean_R2"].sort_values(ascending=True).index
    )

    fig, ax = plt.subplots(figsize=(7, 3.2))
    colors = [COLORS.get(fs, "#999999") for fs in best_df.index]
    bars = ax.barh(
        range(len(best_df)),
        best_df["mean_R2"].values,
        xerr=best_df["std_R2"].values,
        color=colors, alpha=0.88,
        error_kw=dict(elinewidth=1.2, capsize=3, ecolor="#333333"),
    )
    yticklabels = [
        "{:}  [{}]".format(fs, best_df.loc[fs, "model"])
        for fs in best_df.index
    ]
    ax.set_yticks(range(len(best_df)))
    ax.set_yticklabels(yticklabels, fontsize=9)
    ax.set_xlabel("Mean R²  (best model, ±std)", fontsize=10)
    ax.set_title(
        "Delta{} — Feature set comparison (core 4)".format(target),
        fontweight="bold", fontsize=11
    )
    ax.axvline(0, color="k", linewidth=0.6)
    for bar, val in zip(bars, best_df["mean_R2"].values):
        ax.text(
            bar.get_width() + 0.008,
            bar.get_y() + bar.get_height() / 2,
            "{:.4f}".format(val),
            va="center", fontsize=8.5,
        )
    ax.set_xlim(left=min(0, best_df["mean_R2"].min() - 0.05))
    fig.tight_layout()
    out_png = out_dir / "comparison_barplot_{}_core4.png".format(target)
    out_pdf = out_dir / "comparison_barplot_{}_core4.pdf".format(target)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print("Saved {}".format(out_png))

print("Done.")
