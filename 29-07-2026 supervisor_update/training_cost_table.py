#!/usr/bin/env python3
"""Training cost and convergence speed per model, from the run provenance sidecars.

Every number comes from the .config.json written beside each prediction .npz, so this
can be re-run whenever new results land and the table stays honest.

    Source     predictions/regen_v1/<split>/ea_ip__<target>__<model>__<split>__fold<f>__s<seed>.config.json
    Fields     wall_time_seconds   measured wall clock for the run
               best_epoch          epoch of the best validation checkpoint
               epochs_actually_run best_epoch + patience (the early-stopping tail)

Two things worth knowing before reading the output:

  * epochs_actually_run is NOT epochs-to-converge. Patience is a constant 15 for every
    model, so the tail is included in every run and dilutes the between-model ratio.
    best_epoch is the convergence number. Both are printed; quote best_epoch.

  * With cells missing, a plain median per model is over a different set of runs for
    each model, so the columns are not strictly comparable. The PAIRED section restricts
    every model to the (target, fold, seed) cells that ALL models completed, which is the
    defensible comparison. Use it for anything that goes in the paper.

Charging (NCI Gadi, gpuvolta):
    SU = queue_charge_rate x max(ncpus, mem_proportion) x walltime_hours
       = 3 x 12 x hours = 36 SU per GPU-hour
    Source: https://opus.nci.org.au/spaces/Help/pages/90308792  (formula)
            https://opus.nci.org.au/spaces/Help/pages/90308823  (gpuvolta = 3 SU, ncpus multiple of 12)
    The rate is asserted against each sidecar's recorded ncpus where available.

Usage:
    python "29-07-2026 supervisor_update/training_cost_table.py"
    python "29-07-2026 supervisor_update/training_cost_table.py" --by-stage
    python "29-07-2026 supervisor_update/training_cost_table.py" --csv costs.csv
    python "29-07-2026 supervisor_update/training_cost_table.py" --su-per-hour 36
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRED = ROOT / "predictions" / "regen_v1"

# Gadi gpuvolta: 3 SU per resource-hour, jobs request ncpus=12 -> 36 SU per GPU-hour.
SU_PER_GPU_HOUR = 36.0
QUEUE_CHARGE_RATE = 3.0

# What a complete campaign looks like. Kept in step with check_results.py.
STAGES = {
    "R1  A-heldout": {
        "dir": "ea_ip_lomo",
        "models": ["hpg_hier", "wdmpnn", "hpg_hier_octamer",
                   "hpg_hier_junction", "hpg_hier_junction1"],
    },
    "R3  B-heldout clustered": {
        "dir": "ea_ip_lomo_b_clustered",
        "models": ["hpg_hier", "wdmpnn", "hpg_hier_octamer", "hpg_hier_junction"],
    },
}
TARGETS = ["EA_vs_SHE_eV", "IP_vs_SHE_eV"]
FOLDS = range(9)
SEEDS = [42, 43, 44]

MODEL_ORDER = ["wdmpnn", "hpg_hier", "hpg_hier_octamer",
               "hpg_hier_junction", "hpg_hier_junction1"]
LABELS = {
    "wdmpnn": "wDMPNN",
    "hpg_hier": "HPG-hier",
    "hpg_hier_octamer": "HPG-hier octamer",
    "hpg_hier_junction": "HPG-hier junction",
    "hpg_hier_junction1": "HPG-hier junction-1",
}


# --------------------------------------------------------------------------- #
class Run:
    """One completed training run, as recorded by its provenance sidecar."""

    __slots__ = ("stage", "model", "target", "fold", "seed", "wall_s", "best_epoch",
                 "total_epochs", "patience", "ncpus", "cli", "epoch_cap", "batch_size")

    def __init__(self, stage, model, target, fold, seed, meta):
        self.stage, self.model = stage, model
        self.target, self.fold, self.seed = target, fold, seed
        self.wall_s = meta.get("wall_time_seconds")
        self.best_epoch = meta.get("best_epoch")
        self.total_epochs = meta.get("epochs_actually_run")
        # resolved_config holds what the run ACTUALLY used; cli_args only holds what was
        # passed on the command line, so module-level constants (wDMPNN's BATCH_SIZE=512,
        # EPOCHS=300) appear as null there. Prefer resolved, fall back to cli.
        resolved = meta.get("resolved_config", {}) or {}
        cli = meta.get("cli_args", {}) or {}
        self.cli = {k: (resolved.get(k) if resolved.get(k) is not None else cli.get(k))
                    for k in set(resolved) | set(cli)}
        self.patience = self.cli.get("patience")
        self.epoch_cap = self.cli.get("epochs")
        self.batch_size = self.cli.get("batch_size")
        self.ncpus = meta.get("ncpus") or self.cli.get("ncpus")

    @property
    def hours(self):
        return self.wall_s / 3600.0

    @property
    def s_per_epoch(self):
        return self.wall_s / self.total_epochs if self.total_epochs else None

    @property
    def rel_steps(self):
        """Gradient updates to the best checkpoint, in units of the training-set size.

        steps = epochs x ceil(n_train / batch_size), and n_train is common to every model
        within a cell, so epochs / batch_size is proportional to the update count and can
        be compared across models even though the sidecars do not record n_train.

        This is the column to read instead of 'best ep' whenever batch sizes differ.
        """
        if self.best_epoch is None or not self.batch_size:
            return None
        return self.best_epoch / self.batch_size

    @property
    def cell(self):
        """Identity of the experiment, independent of which model ran it."""
        return (self.stage, self.target, self.fold, self.seed)


def parse_name(name: str):
    """ea_ip__<target>__<model>__<split>__fold<f>__s<seed> -> (target, model, fold, seed).

    Split on the '__' delimiter rather than substring-matching model names: 'hpg_hier'
    is a prefix of hpg_hier_octamer, hpg_hier_junction and hpg_hier_junction1, so a
    substring test silently misfiles three models out of five.
    """
    parts = name.replace(".config.json", "").split("__")
    if len(parts) != 6:
        return None
    _, target, model, _split, fold, seed = parts
    if not fold.startswith("fold") or not seed.startswith("s"):
        return None
    try:
        return target, model, int(fold[4:]), int(seed[1:])
    except ValueError:
        return None


def collect() -> tuple[list[Run], list[str]]:
    runs, warnings = [], []
    for stage, cfg in STAGES.items():
        stage_dir = PRED / cfg["dir"]
        if not stage_dir.is_dir():
            warnings.append(f"{stage}: directory absent ({stage_dir})")
            continue
        for path in sorted(stage_dir.glob("*.config.json")):
            parsed = parse_name(path.name)
            if parsed is None:
                warnings.append(f"unparseable sidecar name: {path.name}")
                continue
            target, model, fold, seed = parsed
            try:
                meta = json.loads(path.read_text())
            except json.JSONDecodeError:
                warnings.append(f"unreadable sidecar: {path.name}")
                continue
            run = Run(stage, model, target, fold, seed, meta)
            if not run.wall_s or not run.total_epochs:
                warnings.append(f"sidecar lacks timing: {path.name}")
                continue
            runs.append(run)
    return runs, warnings


def expected_cells() -> dict[str, set]:
    """Every (stage, model, target, fold, seed) the frozen campaign should contain."""
    out = defaultdict(set)
    for stage, cfg in STAGES.items():
        for model in cfg["models"]:
            for target in TARGETS:
                for fold in FOLDS:
                    for seed in SEEDS:
                        out[model].add((stage, target, fold, seed))
    return out


# --------------------------------------------------------------------------- #
# Hyperparameters that must match for a between-model comparison to be about the
# models. Anything here that differs is printed loudly, because a difference makes
# the epoch columns non-comparable and puts the accuracy claims in question too.
PARITY_FIELDS = ["batch_size", "epochs", "patience", "init_lr", "num_workers"]


def parity(runs: list[Run]) -> None:
    """Report whether the models were actually trained under the same protocol.

    This exists because they were not. run_wdmpnn_generalization.py uses BATCH_SIZE=512
    and EPOCHS=300; run_hpg_generalization.py defaults to 64 and 100. Only `patience` is
    guarded (run_wdmpnn_generalization.py:435). With an 8x batch difference an 'epoch' is
    a different amount of optimisation for different models, so epochs-to-best cannot be
    compared across the batch_size boundary — see the relative-steps column.
    """
    seen = defaultdict(lambda: defaultdict(set))
    for r in runs:
        for field, value in (r.cli or {}).items():
            if field in PARITY_FIELDS:
                seen[field][r.model].add(value)

    differing = {f: v for f, v in seen.items() if len({tuple(sorted(map(str, s)))
                                                       for s in v.values()}) > 1}
    print("\nPROTOCOL PARITY")
    if not differing:
        print("  all of " + ", ".join(PARITY_FIELDS) + " match across models.")
        return
    print("  ! models were NOT trained under a matched protocol. Fields that differ:")
    for field, per_model in sorted(differing.items()):
        print(f"    {field}:")
        for model in [m for m in MODEL_ORDER if m in per_model]:
            vals = ", ".join(str(v) for v in sorted(per_model[model], key=str))
            print(f"      {LABELS.get(model, model):<21} {vals}")
    if "batch_size" in differing:
        print("    -> 'best ep' and 'tot ep' are NOT comparable across models with different\n"
              "       batch sizes: one epoch is a different number of gradient updates.\n"
              "       Use the 'rel steps' column, and treat s/epoch (one pass over the same\n"
              "       training set) as the only directly comparable timing number.")
    if "init_lr" in differing or "batch_size" in differing:
        print("    -> this also confounds the ACCURACY comparison: a model trained at a\n"
              "       different batch size / learning rate may be under-tuned rather than\n"
              "       architecturally worse. State it, or run a matched arm.")
    # The learning rate is the one field that would settle the under-tuning question,
    # and no runner records it. Say so rather than let its absence read as agreement.
    if not any(r.cli.get(f) is not None for r in runs for f in ("init_lr", "max_lr")):
        print("    ! no run records init_lr or max_lr, so LR parity cannot be checked from\n"
              "      the sidecars at all. run_hpg_generalization uses flat Adam 1e-3;\n"
              "      run_wdmpnn_generalization uses the chemprop scheduler. These differ.")


def capped(runs: list[Run]) -> None:
    """Runs stopped by the epoch budget rather than by early stopping.

    A run whose best epoch is close to its cap was still improving when it was cut off,
    so its predictions come from an undertrained model and should not be pooled unflagged.
    """
    hits = [r for r in runs if r.epoch_cap and r.total_epochs
            and r.total_epochs >= r.epoch_cap]
    if not hits:
        return
    truncated = [r for r in hits if r.best_epoch is not None
                 and r.epoch_cap - r.best_epoch <= (r.patience or 15)]
    print(f"\nEPOCH CAP\n  {len(hits)} run(s) stopped at the cap rather than by early stopping.")
    if truncated:
        print(f"  {len(truncated)} of those had their best epoch within one patience window "
              f"of the cap — still improving when cut off, so undertrained:")
        for r in sorted(truncated, key=lambda x: (x.stage, x.model, x.target, x.fold)):
            print(f"      {LABELS.get(r.model, r.model):<21}{r.stage:<24}{r.target}  "
                  f"fold{r.fold}  s{r.seed}   best {r.best_epoch} of {r.epoch_cap}")
    else:
        print("  none had its best epoch near the cap, so none looks undertrained.")


def med(values):
    vals = [v for v in values if v is not None]
    return st.median(vals) if vals else float("nan")


def table(runs: list[Run], su_per_hour: float, title: str, expected=None) -> list[dict]:
    by_model = defaultdict(list)
    for r in runs:
        by_model[r.model].append(r)
    models = [m for m in MODEL_ORDER if m in by_model] + \
             sorted(m for m in by_model if m not in MODEL_ORDER)
    if not models:
        print(f"\n{title}\n  no runs")
        return []

    print(f"\n{title}")
    head = (f"  {'model':<21}{'n':>5}{'cover':>8}{'bs':>5}{'best ep':>9}{'tot ep':>8}"
            f"{'rel steps':>11}{'s/epoch':>9}{'wall h':>8}{'SU/run':>8}{'SU total':>10}")
    print(head)
    print("  " + "-" * (len(head) - 2))

    # rel_steps is only meaningful relative to something; normalise to HPG-hier.
    ref = med(r.rel_steps for r in by_model.get("hpg_hier", []))

    rows, grand = [], 0.0
    for model in models:
        rs = by_model[model]
        su_total = sum(r.hours for r in rs) * su_per_hour
        grand += su_total
        cover = ""
        if expected is not None:
            want = len(expected.get(model, ()))
            cover = f"{len(rs)}/{want}" if want else "-"
        bss = {r.batch_size for r in rs if r.batch_size}
        steps = med(r.rel_steps for r in rs)
        row = {
            "model": model,
            "n": len(rs),
            "coverage": cover,
            "batch_size": "/".join(str(b) for b in sorted(bss)) if bss else "?",
            "median_best_epoch": med(r.best_epoch for r in rs),
            "median_total_epochs": med(r.total_epochs for r in rs),
            "rel_grad_steps_vs_hpg_hier": (steps / ref) if ref and steps == steps else float("nan"),
            "median_s_per_epoch": med(r.s_per_epoch for r in rs),
            "median_wall_hours": med(r.hours for r in rs),
            "su_per_run": med(r.hours for r in rs) * su_per_hour,
            "su_total": su_total,
        }
        rows.append(row)
        rel = row["rel_grad_steps_vs_hpg_hier"]
        rel_s = f"{rel:.2f}x" if rel == rel else "?"
        print(f"  {LABELS.get(model, model):<21}{row['n']:>5}{cover:>8}{row['batch_size']:>5}"
              f"{row['median_best_epoch']:>9.0f}{row['median_total_epochs']:>8.0f}"
              f"{rel_s:>11}{row['median_s_per_epoch']:>9.0f}{row['median_wall_hours']:>8.2f}"
              f"{row['su_per_run']:>8.1f}{su_total:>10.0f}")
    print("  " + "-" * (len(head) - 2))
    print(f"  {'TOTAL':<21}{sum(r['n'] for r in rows):>5}"
          f"{'':>8}{'':>5}{'':>9}{'':>8}{'':>11}{'':>9}{'':>8}{'':>8}{grand:>10.0f}")
    print(f"  {'':<21}{'':>5}{'':>8}{'':>5}{'':>9}{'':>8}{'':>11}{'':>9}{'':>8}{'':>8}"
          f"{grand / 1000:>9.2f} kSU")

    # State the comparison against the hierarchy baseline, in the unit that is fair.
    base = next((r for r in rows if r["model"] == "hpg_hier"), None)
    wd = next((r for r in rows if r["model"] == "wdmpnn"), None)
    if base and wd:
        same_bs = base["batch_size"] == wd["batch_size"]
        if same_bs and base["median_best_epoch"]:
            print(f"\n  wDMPNN reaches its best checkpoint in {wd['median_best_epoch']:.0f} "
                  f"epochs vs HPG-hier's {base['median_best_epoch']:.0f} "
                  f"({wd['median_best_epoch'] / base['median_best_epoch']:.1f}x), at "
                  f"{wd['median_s_per_epoch']:.0f} vs {base['median_s_per_epoch']:.0f} s/epoch.")
        else:
            rel = wd["rel_grad_steps_vs_hpg_hier"]
            print(f"\n  Batch sizes differ ({wd['batch_size']} vs {base['batch_size']}), so do NOT "
                  f"compare epochs.\n  In gradient updates wDMPNN takes {rel:.2f}x what HPG-hier "
                  f"does" + (" — i.e. FEWER, the opposite of what the epoch column suggests."
                             if rel == rel and rel < 1 else ".") +
                  f"\n  The fair timing number is s/epoch (one pass over the same data): "
                  f"{wd['median_s_per_epoch']:.0f} vs {base['median_s_per_epoch']:.0f}.")
    return rows


def paired(runs: list[Run], su_per_hour: float) -> list[dict]:
    """Restrict to cells every model completed, then compare — WITHIN each stage.

    With runs missing, an unpaired median per model summarises a different set of folds
    for each model. Pairing fixes that. But it has to be done per stage: junction-1 was
    only ever run in R1, so a global intersection across all five models would silently
    discard every R3 cell and quietly turn this into an R1-only table.
    """
    out = []
    for stage, cfg in STAGES.items():
        sub = [r for r in runs if r.stage == stage]
        if not sub:
            continue
        by_model = defaultdict(dict)
        for r in sub:
            by_model[r.model][r.cell] = r
        # Only models the stage was meant to include — a model absent by design must
        # not be allowed to empty the intersection.
        by_model = {m: v for m, v in by_model.items() if m in cfg["models"]}
        if len(by_model) < 2:
            continue
        common = set.intersection(*(set(v) for v in by_model.values()))
        if not common:
            print(f"\nPAIRED  {stage}\n  no cell was completed by every model — cannot pair")
            continue
        dropped = {m: len(v) - len(common) for m, v in by_model.items()}
        kept = [r for cells in by_model.values() for c, r in cells.items() if c in common]
        out += table(kept, su_per_hour,
                     f"PAIRED  {stage} — the {len(common)} of "
                     f"{len(TARGETS) * len(list(FOLDS)) * len(SEEDS)} cells completed by "
                     f"all {len(by_model)} models  (this is the paper number)")
        if any(dropped.values()):
            detail = ", ".join(f"{LABELS.get(m, m)} -{n}"
                               for m, n in sorted(dropped.items()) if n)
            print(f"  dropped to pair: {detail}")
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--by-stage", action="store_true",
                    help="also break the table down by R1 / R3")
    ap.add_argument("--su-per-hour", type=float, default=SU_PER_GPU_HOUR,
                    help=f"service units per GPU-hour (default {SU_PER_GPU_HOUR:g})")
    ap.add_argument("--csv", type=Path, help="also write the overall table to this path")
    args = ap.parse_args()

    runs, warnings = collect()
    if not runs:
        raise SystemExit(f"no usable sidecars under {PRED}")

    exp = expected_cells()
    n_expected = sum(len(v) for v in exp.values())
    print(f"Training cost and convergence — {len(runs)} runs of {n_expected} expected "
          f"({100 * len(runs) / n_expected:.1f}%)")
    print(f"source: {PRED.relative_to(ROOT)}/*/ *.config.json     "
          f"charge: {args.su_per_hour:g} SU per GPU-hour")

    # Guard the charge rate against what the jobs actually requested.
    ncpus = {r.ncpus for r in runs if r.ncpus}
    if ncpus and ncpus != {12}:
        print(f"  ! sidecars record ncpus={sorted(ncpus)}, not 12 — "
              f"{args.su_per_hour:g} SU/h assumes 12 CPUs x "
              f"{QUEUE_CHARGE_RATE:g} SU. Pass --su-per-hour.")

    # Patience is what separates total epochs from epochs-to-best. If it ever varies,
    # the two epoch columns stop being comparable across models.
    pats = {r.patience for r in runs if r.patience is not None}
    if len(pats) > 1:
        print(f"  ! patience is not constant across runs: {sorted(pats)} — "
              f"the 'tot ep' column is not comparable between models")

    parity(runs)
    capped(runs)

    rows = table(runs, args.su_per_hour, "ALL AVAILABLE RUNS", expected=exp)

    missing = n_expected - len(runs)
    if missing:
        print(f"\n  {missing} run(s) outstanding — medians above are over what exists, "
              f"and each model's column covers a different set of cells.")
        have = defaultdict(set)
        for r in runs:
            have[r.model].add(r.cell)
        for model in MODEL_ORDER:
            gap = exp.get(model, set()) - have.get(model, set())
            for stage, target, fold, seed in sorted(gap):
                print(f"      missing  {LABELS.get(model, model):<20} {stage:<24} "
                      f"{target}  fold{fold}  s{seed}")

    paired(runs, args.su_per_hour)

    if args.by_stage:
        for stage in STAGES:
            sub = [r for r in runs if r.stage == stage]
            if sub:
                table(sub, args.su_per_hour, stage)

    for w in warnings:
        print(f"  ! {w}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
