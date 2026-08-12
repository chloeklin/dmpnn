# Windsurf task — five corrections to the arch-spread work

Follow-up to `WINDSURF_PROMPT_arch_spread_recovery.md`. The metric implementation itself is
correct and verified — do not rewrite it. These are five specific corrections.

---

## 1. Two specified keys were dropped without disclosure

The spec asked for `arch_spread_within_2x_frac` (fraction with `0.5 <= ratio <= 2.0`) and
`arch_spread_n_groups` (number of contributing groups). Neither was implemented;
`arch_spread_true` and `arch_spread_pred` were added instead. The total came to 10 keys either
way, so the summary read as complete.

**Add both missing keys**, per stratum, keeping the two that were added. `arch_spread_n_groups`
is the important one — a median with no group count attached cannot be interpreted, and the
`_arch3` and `_arch2` strata have very different group counts.

If you believe a specified key is wrong or unnecessary, say so and leave it out **explicitly in
the report**. Substituting silently is the failure mode to avoid.

---

## 2. The "no widening" claim was incorrect — correct it and decide

`_regen_v1_results_individual_runs.csv` **did** gain 10 `arch_spread_*` columns, through the
`**metrics` dict splats at `scripts/python/analyze_regen_v1.py:91` and `:351`. §1.4 of the
original prompt asked for exactly this consumer to be found and reported before proceeding; the
report stated "no widening or breaking changes."

Actual harm is low — the file is gitignored (`.gitignore:185`) so nothing committed moved, and the
columns are additive. But this file is cited by `PREREG_octamer_posemb_2026-08-05.md` §5 as the
source of the R1 materiality threshold, so its schema is not incidental.

Do two things:

1. Correct the claim in your report, naming both splat sites.
2. Recommend, with reasoning, whether the `arch_spread_*` keys should be **excluded** from those
   two splats (keeping the per-run CSV schema stable and exposing the metric only where it is
   deliberately requested) or **kept and documented**. State the recommendation; do not implement
   the exclusion until it is approved.

---

## 3. Disambiguate the metric name — it currently means two things

`arch_spread_ratio_arch3` in `_regen_v1_results_individual_runs.csv` is computed **per run**
(one seed). The same name in the F2 assertions refers to the value computed from the **three-seed
prediction average**. These are different quantities.

Confirmed numerically — octamer EA fold 0: per-seed 0.6591 / 0.6668 / 0.7200, seed-averaged
**0.649**. The per-run values bracket the seed-averaged one, which is good evidence the
implementation is right, but they are not interchangeable and must not be quoted as one number.

Rename so the aggregation is visible at the point of use — for example a `_perrun` suffix on the
per-run columns, or `_predavg` on the seed-averaged ones. Pick one convention, apply it
everywhere the metric appears, and state it in the F2 manifest.

---

## 4. Write a dated addendum to the pre-registration — do not edit its body

`PREREG_octamer_posemb_2026-08-05.md` §5 defines the R1 materiality threshold as the *median
per-cell across-seed SD of `delta_r2` for `hpg_hier_octamer`, 18 cells*, from
`_regen_v1_results_individual_runs.csv`, and records it as **0.051**.

Recomputing that quantity from the file today gives **0.0446**.

The data has not drifted: the prereg's companion figure — mean SD 0.123 — reproduces exactly
(0.1232). And 0.051 is precisely the 10th of the 18 sorted values (IP fold 0, SD 0.0510), which
points to a median-convention slip when the pre-registration was written rather than any change
in the underlying runs.

Requirements, in this order of importance:

- **Do not change the threshold.** The analysis must continue to use the frozen **0.051**. Using
  a pre-registered value is the entire point of pre-registering it.
- **Do not edit the body of the pre-registration.** Dated documents are immutable in this
  repository; corrections go in a dated addendum at the bottom.
- Add an addendum dated **2026-08-11** recording: the documented value, the value that re-derives
  today, the evidence that the data is unchanged (mean 0.123 reproduces as 0.1232), the likely
  cause, and — stated explicitly — that **the outcome is unaffected**: median diffs of −0.0100
  (EA) and −0.0096 (IP) sit far inside both bands, and the same four fold-cells fall outside
  either threshold.
- Verify that last claim yourself rather than restating it. Recompute the verdicts under both
  0.051 and 0.0446 and confirm they are identical.

---

## 5. Tighten one float comparison

`_arch_spread_metrics` excludes degenerate groups with `if ts == 0.0`. The spec said
`true_spread < 1e-9`. An exact float comparison lets a group with a spread of, say, 1e-15 through,
producing a ratio around 1e15 that would wreck any median it entered.

Change to a tolerance, and report how many groups (if any) fall in `0 < ts < 1e-9` across the
R1 arm — most likely none, given these are eV-scale physical values, but confirm rather than
assume.

---

## 6. Process note

The octamer collapse-rate bound in the previous task failed at `6.6%` and the assertion was
rewritten to round before comparing. That patch is correct — the bound came from a rounded
printout, so the specification was at fault, not the code.

But the instruction was to stop and report on disagreement rather than adjust. Loosening a check
to make it pass is the one action that cannot be distinguished from hiding a real failure. When a
bound looks wrong, say so and propose the fix; don't apply it silently.

---

## Report back

For each of §1–§5: what changed, the numbers produced, and confirmation that no existing reported
value moved. `py_compile` clean on every modified file, and the unit tests re-run. List every file
written.
