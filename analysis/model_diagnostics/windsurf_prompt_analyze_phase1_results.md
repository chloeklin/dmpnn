# Windsurf prompt — analyze & report: junction n=1, octamer LOMO, seeds 43/44

Copy the block below into Windsurf. It only **reads existing run outputs and updates the report** — it must not launch or re-run any training.

---

**Task: parse three completed result sets, compute the standard diagnostics, and write them into `analysis/model_diagnostics/variant_results_report.md`. Do NOT run/submit any training or re-launch jobs — read the existing output files only.**

## Guardrails (project conventions — follow exactly)
- **LOMO / monomer-heldout aggregates = MEDIAN across folds**, not mean (means are dragged by pathological folds — e.g. ChemArch EA fold-6, junction IP fold-5). Report mean too, but headline the median.
- **Metric decomposition:** `group-mean R²` = chemistry baseline; `ΔR²` + `pairwise ordering accuracy` = architecture recovery; `overall R²` / `overall MAE (eV)` = bottom line. Keep these three axes separate in every table.
- **Seed statistics:** treat the 9 folds as the replication unit. For seeds 42/43/44, first average each fold's metric across the 3 seeds, then take median-across-folds of those per-fold means; report per-fold seed spread as mean ± std. **Do paired Wilcoxon across the 9 folds only — never across fold×seed pseudo-replicates.**
- Annotate any off-scale/pathological fold rather than letting it distort an aggregate; handle NaN/inf explicitly (report count dropped, don't silently drop).
- Before editing the report, dump all computed numbers to a scratch file `analysis/model_diagnostics/_phase1_metrics_scratch.md` so they can be checked; then transcribe into the report.

## Result set 1 — junction `n_coupling_steps=1`
Compare **junction n=1 vs junction n=2 vs baseline HPG-hier**, EA and IP, LOMO.
- Per-target medians: group-mean R², ΔR², ordering, overall R², overall MAE.
- Per-fold table (n=1 vs baseline and vs n=2), flag notable moves.
- **Decision question to answer explicitly:** did n=1 keep the EA chemistry win (fold-1 sulfone rescue; EA LOMO group-mean ≥ wDMPNN 0.965) **while recovering** the IP chemistry (fold-5 bithiophene) and architecture (IP ΔR², EA fold-6) that n=2 damaged? State whether n=1, n=2, or baseline is the best single model on the current evidence.

## Result set 2 — octamer (explicit sequence) LOMO (38-cell run)
Compare **octamer vs baseline HPG-hier vs wDMPNN**, EA and IP, LOMO.
- Medians + per-fold: group-mean R², ΔR², ordering, overall R²/MAE.
- **Decision question:** does the architecture recovery seen in the GD/fold-0 re-gate (ΔR² 0.92, ordering 0.89) **hold on unseen chemistry across all folds**, or was fold-0 unrepresentative? Does octamer beat baseline HPG-hier on architecture without losing chemistry? Give a keep / drop / needs-more-seeds recommendation.

## Result set 3 — seeds 43/44 (with existing seed 42)
Re-run the aggregation for every model/variant that now has 3 seeds. Produce error bars and **state which seed-42 claims survive**:
1. **EA LOMO chemistry: HPG-hier+junction ≥ wDMPNN** — holds with error bars?
2. **fold-1 sulfone rescue** (0.575→0.925) — magnitude and CI across seeds.
3. **IP fold-5 collapse under junction** (0.770→0.494) — real across seeds?
4. **HPG-hier + wDMPNN ensemble beats both** (EA MAE, IP R²/MAE) — holds?
For each: report the 3-seed mean ± std per relevant fold, the median-across-folds, and the paired-Wilcoxon p (across folds) for the head-to-head. Mark each claim **confirmed / weakened / overturned**.

## Output
1. Update `variant_results_report.md`:
   - §5 (Phase-1 variants): fill junction n=1 result under the existing junction subsection; update the octamer subsection from "LOMO pending" to the actual LOMO numbers + recommendation.
   - §6 (Variant status table): update statuses for `+junction (n=1)` and `+Q2 octamer`.
   - §7/§8: update open questions and caveats — move any now-answered items out, note error-bar status.
2. Add a short **"What changed / what flipped"** section at the top of the report (3–6 bullets): every seed-42 claim that changed once seeds/LOMO landed, and the current best single model.
3. Keep the scratch metrics file for verification.

**Report back:** the "what flipped" summary, the best-single-model call, and any claim that overturned — plus a list of the exact output files you parsed so I can spot-check.

---