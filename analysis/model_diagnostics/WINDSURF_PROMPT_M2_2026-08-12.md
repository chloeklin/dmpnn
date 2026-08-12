# Windsurf task — implement M2 (8-slot chain WITH edge features) + Gadi pilot scripts

Rung 4 of `ARCHITECTURE_LADDER_2026-08-12.md`. **Implementation and PBS generation only —
do not train, do not submit, do not log in to Gadi.**

Independent of the M1 task; the two can proceed in parallel.

---

## 0. Pilot definition has changed

The arms C/D pilot used **2 folds × 1 target × 3 seeds**, and the two folds pointed in opposite
directions — fold 0 said the readout mattered, fold 4 said the topology did. The pilot could
not be read.

**A pilot is now: all 9 folds × one target (EA) × three seeds = 27 runs.** Protocol-compliant,
supports a 9-fold paired sign test (minimum attainable p = 0.0039), roughly half a full arm.
Do not pilot on a subset of folds.

---

## 1. Why M2 exists — a confound arms C and D cannot break

The intended 2×2 was topology × readout:

| | stoich / mean readout | attention readout |
|---|---|---|
| **2-node graph** | HPG-hier | arm C |
| **8-slot chain** | arm D | octamer |

But `hpg_hier.py:265` returns from the octamer branch **before** the `Stage2Layer` loop that
consumes `batch.stage2_edge_features`. So:

| | readout | topology | uses the 16-d port-pair features? |
|---|---|---|---|
| HPG-hier | stoich | 2-node | **yes** |
| arm C | attention | 2-node | **yes** |
| arm D | mean | 8-slot | **no** |
| octamer | attention | 8-slot | **no** |

Readout varies cleanly within each topology. **But every move between rows changes topology
*and* drops the junction edge features simultaneously.** Factors 1 and 4 are confounded, and no
amount of extra folds on arms C/D separates them.

**M2 is the missing cell: an 8-slot chain that *does* receive the edge features.** With it:

- `M2 − HPG-hier` = topology alone, edge features held present
- `arm D − M2` = edge features alone, topology held at 8 slots

---

## 2. What to implement

Extend `OctamerEncoder` (`chemprop/models/hpg_hier.py:87`) to optionally consume edge features
between adjacent slots.

**The features already exist and do not need computing.** `_stage2_edges`
(`chemprop/featurizers/molgraph/hpg_hier.py:112`) builds
`pairs[source, target]` — a `2 × 2 × 16` array of port-pair one-hots, four monomer-pair
combinations — plus the transition weight, giving the 17-d vector used by `Stage2Layer`.

For adjacent slots *i*, *i+1* holding monomers *(m_i, m_{i+1})* with values 0 or 1, the edge
feature is `pairs[m_i, m_{i+1}]` concatenated with `transition[m_i, m_{i+1}]`. **This is
indexing into an existing array, not new featurisation.**

`OctamerEncoder.forward` currently builds a bidirectional path edge index via
`_make_path_edge_index` and calls `OctamerPathLayer(h_flat, edge_index, n_nodes)`. You need to:

1. carry the `2 × 2 × 17` monomer-pair feature block through to the encoder — the featurizer
   must emit it alongside `octamer_sequences` for octamer mode, where today it emits it only
   for transition-graph mode
2. for each directed path edge, index the block by the (source monomer, target monomer) pair to
   get its 17-d vector
3. give `OctamerPathLayer` the same `mode` treatment `Stage2Layer` has, so edge features enter
   message construction the same way — **match `stage2_edge_weight="feature"`, which is what
   every current HPG run uses**, so M2 is comparable

**Do not invent a new way of using the features.** The point is to hold everything constant
except their presence. If `Stage2Layer` concatenates them onto the source embedding before the
message MLP, do exactly that.

Register the variant in `_VARIANT_FLAGS` (`run_hpg_generalization.py:79`) and a token in
`evaluation/naming.py`. Suggested: `hpg_hier_octamer_edges`.

### Watch the readout

M2's purpose is the **topology-versus-edge-features** contrast, so it should use the **mean**
readout, matching arm D. That gives a clean chain:

```
HPG-hier (2-node, stoich, +features)
   → M2   (8-slot, mean, +features)     = topology
   → armD (8-slot, mean, −features)     = edge features
   → octamer (8-slot, attention, −features) = readout
```

Note HPG-hier uses `stoich_weighted` and M2 uses `mean`. On an 8-slot chain the stoichiometric
readout **is** mean pooling over positions — the slot counts already encode composition, since
`n_A = round(8 · fracA)`. This is stated in HANDOFF §7 and is why arm D was defined this way.
**Say so explicitly in the pre-registration**, because it is the one step where a reader will
suspect two things changed.

---

## 3. Verification before any job is generated

1. **Parameter count.** M2 must differ from arm D by exactly the parameters the edge-feature
   pathway adds — the widened input to the message MLP. Compute it by hand from `d_h = 128` and
   the 17-d edge dimension, and confirm the model matches. Report the arithmetic, not just the
   number. Reference points: arm D = 133,376; octamer = 133,505; the difference of 129 is the
   attention readout.
2. **Edge features actually reach the layer** — non-zero gradient with respect to the
   edge-feature tensor after a backward pass. A silently-ignored input would make M2 a
   duplicate of arm D.
3. **Indexing is correct.** For a known polymer, verify that a slot pair (A→B) receives the
   same 17-d vector that `Stage2Layer` would use for the A→B monomer edge. Assert equality
   against `_stage2_edges` output rather than eyeballing.
4. **M2 differs from arm D** on a real forward pass. Identical outputs mean the features are
   not being used.

---

## 4. Pre-registration — before generating jobs

Create `analysis/model_diagnostics/PREREG_M2_2026-08-12.md`, following
`PREREG_octamer_posemb_2026-08-05.md`.

**Question.** Of the octamer's advantage over HPG-hier, how much is the 8-slot topology and how
much is the loss of the 16-d port-pair junction features?

**Primary quantity:** ΔR² on `all` rows, A split, per fold.

**Pre-registered readings:**

| Outcome | Reading |
|---|---|
| M2 ≈ octamer, arm D ≈ octamer | the edge features are irrelevant; **topology did it** |
| M2 ≫ arm D | the features matter, and the octamer wins *despite* discarding them |
| M2 ≈ HPG-hier | the 8-slot topology contributes nothing once features are held; **the loss of features is what changed the model** |
| M2 midway | topology and features interact; do not attribute |

The second row is the interesting one and worth calling out: if M2 beats the octamer, the
octamer is leaving information on the table and the obvious next model is 8-slot + attention +
features.

**Materiality threshold:** derive as in the posemb pre-registration, state the value and source
before running.

**State what this does not resolve:** position embeddings and sequence sampling are already
excluded; the protocol-matching confound is untouched by any rung.

---

## 5. PBS generator

Follow `scripts/shell/generate_octamer_posemb_r1.sh`, including the header stating it generates
only and does not submit.

`scripts/shell/generate_m2_pilot.sh` producing **27 jobs**: EA only, folds 0–8, seeds
42/43/44, our configuration (batch 64, 100 epochs, patience 15, 16 sampled sequences, chain
length 8, `--frozen_protocol`).

PBS header: `gpuvolta`, `ncpus=12`, `ngpus=1`, `mem=100GB`, `jobfs=100GB`,
`walltime=06:00:00`, project `hm62`, storage `scratch/um09+gdata/dk92`, modules
`python3/3.12.1 cuda/12.0.0`, venv `/home/659/hl4138/dmpnn-venv/bin/activate`,
`PROJECT_DIR=/scratch/um09/hl4138/dmpnn`.

Pre-flight: refuse to emit a job whose output exists; verify the HPG-hier, arm D and octamer
comparators exist for every fold. **Arm D currently exists for folds 0 and 4 only** — report
which folds lack an arm D comparator rather than failing silently, since the M2 ↔ arm D
contrast will be incomplete until the arm D pilot is extended.

Output token `__m2`, prediction directory `predictions/m2/`.

Estimated cost: 27 runs × ~36.5 SU ≈ **1.0 kSU**.

**Do not generate IP or the B split.**

---

## 6. Protocol

Three seeds averaged at the **prediction** level, metric computed once. `y_pred`, never
`y_pred_final`. `--frozen_protocol` with the CUDA guard in force. Do not modify
`evaluation/metrics.py`, any dated document, or any existing prediction.

---

## 7. Report back

1. The §3 verification, especially the parameter arithmetic and the `_stage2_edges` equality
   assertion.
2. The pre-registration file.
3. Which folds lack an arm D comparator.
4. Job count, and confirmation nothing was submitted or trained.
5. `py_compile` clean; tests; every file written.
