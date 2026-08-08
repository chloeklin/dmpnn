# Models under comparison, and what they cost

Prepared 30 July 2026 for a supervisor catch-up. Every architectural statement below is taken
from the code, with file and line references. Every cost number is computed from the 481 run
sidecars, not estimated.

---

## 1. The prediction task

Each row of `data/ea_ip.csv` is one copolymer built from two monomers, **A** and **B**. There are
9 A monomers × 682 B monomers × 7 composition/architecture cells = **42,966 rows**. We predict two
electronic properties, `EA_vs_SHE_eV` and `IP_vs_SHE_eV`.

Each row carries:

- `smiles_A`, `smiles_B` — the two monomer structures
- `fracA`, `fracB` — how much of each monomer is present (0.25 / 0.5 / 0.75)
- `poly_type` — how the monomers are arranged along the chain: `block`, `alternating`, or `random`

**The central difficulty.** Which monomers you use explains ~90% of the variance in the targets.
How they are *arranged* explains ~1% (0.98% for EA, 1.46% for IP, measured within fixed A, B and
fracA). So a model can score extremely well overall while being completely blind to architecture.

That is why we report two numbers, not one:

- **Group-mean R²** — can the model place a chemistry (an A–B pair) at roughly the right value?
  This is the easy 90%.
- **ΔR²** — *within* a fixed A–B pair, can the model tell the seven architectures apart and rank
  them correctly? This is the hard 1%, and it is what the whole comparison is about.

All five models below are trying to capture that second thing. They differ in how much structural
information they are architecturally *able* to see.

---

## 2. wDMPNN — the baseline

*(`scripts/python/run_wdmpnn_generalization.py`; Aldeghi & Coley style)*

**The idea in plain terms.** Treat the copolymer as one ordinary molecule graph and let information
flow between neighbouring atoms. Composition is handled not by changing the graph's shape but by
*weighting* it — bonds and atoms carry numerical weights reflecting how much of each monomer is
present.

**What the code builds** (lines 271–278):

```python
mp  = nn.WeightedBondMessagePassing()   # messages along bonds, scaled by weights
agg = nn.WeightedMeanAggregation()      # weighted average over all atoms -> one vector
ffn = nn.RegressionFFN(...)             # that vector -> predicted EA or IP
mpnn = models.MPNN(mp, agg, ffn, batch_norm=False)
```

- **Input:** one flat graph per polymer, spanning both monomers, with stoichiometric weights.
- **Output:** one number.
- **Structure:** a single level. Atoms are the only nodes.

**What it can and cannot see.** Composition reaches it as a weight. Arrangement — block versus
random versus alternating — has to be inferred from whatever the weighting scheme implies; there is
no explicit representation of chain sequence anywhere in the model. This is the honest reason we
expect it to do worse on ΔR², and the results are consistent with that (ΔR² ≈ 0.43–0.45 versus
≈ 0.80 for HPG-hier on the A split).

**Note for accuracy:** the file also contains `WDMPNNWithinGroupLoss` (line 92), an optional extra
loss term controlled by `lambda_within`. It defaults to 0 and was **not** used in this campaign, so
every wDMPNN run here is the plain `models.MPNN`. It belongs to a separate pilot.

---

## 3. HPG-hier — the two-stage hierarchical model

*(`chemprop/models/hpg_hier.py`, class `HPGHierMPNN`)*

**The idea in plain terms.** Do it in two levels instead of one. First understand each monomer on
its own. Then treat the polymer as a tiny graph whose *nodes are monomers*, and let the monomers
talk to each other. This makes composition and connectivity part of the structure rather than a
weight.

**Stage 1 — atoms to monomers** (`forward`, lines 239–250):

```python
atom_embeddings, _ = self.stage1(batch.atom_graph)      # message passing within each monomer
monomers = self._pool_monomers(atom_embeddings, ...)    # sum-pool atoms -> one vector per monomer
```

Output: one embedding per monomer. For a two-monomer polymer, two vectors.

**Stage 2 — a 2-node transition graph** (line 251 onward):

```python
h = self.stage2_input(torch.cat([monomers, batch.monomer_fracs.unsqueeze(-1)], dim=-1))
for layer in self.stage2:
    h = layer(h, batch.stage2_edge_index, batch.stage2_edge_features)
```

The graph has exactly two nodes, A and B. The monomer fraction is concatenated onto each node as an
input feature. The edges between them carry a **17-dimensional** feature vector: 16 dimensions of
port-pair features (which chemical attachment point joins to which) plus one transition-weight
scalar describing how likely an A→B step is. That transition weight is what encodes `block` versus
`random` versus `alternating`.

Each `Stage2Layer` (lines 37–48) does:

```python
messages  = MLP([h[source] ; edge_features])     # mode="feature"
aggregate = scatter_sum(messages, target)
h         = LayerNorm(h + MLP([h ; aggregate]))  # residual update
```

**Readout — stoichiometry-weighted** (line 268):

```python
polymers = scatter_sum(h * batch.monomer_fracs.unsqueeze(-1), batch.polymer_batch, len(batch))
```

Literally **fᴀ·hᴀ + f_B·h_B**. The composition weights the two monomer embeddings directly.

**Loss** (lines 271–274): plain mean squared error.

**What it can see that wDMPNN cannot.** Arrangement enters twice — once as the transition weight on
the edge, once through the port-pair features describing how the monomers actually join. This is
the architectural basis for Claim 7.

---

## 4. HPG-octamer — an explicit 8-slot chain

*(same file; `OctamerEncoder` line 87, `OctamerPathLayer` line 51)*

**The idea in plain terms.** Instead of representing the polymer as two nodes with a statistical
"how often does A follow B" weight, write out an actual short chain of eight monomer slots — for
example A-A-B-A-B-B-A-B — and pass messages along it. The arrangement stops being a statistic and
becomes a literal sequence.

**How the eight slots are chosen.** `_build_octamer_sequences` enumerates all `C(8, n_A)`
arrangements, where `n_A = round(8 · fᴀ)`, then:

- **`block` and `alternating`** (non-uniform transition matrix) → take the single most likely
  arrangement (argmax). **One** sequence.
- **`random`** (uniform transition matrix) → **sample 16** arrangements with replacement
  (`n_random_samples = 16`).

**The encoder** (line 114 onward): slot values 0/1 are mapped to the Stage-1 monomer embeddings,
then `OctamerPathLayer` runs bidirectional message passing along the 8-slot path:

```python
agg = scatter_sum(self.msg(h[src]), dst, n_nodes)
h   = LayerNorm(h + MLP([h ; agg]))
```

**An important limitation, visible in the code.** `OctamerPathLayer.forward` takes no edge features
at all — just `h`, `edge_index`, `n_nodes`. So the octamer **discards the 16-d port-pair features
and the transition weight** that HPG-hier uses. It gains an explicit sequence and loses the
chemistry of the junctions. This is a real trade, not a pure upgrade.

**Readout — attention pooling** (`AttentionReadout`, line 65): a learned score per slot, softmaxed,
then a weighted sum. Unlike the stoichiometric readout, the weights are learned rather than set by
composition.

**The 16 replicas are averaged inside the loss** (lines 254–260):

```python
all_preds     = self.head(oct_embeds)                       # one prediction per replica
pred_sum      = scatter_sum(all_preds, batch.octamer_polymer_batch, n_polymers)
replica_counts = torch.bincount(batch.octamer_polymer_batch, ...)
return pred_sum / replica_counts.unsqueeze(-1)              # the MEAN is what gets returned
```

`_loss` then takes MSE against that mean. **So on random-arrangement rows the octamer is trained as
a 16-member ensemble**, and the gradient it learns from is averaged over 16 samples. That has two
consequences worth stating separately:

1. Averaged predictions are more accurate than single ones.
2. An averaged gradient is less noisy, so training is more stable.

Point 2 is the leading explanation for why the octamer has the lowest seed-to-seed variability of
any model in the campaign. It may be doing more work than the "explicit sequence" story credits.

---

## 5. HPG-junction — letting the monomers mix before pooling

*(same file; `JunctionCouplingLayer` line 129, used at lines 242–248)*

**The idea in plain terms.** In HPG-hier, each monomer is summarised *in isolation* before the two
summaries are allowed to interact. But in a real polymer the atoms near a junction are chemically
influenced by the monomer on the other side. Junction coupling lets atoms see across the A–B
boundary *before* the monomer summary is formed.

**What the code does** (lines 242–248):

```python
combined_ei = torch.cat([intra_ei, batch.junction_edge_index], dim=1)   # within-monomer + across
combined_w  = torch.cat([intra_w,  batch.junction_edge_weights])
for layer in self.junction_layers:
    atom_embeddings = layer(atom_embeddings, combined_ei, combined_w)
```

Ordinary intra-monomer bonds get weight 1; the new cross-monomer junction edges get their own
weights. This runs **before** `_pool_monomers`, so it changes what each monomer embedding contains.
After that, Stage 2 and the readout are identical to HPG-hier.

- **`hpg_hier_junction`** — 2 coupling steps (`n_coupling_steps=2`).
- **`hpg_hier_junction1`** — 1 coupling step. **Run on the A split (R1) only, by design.** Any
  comparison pooling all five models across both splits would silently discard all of R3; pair
  within split.

**Result so far:** junction coupling has not helped. On the B split it is behind HPG-hier on ΔR²
(median paired difference −0.046 EA, −0.054 IP), and it is also the *least* stable model
(median across-seed SD of ΔR² 0.140 on R1, roughly double HPG-hier's).

---

## 6. HPG-octamer K=1 — the planned ablation, no results yet

**The question.** The octamer is our best and most stable model. But it changes five things at once
relative to HPG-hier: topology (2 nodes → 8 slots), positional embeddings, readout (stoichiometric
→ attention), edge features (present → **absent**), and replicas (1 → 16). We cannot currently say
which of those five is responsible.

**What this arm changes.** One flag: `--n_random_samples 1` instead of 16. Nothing else.

**Why this particular one first.** It is the only ablation we can currently measure. Removing the
16 replicas removes both the prediction averaging *and* the gradient variance reduction at once, so
if ensembling is what's carrying the octamer, the effect should be **large**. The alternative
ablations (isolating topology and readout separately) are trying to split a difference of +0.019
ΔR², which is already smaller than our run-to-run variation — so they cannot give a readable answer
at three seeds, whatever they show.

**Scope.** It only affects `random` rows, because `block` and `alternating` take the argmax path
and build a single sequence regardless:

| `poly_type` | rows | share | affected |
|---|---|---|---|
| `random` | 18,414 | 42.9% | **yes** |
| `block` | 18,414 | 42.9% | no |
| `alternating` | 6,138 | 14.3% | no |

Results must be reported split by row type; a pooled number would dilute the effect by more than
half. `block` and `alternating` not moving is also a built-in correctness check on the arm.

Predictions and reading rules are recorded in advance in
`analysis/model_diagnostics/PREREG_octamer_k1_2026-07-30.md`.

**Important:** this changes the *training objective*, not just how many samples are drawn at test
time. It should not be described as "fewer test-time samples."

---

## 6a. How each model applies weights — the mechanism

This is the heart of the comparison, so it is worth being exact. All four models receive the same
two pieces of numerical information — the **monomer fractions** (`fracA`, `fracB`) and the
**transition probabilities** between attachment points — but they inject them at different places.

The transition probabilities come from the `WDMPNN_Input` string. For an alternating copolymer:

```
[*:1]c1cc(F)c([*:2])cc1F . [*:3]c1c(O)cc(O)c([*:4])c1O | 0.5 | 0.5 | <1-3:0.5:0.5<1-4:0.5:0.5<2-3:0.5:0.5<2-4:0.5:0.5
└──────── monomer A ────────┘ └──────── monomer B ───────┘  fracA  fracB  └────── transition rules ──────┘
```

Each `<i-j:w_ij:w_ji>` gives the probability of stepping from attachment point *i* to *j* and back.
The original featurizer validates that **incoming weights sum to 1 at every attachment point**
(`polymer-chemprop/chemprop/features/featurization.py:360-363`) — they are genuine probabilities,
not arbitrary scalars. For `block` the same-monomer transitions dominate (0.375 vs 0.125); for
`alternating` only cross-monomer transitions are non-zero.

### wDMPNN — weights multiply messages on edges, and atoms at readout

Two separate weightings, both in the original (`polymer-chemprop/chemprop/models/mpn.py`):

**(1) Edge weights during message passing** (lines 113–120). Bonds *inside* a monomer get weight
1.0; the stochastic bonds *between* monomers get the transition probability:

```python
nei_a_weight = index_select_ND(w_bonds, a2b)          # weight per incoming bond
nei_a_message = nei_a_message * nei_a_weight[..., None]   # scale each message
a_message = nei_a_message.sum(dim=1)
message = a_message[b2a] - rev_message                # D-MPNN's no-backtrack subtraction
```

So a message crossing an A→B junction is scaled by how likely that junction is. This is exactly the
paper's contribution: the polymer becomes an *ensemble* of chains, and the transition probabilities
weight the ensemble.

**(2) Atom weights at readout** (lines 153–159). Every atom carries the fraction of the monomer it
belongs to, and the readout is a fraction-weighted mean:

```python
mol_vec = w_atom_vec[..., None] * mol_vec
mol_vec = mol_vec.sum(dim=0) / w_atom_vec.sum(dim=0)  # weighted mean
mol_vec = degree_of_polym[i] * mol_vec                # 1 + log(Xn)
```

**Confirmed preserved in our chemprop-2.2.0 port.** `_WeightedBondMessagePassingMixin.message`
(`chemprop/nn/message_passing/mixins.py:38-40`) applies `nei_h * nei_w` identically, and
`WeightedMeanAggregation` (`chemprop/nn/agg.py:169`) computes `Σ w·h / Σ w`. The reverse message is
left unweighted in both versions.

**One deliberate omission that is harmless here.** Our port drops the `degree_of_polym` multiplier.
That is a no-op for this dataset: **0 of 42,966 rows** carry the `~Xn` degree-of-polymerisation
token, so `Xn = 1` and the original multiplier is `1 + log(1) = 1`. Worth stating rather than
leaving as a silent difference.

### HPG-hier — weights enter three times, never as a message multiplier

1. **As a node input feature** (line 251): `stage2_input(cat([monomers, monomer_fracs]))`. The
   fraction is concatenated onto each monomer node before Stage 2, so the network can learn a
   non-linear response to composition rather than only a linear scaling.
2. **As an edge feature, not a multiplier** (`Stage2Layer` mode `"feature"`, line 40):

   ```python
   messages = self.message(torch.cat([h[source], edge_features], dim=-1))
   ```

   The 17-dim edge vector — 16 port-pair dims plus the transition weight — goes *into* the MLP as
   input. Contrast wDMPNN, where the weight *multiplies* the message. HPG-hier can therefore learn
   an arbitrary function of the transition weight; wDMPNN is constrained to scale linearly by it.
   (The unused `"multiplier"` mode at line 43 would reproduce wDMPNN's behaviour — that is the Q1
   "wedge" variant, never regenerated.)
3. **As the readout weighting** (line 268): `scatter_sum(h * monomer_fracs, polymer_batch)` —
   literally **fᴀ·hᴀ + f_B·h_B**. Same idea as wDMPNN's weighted mean, applied over 2 monomer nodes
   instead of ~40 atoms.

### HPG-octamer — composition becomes slot counts; probabilities are discarded

The octamer converts both weightings into **structure**:

- **Composition → slot counts.** `n_A = round(8 · fracA)`, so fracA = 0.75 becomes six A slots and
  two B slots. The fraction is no longer a number multiplying anything; it *is* the sequence
  composition. This is why the stoichiometric readout would be equivalent to mean pooling on an
  8-slot chain — the counts already encode composition.
- **Transition probabilities → which sequence gets built.** Non-uniform (block, alternating) → the
  argmax arrangement. Uniform (random) → 16 sampled arrangements.
- **Then the probabilities are thrown away.** `OctamerPathLayer.forward(h, edge_index, n_nodes)`
  takes **no edge features and no weights** (line 59). Messages along the 8-slot path are
  unweighted:

  ```python
  agg = _scatter_sum(self.msg(h[src]), dst, n_nodes)
  ```

  So both the 16-dim port-pair features and the transition-weight scalar are unavailable to the
  octamer. It knows the *arrangement* but not the *chemistry of the joins*.
- **Readout uses learned weights, not composition weights** (`AttentionReadout`, line 65): a linear
  score per slot, softmaxed. The weights are learned rather than being read off `fracA`.

### HPG-junction — an extra weighted graph before pooling

Adds one more weighting, at the atom level, *before* monomers are summarised (lines 242–248):

```python
combined_ei = cat([intra_ei, batch.junction_edge_index])       # within-monomer + cross-monomer
combined_w  = cat([ones(n_intra), batch.junction_edge_weights])  # 1.0 inside, junction weight across
for layer in self.junction_layers:
    atom_embeddings = layer(atom_embeddings, combined_ei, combined_w)
```

This is structurally the closest thing in the HPG family to wDMPNN's mechanism — weighted atom-level
message passing across the junction, with intra-monomer bonds at weight 1.0. It then feeds the
ordinary HPG-hier Stage 2. Notably it has **not** helped, and it is the least stable variant.

### Summary

| | where composition enters | where transition probability enters | applied as |
|---|---|---|---|
| wDMPNN | atom weights at readout (weighted mean) | edge weights during message passing | **multiplier** |
| HPG-hier | node input feature **and** readout weights | edge feature into the message MLP | **learned function** |
| HPG-octamer | slot counts (`n_A = round(8·fᴀ)`) | selects which sequence is built, then discarded | **structure only** |
| HPG-junction | as HPG-hier | as HPG-hier, **plus** weighted atom-level junction edges | multiplier (atoms) + learned function (Stage 2) |
| HPG-octamer K=1 | as octamer | as octamer, but 1 sampled sequence instead of 16 | structure only |

The one-line version: **wDMPNN multiplies by the probabilities, HPG-hier learns a function of them,
and the octamer converts them into structure and then ignores them.**

---

## 7. Side by side

| | wDMPNN | HPG-hier | HPG-octamer | HPG-junction |
|---|---|---|---|---|
| levels | 1 (atoms) | 2 (atoms → monomers) | 2 (atoms → 8 slots) | 2, with cross-talk first |
| Stage-2 structure | — | 2-node graph | 8-slot chain | 2-node graph |
| arrangement represented as | stoichiometric weights | transition weight + port-pair features | an explicit sequence | transition weight + port-pair features |
| junction chemistry (16-d port-pair) | no | **yes** | **no — discarded** | yes |
| readout | weighted mean over atoms | fᴀ·hᴀ + f_B·h_B | learned attention over 8 slots | fᴀ·hᴀ + f_B·h_B |
| replicas per polymer | 1 | 1 | **16 on random rows, averaged in the loss** | 1 |
| batch size | 512 | 64 | 64 | 64 |
| epoch cap | 300 | 100 | 100 | 100 |
| LR | Adam + NoamLR, 1e-4 → 1e-3 → 1e-4 | Adam **flat 1e-3**, no scheduler | flat 1e-3 | flat 1e-3 |

LR sources: `chemprop/models/model.py:74-76` and `:358,375` for wDMPNN; `chemprop/models/hpg_hier.py:163`
(`init_lr: float = 1e-3`) and `:289` (`Adam(..., lr=self.hparams.init_lr)`) for the HPG family.
Neither is passed on the command line, so every run used its module default. **Both peak at 1e-3**;
wDMPNN warms up to it and decays away, HPG sits there throughout.

---

## 8. What each model cost

Computed from `wall_time_seconds` across all 481 runs. Gadi charging is
`SU = 3 × max(ncpus, mem_proportion) × walltime_hours`, and jobs request `ncpus=12`, so
**36 SU per GPU-hour**.

| model | runs | median wall time | median SU/run | median s/epoch | median best epoch | total spent |
|---|---|---|---|---|---|---|
| wDMPNN | 108 | 2.36 h | **85.0** | 134.2 | 48 | 9.70 kSU |
| HPG-hier | 107 | 1.01 h | **36.5** | 115.4 | 16 | 4.23 kSU |
| HPG-octamer | 107 | 1.14 h | 41.1 | 119.2 | 19 | 5.04 kSU |
| HPG-junction | 106 | 1.03 h | 37.1 | 119.1 | 16 | 4.23 kSU |
| HPG-junction-1 | 53 | 1.15 h | 41.3 | 117.6 | 18 | 2.45 kSU |
| | | | | | | **25.66 kSU** |

### ⚠ The table above is superseded — the wDMPNN figures were an implementation defect

**Do not quote any of the wDMPNN cost numbers above, and do not make any compute comparison
between the two model families.** Corrected 5 August.

The 85.0 SU/run and 134.2 s/epoch figures were dominated by a defect in our port, not by the
wD-MPNN method. `_WeightedBondMessagePassingMixin.message` rebuilt the atom-to-bond mapping with a
Python loop over a CUDA tensor — one host–device synchronisation per bond, executed
`depth − 1 = 2` times per forward pass. The HPG models were never affected: they use
`MABBondMessagePassing`, which contains no such loop.

After vectorising that loop and re-running wDMPNN under the original paper's configuration:

| | batch | epochs run | s/epoch | SU/run |
|---|---|---|---|---|
| **wDMPNN, original config, vectorised** | 50 | 30 | **20.2** | **6.0** |
| wDMPNN, regen_v1, pre-fix | 512 | ~63 | 134.2 | 85.0 |
| HPG-hier | 64 | ~31 | 115.4 | 36.5 |
| HPG-octamer | 64 | ~34 | 119.2 | 41.1 |

The full 54-run A-split wDMPNN arm cost **0.33 kSU**, against a projection of ~4.6 kSU.

**The claim reverses.** At comparable batch sizes (50 vs 64) and comparable epoch counts (30 vs
~31), wDMPNN is now roughly **6× cheaper per run** than HPG-hier, not 2.3× more expensive.

**But do not simply publish the inverse.** Two reasons:

1. HPG runs with `num_workers=0` against wDMPNN's `4`. That is an unfixed inefficiency on our
   side, not an architectural property.
2. Only the wDMPNN side has been re-measured post-vectorisation. The HPG timings predate that
   commit, and although HPG does not use the affected code path, they have not been re-run under
   identical conditions.

**Correct position: withdraw the compute claim entirely.** If it is wanted later, re-time HPG with
dataloader workers enabled and report per-epoch seconds on identical hardware — a few hundred SU,
since wall time per epoch is low-variance and needs neither multiple seeds nor a full arm.

The one internal comparison that survives untouched: **the octamer costs only ~13% more than
HPG-hier** (41.1 vs 36.5 SU) despite building and encoding 16 sequences on 42.9% of rows. Both run
at batch 64 under the same cap and LR, neither is affected by the defect, so this comparison is
clean.

### Why we are not "fixing" this by re-running wDMPNN at batch 64

wDMPNN's configuration is the baseline method's own published implementation. Re-running it at
batch 64 would not produce a fairer version of the published baseline — it would produce a
*different, unvalidated* method, for three reasons visible in the code:

1. `warmup_steps = warmup_epochs × steps_per_epoch` (`chemprop/models/model.py:364`). At batch 64
   there are 8× more steps per epoch, so the NoamLR warmup would stretch over 8× more updates.
2. `cooldown_epochs = trainer.max_epochs − warmup_epochs` (line 371). The decay *shape* is defined
   relative to the epoch cap, so changing the cap changes the learning-rate trajectory, not just
   the stopping point.
3. Consequently batch size, epoch cap and LR schedule are **coupled**. There is no such thing as
   changing only the batch size.

So a "matched-protocol wDMPNN" is a materially different training recipe that the original authors
never validated. Comparing against it would invite the opposite objection: that we hand-built a
weaker baseline.

**The position we take instead.** Report the comparison against the baseline *as published*, and
state the confound as a limitation rather than pretending to have removed it. Three claims are
available and they must be labelled differently:

| claim | status |
|---|---|
| ~~"HPG-hier is 1.16× faster per epoch"~~ | **Withdrawn 5 Aug.** Both figures came from wDMPNN timings inflated by the Python-loop defect. After vectorisation wDMPNN runs at 20.2 s/epoch against HPG-hier's 115.4 — the comparison reverses. |
| ~~"As configured by their authors, HPG-hier trains 2.3× cheaper per run"~~ | **Withdrawn 5 Aug.** Same cause. wDMPNN at its published configuration costs 6.0 SU/run against HPG-hier's 36.5. |
| "HPG-hier is architecturally cheaper" | **Not available and not true on current evidence.** |
| **No compute comparison between the families** | **The correct position.** Re-time HPG with dataloader workers before making any such claim. |

A batch-64 wDMPNN arm remains available as an explicitly-labelled **secondary control** (~4.6 kSU
for R3, cost unverified — wDMPNN has never been run at batch 64) if a reviewer asks whether the
baseline is merely under-tuned. It would be reported alongside, never in place of, the published
configuration. Given that the ΔR² gaps in question are 0.15–0.26 — roughly 6–16× the measured
run-to-run variation — a tuning artifact of that magnitude would be extraordinary, so this control
is a response to challenge rather than a prerequisite for the claim.

### Provenance, resolved against the original repository

Checked against the Aldeghi & Coley release (`polymer-chemprop`, **chemprop 1.4.0**,
`chemprop/args.py`). Our vendored copy is **chemprop 2.2.0** — a different major version, so this is
a reimplementation, not the original code.

**The defaults are the paper's configuration — confirmed from the SI.** Aldeghi & Coley state
explicitly (SI §"Baselines", p. S4): *"No hyperparameter optimization was performed for the D-MPNN
and wD-MPNN models and the validation set was used only for early stopping."* So the chemprop 1.4.0
CLI defaults are authoritative, and the table below is the paper's configuration, not a guess.

| field | paper / original default | ours | status |
|---|---|---|---|
| `batch_size` | **50** (`args.py:93`) | **512** | **10.2× larger** |
| `epochs` | **30** (`args.py:382`) | **300** cap | **10× larger** |
| early-stop halting | **none** — no `patience` exists in v1 | **patience 15** | **our addition** |
| validation set role | best-checkpoint selection only | best-checkpoint via `--frozen_protocol` | matches |
| `hidden_size` | **300** (`args.py:306`, and stated in SI p. S2) | 300 (`chemprop/conf.py:7`) | matches |
| `depth` | 3 (`args.py:308`) | 3 | matches |
| `warmup_epochs` | 2.0 (`args.py:384`) | 2 | matches |
| `init_lr` / `max_lr` / `final_lr` | 1e-4 / 1e-3 / 1e-4 (`args.py:389-393`) | same | matches |
| Xn embedding scaling | **diblock dataset only**, not EA/IP (SI p. S4) | omitted | **matches — correctly** |

So only three fields differ, and one of them (early-stop halting) has no counterpart in the original
at all. Two further points of agreement worth recording: our R1 split reproduces the paper's own
design — *"a 9-fold cross-validation where the dataset was split according to the identity of monomer
A"* (main text p. 7) — and the `degree_of_polym` multiplier our port drops was applied by the authors
only to the diblock copolymer phases dataset, never to EA/IP, which is independently consistent with
0 of 42,966 rows carrying a `~Xn` token.

Two observations.

**The batch/epoch rescaling was deliberate and sensible.** 50 → 512 is 10.2×; 30 → 300 is 10×. Since
NoamLR is defined over `total_steps = total_epochs × steps_per_epoch`
(`polymer-chemprop/chemprop/nn_utils.py:159`), scaling both by ~10× preserves the total optimiser-step
budget — roughly 20,000 steps either way at ~33,400 training rows. Whoever configured this kept the
schedule's shape intact. That deserves crediting rather than being called a deviation.

**But the early stop truncates the schedule, and the original has no early stop.** Chemprop 1.4.0
trains a fixed `epochs` and keeps the best-validation checkpoint; there is no `patience` anywhere in
`args.py` or `chemprop/train/`. Our patience-15 stop fires at a median `best_epoch` of 48, so
training ends around epoch 63 of the declared 300 — **~21% of the schedule**. Because the decay is
calibrated to reach `final_lr` exactly at `total_steps`
(`exponential_gamma = (final_lr/max_lr)^(1/(total_steps − warmup_steps))`, `nn_utils.py:162`), the
learning rate at the stopping point is approximately:

```
steps_per_epoch = ceil(33418/512) = 66;  warmup = 132;  total = 19,800;  stop ≈ 63 × 66 = 4,158
LR(stop) ≈ 1e-3 × 0.1^((4158−132)/19668) ≈ 6.2e-4
```

So **our wDMPNN stops at roughly 62% of peak learning rate and never anneals**, whereas the original
configuration (batch 50, 30 epochs, no early stop) runs its full ~20,070 steps and reaches 1e-4.
Learning-rate annealing usually matters substantially for final accuracy. This is a more specific
and more plausible under-training mechanism than batch size alone, and it is the one to state as a
limitation.

**Implication for the "don't deviate from the paper" concern:** the configuration already deviates
in three fields, one of which (early stopping) does not exist in the original at all. The cleanest
resolution is therefore not to defend the current settings, nor to invent a batch-64 variant, but to
**run the published configuration as published** — batch 50, 30 epochs, no early stopping. That is
the paper, requires no justification, and by the per-epoch timings would likely cost *less* wall
time than the current setup (30 epochs versus ~63).

### A confound that invalidates the timing comparison

`_WeightedBondMessagePassingMixin.message` (`chemprop/nn/message_passing/mixins.py:26-34`) rebuilds
the atom-to-bond mapping with a **Python loop over every bond in the batch**, on every call:

```python
a2b_dict = [[] for _ in range(len(bmg.V))]
for b_idx, tgt_atom in enumerate(edge_index[1]):   # iterates a CUDA tensor -> per-element GPU sync
    a2b_dict[tgt_atom].append(b_idx)
...
for a_idx, bond_ids in enumerate(a2b_dict):
    padded[a_idx, :len(bond_ids)] = torch.tensor(bond_ids, ...)
```

`message` is called `depth − 1 = 2` times per forward pass, so this runs twice per batch, iterating
tens of thousands of elements each time with a host-device synchronisation per element. The original
chemprop 1.4.0 precomputes `a2b` once in the featurizer and returns it from
`mol_graph.get_components()`, so it has no such cost. The HPG models use pure `scatter` operations
with no Python loops.

**Consequence: wDMPNN's 134.2 s/epoch is contaminated by an implementation inefficiency in our port,
not a property of the wD-MPNN method.** Neither the 1.16× per-epoch figure nor the 2.3× per-run
figure can currently be attributed to the architectures. **No compute comparison against wDMPNN
should be published from these runs.**

This does **not** affect any accuracy result. The loop is slow but correct — it produces the same
mapping — so Claims 7 and 8 and the octamer-versus-wDMPNN comparison stand unchanged. Only timing is
affected.

**Cheap path to a defensible number:** vectorise the mapping (it depends only on graph structure, so
it can be computed once in the collate function, or built with `argsort`/`scatter` and no loop), then
re-time on 2–3 cells with a single seed. Wall time per epoch is low-variance, so timing does not need
the full 54-run arm — roughly 250 SU buys a clean measurement.

### One genuinely clean efficiency result

**The octamer costs only 13% more than HPG-hier** (41.1 versus 36.5 SU) despite building and
encoding 16 separate 8-slot sequences on 42.9% of rows. Both run at batch 64 under the same cap and
the same flat LR, so this comparison is *not* protocol-confounded. The replicas are batched
together, so the ensemble is close to free. If the K=1 arm shows the replicas are doing the work,
that becomes a nice result: a 16-member ensemble for 13% more compute.

---

## 9. What is not yet established — state these plainly

1. **Nothing in this campaign reaches statistical significance.** With 9 folds the smallest
   attainable two-sided p from a paired sign test is 0.0039; within the S and D fold groups it is
   0.125 and 0.0625. We report consistency of direction and effect size, and say so.
2. **Claim 7 (HPG-hier recovers architecture better than wDMPNN) is well outside the noise** —
   median paired ΔR² difference −0.208 on cross-scaffold EA folds, with only 1 of 5 folds inside
   the measured seed SD. It is confounded by the protocol mismatch above, but the effect is large.
3. **Claim 8 (the octamer's advantage) is suggestive, not established** — +0.019 EA and +0.032 IP,
   consistent in direction across all 5 cross-scaffold folds, but 4 of 5 (EA) and 2 of 5 (IP)
   per-fold differences are smaller than the measured run-to-run variation.
4. **Run-to-run variation is large and is one of our findings.** Three runs identical in model,
   seed, split, code and GPU gave group-mean R² 0.450 / 0.790 / 0.978 — SD **0.268**. Full
   determinism was never enabled (no runner calls `torch.use_deterministic_algorithms`), so
   fixed-seed runs are not bit-reproducible. Reported results average three seeds at the prediction
   level to mitigate this.
5. **The octamer's advantage remains unattributed across five factors.** The K=1 arm addresses one
   of them. Positional embeddings and the discarded edge features stay confounded regardless.
