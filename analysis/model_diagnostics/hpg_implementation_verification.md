# HPG Reimplementation — Verification Report

**Compared:** original `HPG/src/{hpggat.py, mol2graph.py, polyG.py, smiles_utils.py}` (DGL, Han et al. 2025) vs your pure-PyTorch port `chemprop/{nn,data,models,featurizers/molgraph}/hpg.py`.

**Verdict:** the GAT math and graph topology are faithfully reproduced; the default `pooling_type="sum"` matches the original readout. There is **one documentation/feature mismatch** (atom features are 130-d, not the 49-d the docstrings claim), **one genuine structural difference** (connection-point atoms are dropped instead of kept as Mg pseudo-atoms), and **two minor deviations**. None is fatal, but two should be resolved or explicitly documented before you call it "HPG reproduced."

---

## ✅ Faithful (verified line-by-line)

- **GAT layer** (`HPGGATLayer` vs original `GAT`): attention `a = (h_src·attn_src) + (h_dst·attn_dst) + W_edge(e)`, LeakyReLU, **edge-softmax over incoming edges per destination**, message `m = α · W_node(h_src)` (**edge enters attention only, not message content**), scatter-sum aggregate, **mean over heads**. Matches exactly.
- **Encoder shape**: depth 6, hidden 128, 8 heads, scalar edge dim 1, per-layer distinct weights, LeakyReLU activation between layers. Matches original `GATNet` (`dims=[49]+[128]*6`, aside from the 49→130 input change below).
- **Graph topology**: fragment nodes first, then atoms; three edge types — fragment↔fragment = `degree`, atom↔atom = bond order (incl. **aromatic = 1.5**), atom→fragment = 1.0 (directed). Fragment nodes initialised to `ones(d_v)`. All match `pol2hig_mk2` / `mon2hig_mk2`.
- **Readout**: `pooling_type="sum"` = sum over **all** nodes → linear (this is the original `dgl.sum_nodes` readout, and it is the default). Faithful.

## ⚠️ Deviation 1 — atom features are 130-d, not 49-d (docstrings are stale)

- **Original:** 49-d = ~20 symbol (`one_of_k` over a fixed symbol set) + 5 H-count + 7 degree + 1 aromatic + 6 hybridization + 1 ring + 9 formal-charge.
- **Yours:** `HPG_ATOM_FDIM = 130` = **101 atomic-number one-hot (Z=1..100 + unk)** + 5 + 7 + 1 + 6 + 1 + 9. Element identity is encoded as an atomic-number one-hot ("to match Chemprop v1"), not the original's symbol set.
- The code is **internally consistent** (fragment nodes are `ones(130)`, `d_v=130`), so it runs — but the docstring/comments claiming "*exactly replicate the original 49-dim encoding*" and the `# 49-dim` comments (featurizer lines ~7–9, 74; data `d_v` comments) are **wrong/stale** and will mislead anyone reading it.
- **Impact:** this is a *justified* choice — matching your chemprop-v1 atom features gives a **fair internal comparison** to wDMPNN/ChemArch (same atom features across all models). But it is **not a literal HPG reproduction**, so don't claim it is. **Fix:** correct the docstrings, and decide explicitly (see below).

## ⚠️ Deviation 2 — connection points handled differently (genuine, and relevant to Increment 2)

- **Original:** wildcards `[R],[Q],[T],[U]` are replaced by **Mg pseudo-atoms** (`[Mg:1]`…) and **kept as real atom nodes** — the junction is an actual atom in the graph, participates in message passing and in the sum-pool readout, and the junction bond is present.
- **Yours:** wildcards are **removed**; "any bond touching a wildcard" is skipped. Inter-monomer connectivity is represented **only** via the abstract fragment↔fragment edge, not at the atom level. A junction atom therefore has a **different local environment** (lower degree, missing junction bond) than in the original.
- **Impact:** arguably cleaner (topology instead of Mg placeholders), but it **loses atom-level junction chemistry** — which for conjugated EA/IP copolymers is exactly the cross-monomer coupling Increment 2 wants to add. **Flag it**, and note that Increment 2 is partly *restoring* what the original encoded via Mg atoms. Confirm this matches how your wDMPNN treats junctions, for a fair comparison.

## ⚠️ Minor deviations

3. **LeakyReLU slope in attention:** yours uses `negative_slope=0.2` (the canonical GAT value); the original uses `F.leaky_relu(attn)` = torch default **0.01**. Yours is arguably better, but it is a deviation — set to 0.01 if you want bitwise fidelity, otherwise document.
4. **Multi-component / conditions machinery dropped:** original concatenates up to 6 components (polymer+salt+solvent+…) + weight ratios + temperature before the head. Yours is single-component (just the copolymer). **Correct simplification** for EA/IP (no salt/temperature), not a bug — just note the original's head (`linear_g1 128→64` → 512-hidden → out) differs from your `linear_pool 128→64` → FFN.

## Recommended actions before running the baseline

1. **Fix the stale "49-dim" docstrings/comments** (featurizer + data). They currently assert something false.
2. **Pick and document the baseline's intent, explicitly:**
   - *Recommended:* keep **130-d chemprop-v1 atom features** and call it "**HPG-GAT re-featurized to our atom set**" — this is the right choice for a controlled comparison against wDMPNN/ChemArch (atom features held constant across models). State it in the paper.
   - *Optional literal check:* add a 49-d original-feature mode + keep-Mg-junction mode, run once, and confirm you reproduce the paper's qualitative behaviour on a shared task. Nice-to-have for a reviewer, not required for your comparison.
3. **Add a small unit/smoke test:** one known copolymer → assert (a) `n_fragments` fragment nodes exist, (b) edge counts per type are correct, (c) `V.shape[1] == 130`, (d) a forward pass runs and gradients flow, (e) `pooling_type="frac_weighted"` at init ≈ your Frac baseline, and `HPG_frac_*` zero-init variants start ≈ HPG_frac.
4. **Sanity target:** the `sum` baseline should train to sensible EA/IP R² on Group-disjoint fold 0 before you trust the sweep.

*Code refs: `chemprop/nn/hpg.py` (HPGGATLayer, HPGMessagePassing), `chemprop/models/hpg.py` (HPGMPNN, VALID_POOLING_TYPES, `_pool_sum`), `chemprop/data/hpg.py` (HPGMolGraph, BatchHPGMolGraph), `chemprop/featurizers/molgraph/hpg.py` (HPGMolGraphFeaturizer, `_hpg_atom_features`, HPG_ATOM_FDIM=130). Original: `HPG/src/hpggat.py` (GAT, GATNet), `mol2graph.py` (mol2dgl_single, pol2hig_mk2), `smiles_utils.py` (49-d atom features).*
