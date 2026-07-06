# WD-MPNN Reimplementation — Code Review

Comparison of your `dmpnn` reimplementation against the paper (Aldeghi & Coley, *Chem. Sci.* 2022)
and the official reference code (`polymer-chemprop`, `chemprop/models/mpn.py` + `chemprop/features/featurization.py`).

## Headline diagnosis

The paper reports, for the **Held-One-Monomer-Out** setting:

| Model | IP RMSE (eV) |
|---|---|
| Baseline D-MPNN (monomers only, **no** connectivity) | ~0.20 |
| **wD-MPNN** (connectivity + stoichiometry) | **~0.09** |

Your result is **RMSE 0.219**. That is essentially the *baseline* D-MPNN level, not the wD-MPNN
level. This is the key clue: it strongly suggests the inter-monomer connectivity and edge weighting
are **not actually reaching the network** — the model is behaving as if it sees disconnected monomers.
That points the finger at graph construction / message routing rather than at hyperparameters.

The review below is consistent with that: the top-ranked bug destroys edge routing.

---

## Ranked issues

### 1. CRITICAL — `edge_index` pairs each bond's source with the WRONG target (graph is scrambled)

**Your code:** `chemprop/featurizers/molgraph/molecule.py:603-605`
```python
src = b2a
tgt = [a for a in range(n_atoms) for _ in a2b[a]]
edge_index = np.array([src, tgt], dtype=int)
```

`src = b2a` is in **bond-creation order** (`b2a[k]` = source atom of bond `k`). But `tgt` is built by
flattening `a2b` **grouped by atom**, so `tgt[k]` is *not* the target of bond `k`. The two rows of
`edge_index` are in different orders, so every column `(src[k], tgt[k])` is a bad pair, while `E[k]`,
`W_bonds[k]` and `rev_edge_index[k]` remain in bond order — everything is mutually inconsistent.

I reproduced the exact indexing logic on a 4-atom chain (0-1-2-3):

```
bond k : (src, tgt) produced | TRUE target
  0: (0, 0)  | 1   MISMATCH
  1: (1, 1)  | 0   MISMATCH
  2: (1, 1)  | 2   MISMATCH
  3: (2, 2)  | 1   MISMATCH
  4: (2, 2)  | 3   MISMATCH
  5: (3, 3)  | 2   MISMATCH
reverse-edge invariant (edge_index[:,rev[k]] == swap(edge_index[:,k])):  FALSE
```

For a chain it degenerates to **all self-loops** — no information flows between atoms at all. On real
molecules it produces a randomly rewired graph. `edge_index[0]` (source) is correct, so `initialize()`
(`h⁰`) is fine, but **all message routing between atoms is corrupted**, and the reverse-edge
bookkeeping used to exclude the back-bond is wrong.

**Official** (`featurization.py`) never rebuilds a COO `edge_index`; it uses `a2b`/`b2a`/`b2revb`
directly, so the pairing is correct by construction.

**Why it matters:** this alone can explain most of the gap. It disables the whole premise of the model
(learning from monomer connectivity), which is exactly consistent with your baseline-level RMSE.

**Fix** — build the target per bond, not by flattening. During construction you already know it: for
`b1 = a1→a2` the target is `a2`, for `b2 = a2→a1` the target is `a1`. Keep a parallel list:
```python
b2tgt = []
...
# every place you do b2a.append(a1); b2a.append(a2)
b2tgt.append(a2); b2tgt.append(a1)   # target of each directed bond
...
edge_index = np.array([b2a, b2tgt], dtype=int)
```
Then verify: `edge_index[0, rev] == edge_index[1]` and `edge_index[1, rev] == edge_index[0]` for every
bond.

---

### 2. HIGH — residual uses the post-activation `h⁰` instead of the pre-activation `W_i` output

**Official** `mpn.py:96,123`:
```python
input   = self.W_i(f_bonds)                 # pre-activation
message = self.act_func(input)              # h^0 for the running state
...
message = self.act_func(input + message)    # residual anchor = PRE-activation input
```

**Your code** `nn/message_passing/base.py:202-204` and `mixins.py` `initialize`:
```python
H_0 = self.initialize(bmg)   # weighted initialize already returns tau(W_i(...))
H_0 = self.tau(H_0)          # base.py: H_0 overwritten to POST-activation
...
H = self.update(M, H_0)      # update: tau(H_0 + W_h M)  -> anchor = POST-activation
```

The official (and standard chemprop) adds the **pre-activation** `W_i(f_bonds)` back at every step; you
add `τ(W_i(...))` (a non-negative, ReLU'd tensor). Different residual anchor → different fixed point of
the iteration. This affects **every** message-passing step and both the weighted and unweighted paths.

**Fix:** keep `H_0` as the raw `W_i` output and only activate the running copy:
```python
H_0 = self.W_i(cat[x_v, E])   # no tau
H   = self.tau(H_0)
for t in ...:
    H = self.update(self.message(H, bmg), H_0)   # H_0 stays pre-activation
```

---

### 3. MEDIUM — reverse message is weighted in your code but unweighted in the official

**Official** `mpn.py:117-120`:
```python
nei_a_message = nei_a_message * nei_a_weight[..., None]   # incoming weighted
a_message     = nei_a_message.sum(dim=1)
rev_message   = message[b2revb]                           # <-- NOT weighted
message       = a_message[b2a] - rev_message
```

**Your code** `nn/message_passing/mixins.py:40-41`:
```python
rev_msg = H[b2revb] * w_bonds[b2revb].unsqueeze(-1)       # <-- weighted
msg     = a_msg[b2a] - rev_msg
```

Net contribution of the back-bond differs: official leaves `(w_rev − 1)·m_rev`, you leave `0`. For
non-stochastic bonds (`w = 1`) both agree, so this **only affects the stochastic inter-monomer edges** —
precisely the edges that make wD-MPNN better than the baseline. Your version is arguably closer to the
idealized paper equation `mᵥw = Σ_{k≠w} w_kv h_kv`, but to **reproduce the reported numbers** you must
match the official (subtract the *unweighted* reverse). Flagging as a deliberate-decision point.

---

### 4. MEDIUM — extra BatchNorm on the graph embedding (not in the original)

**Your code** `models/model.py:169-173`:
```python
H = self.bn(H)                                  # BatchNorm1d on pooled graph vector
if isinstance(bmg, BatchPolymerMolGraph):
    H = H * bmg.degree_of_polym.unsqueeze(1)    # then Xn scaling
```

The official readout (`mpn.py:145-171`) has **no** BatchNorm on the molecule vector: it does the
weighted mean, multiplies by `degree_of_polym`, and hands the result to the FFN. Adding BN changes the
scale/normalization the FFN sees and interacts with the `Xn` multiply that follows it. It may help or
hurt, but it is a deviation from the reference architecture — turn it off when trying to reproduce.

(Note: `degree_of_polym` itself is correctly implemented and applied — good.)

---

### 5. LOW / VERIFY — atom & bond featurizer parity

Your `initialize()` concatenates `V[source] ‖ E` and the official concatenates `f_atoms[a1] ‖ f_bond`,
which is equivalent *provided the atom/bond feature sets are identical*. Confirm that your
`MultiHotAtomFeaturizer` / `MultiHotBondFeaturizer` reproduce the official `atom_features()` /
`bond_features()` exactly (same feature list, same one-hot bucket sizes, same handling of the wildcard/
attachment atoms). `DEFAULT_POLY_ATOM_FDIM=72` / `DEFAULT_POLY_BOND_FDIM=86` are hard-coded in
`setup()`; verify these equal the runtime dims (`self.atom_fdim`, `self.bond_fdim` are reset in the
featurizer) so `W_i` isn't silently built at the wrong width. A systematic feature mismatch shifts every
prediction.

---

### 6. LOW / VERIFY — data, split, and units

Confirm, independent of the model code, that: the HOMO 9-fold split matches the official
held-one-monomer-out folds; targets are in the same units (eV) and use the same normalization/scaling as
the official pipeline; and the polymer strings carry the same `~Xn` degree-of-polymerization tokens the
paper used. Also remove the debug `print(...)` in `base.py:257` — harmless, but a sign the config path
should be double-checked.

---

## Suggested order of attack

1. **Fix #1 (edge_index)** and re-run one fold. This is the most likely single cause; expect a large
   jump toward ~0.09–0.12 if it was the dominant issue. Add the `rev`-invariant assertion so it can
   never silently regress.
2. **Fix #2 (pre-activation residual)** to match the official update exactly.
3. **Align #3 (unweighted reverse)** and **disable #4 (BatchNorm)** for a faithful reproduction run.
4. Then confirm #5/#6 (featurizer + data parity) before concluding.

## Things that are already correct

- Weighted incoming-message aggregation in `message()` (weights gathered per target atom). 
- Weighted mean readout `Σ wᵥhᵥ / Σ wᵥ` (`agg.py`) matches `mpn.py:156-159`.
- `degree_of_polym = 1 + log10(Xn)` and its multiply at readout.
- Depth accounting (`range(1, depth)` = `depth-1` iterations) matches the official `range(depth-1)`.
- Source-atom features in `h⁰` (`V[edge_index[0]] ‖ E`) match `f_atoms[a1] ‖ f_bond`.
