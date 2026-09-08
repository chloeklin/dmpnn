# wD-MPNN `message` vectorisation audit — code comparison

## 1. Commit that changed the method

| | |
|---|---|
| Commit | `4d3eb86d3ee0a646beab618e5e9c1cf33ec9d80f` |
| Date | 2026-07-30 23:04:41 +1000 |
| Message | `new` |
| Parent (pre-fix) | `ce28674f62ad908c0ce40150f72f4c92bf3dd0cd` |
| File | `chemprop/nn/message_passing/mixins.py` |

The diff replaces the in-line `message` implementation with a vectorised `message` that calls `_get_a2b_padded_and_mask`, and keeps the original loop implementation as `_message_reference`.

## 2. Pre-fix (loop) implementation

From `4d3eb86^` (`ce28674`), the `message` method did the following:

1. `b2a = edge_index[0]` — source atom of each directed bond.
2. Built `a2b_dict[a]`, a Python list of all bond indices whose target (`edge_index[1]`) is atom `a`, in the order they appear in `edge_index`.
3. Filled a `[num_atoms, max_nb]` tensor `padded`: row `a` contains `a2b_dict[a]` followed by `-1` padding.
4. Gathered `nei_h = H[padded.clamp_min(0)]` and `nei_w = w_bonds[padded.clamp_min(0)]`; padded rows were zeroed by `mask = padded >= 0`.
5. Computed `a_msg[a] = sum_b w_bond * H_bond` over the incoming bonds of `a`.
6. For each bond `i` from `u` to `v`, computed `msg[i] = a_msg[u] - H[rev_edge_index[i]]` — the reverse bond message is **unweighted**.

## 3. Post-fix (vectorised) implementation

The current `message` (post `4d3eb86`) is the same after the atom-to-incoming-bond map is built by `_build_a2b_padded`:

- `torch.bincount` gives the number of incoming bonds per atom.
- `torch.argsort(..., stable=True)` over `edge_index[1]` produces the same within-atom bond ordering as the Python loop.
- `_get_a2b_padded_and_mask` caches the map, so the `depth - 1` calls inside a forward pass do not rebuild it.

The arithmetic then mirrors the loop:

```python
nei_h = H[padded.clamp_min(0)]
nei_w = w_bonds[padded.clamp_min(0)]
nei_h = nei_h * nei_w.unsqueeze(-1) * mask.unsqueeze(-1)
a_msg = nei_h.sum(dim=1)
rev_msg = H[b2revb]
msg = a_msg[b2a] - rev_msg
```

The same lines appear in `_message_reference`.

## 4. Are they mathematically the same operation?

**Yes**, for the `message` step itself.

The only difference is how the padded atom-to-bond map is constructed. The loop appends in `edge_index[1]` order; the vectorised version uses a stable `argsort` on `edge_index[1]`, which preserves that same order within each target-atom group. After the map is built, the gather, weight multiplication, masking, sum, and unweighted reverse subtraction are identical. The regression test `tests/test_wdmpnn_message_vectorized.py` asserts `torch.allclose(_message_reference(H, bmg), message(H, bmg), atol=1e-6)` on graphs with differing degrees and an isolated atom.

## 5. Tiny-example verdict

`analysis/wdmpnn_audit/tiny_example.py` builds a 3-atom chain plus an isolated atom, provides an explicit `H`, and checks both implementations against a hand-derived D-MPNN message. Both `message` and `_message_reference` match the hand reference to `atol=1e-6` at step 1 and step 2.

## 6. What about the 0.34 eV pre/post prediction difference?

Because the two `message` implementations are mathematically identical, the 0.34 eV maximum difference in `analysis/architecture_diagnostic/wdmpnn_prefix_postfix_diffs.csv` is **not caused by the vectorisation**. The output difference is elsewhere — either in a different code change (the `rev_edge_index` handling, `initialize`, `update`, or `W_h`) or in how the two prediction sets were produced. This audit therefore does not settle which *prediction set* is correct; it only settles that the `message` step is correct in both forms.

## 7. Which is correct?

Both the loop and the vectorised `message` correctly implement the same D-MPNN no-backtracking rule:

\[ m_{vw}^{(t)} = \sum_{u \in \mathcal{N}(v) \setminus w} w_{uv} h_{uv}^{(t-1)} - h_{wv}^{(t-1)} \]

where the reverse bond message `h_{wv}^{(t-1)}` is **unweighted**. The tiny example confirms this. There is no evidence that either `message` form is wrong.
