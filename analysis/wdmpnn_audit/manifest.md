# wD-MPNN vectorisation audit manifest

## Commits inspected

| commit | date | message | relevance |
|---|---|---|---|
| `bf312f6` | 2025-07-23 16:11:04 +1000 | `init commit` | before `mixins.py` existed |
| `d1c3202` | 2025-07-29 16:29:47 +1000 | `added w-DMPNN for chemprop` | first `_WeightedBondMessagePassingMixin` with `return self.W_h(message)` and post-activation `initialize` |
| `af5912c` | 2025-09-22 12:33:30 +1000 | `fix wdmpnn` | earlier wdmpnn fix |
| `0fe2fdb` | 2025-09-22 19:44:40 +1000 | `fix wdmpnn` | earlier wdmpnn fix |
| `61f0e16` | 2026-07-05 18:16:13 +1000 | `script` | `rev_msg = H[b2revb] * w_bonds[b2revb]` (weighted reverse) and `return self.W_h(message)` |
| `3282b22` | 2026-07-06 22:37:15 +1000 | `fix` | switched to unweighted reverse, pre-activation residual (`H0`), `update(M, H0)`; kept the Python loop |
| `ce28674` | 2026-07-30 23:04:41 +1000 (parent of `4d3eb86`) | — | pre-vectorisation loop version; identical `message` math to `4d3eb86` |
| `4d3eb86` | 2026-07-30 23:04:41 +1000 | `new` | vectorised the `message` loop; added `_build_a2b_padded`, `_get_a2b_padded_and_mask`, `_message_reference` |

The commit that introduced the vectorisation is **4d3eb86**. The pre-fix `message` is the parent `ce28674`; the post-fix `message` is the same logic built by `_build_a2b_padded` and cached.

## Judgement calls

1. The pre/post boundary is defined by the vectorisation commit `4d3eb86` (2026-07-30 23:04:41 +1000). Prediction files with mtime before that are pre-fix; files with mtime on or after are post-fix.
2. The tiny example in `tiny_example.py` uses `WeightedBondMessagePassing(d_v=2, d_e=2, d_h=2)` but bypasses `initialize()`; it provides an explicit `H` tensor to `message()` and `_message_reference()`.
3. The hand-derived expected values assume D-MPNN no-backtracking with an unweighted reverse term, matching the code comment in the current implementation.
4. The `analysis/architecture_diagnostic/wdmpnn_prefix_postfix_diffs.csv` output difference (max 0.3396 eV) is **not** reproduced by `tiny_example.py`; the two `message` implementations match to `atol=1e-6`. The 0.34 eV must come from another source (e.g. `initialize`, `update`, `W_h`, or a different pre/post training run).
5. The pre-fix / post-fix classification of checkpoints is based on file mtimes, not on stored commit hashes. Mtime gives an upper bound, not a guarantee.

## Outputs produced

- `analysis/wdmpnn_audit/code_comparison.md` — both `message` implementations, side-by-side explanation, and the verdict that they are mathematically identical.
- `analysis/wdmpnn_audit/tiny_example.py` — reproducible 3-atom-chain hand-check.
- `analysis/wdmpnn_audit/AFFECTED.md` — pre-fix and post-fix prediction inventory and consumers.
- `analysis/wdmpnn_audit/manifest.md` — this file.
