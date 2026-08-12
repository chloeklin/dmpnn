"""Unit tests for the arch-spread metrics (§1.5).

Five cases verified:
1. Perfectly-recovered group   — ratio == 1.0, within_2x == 1.0
2. Fully-collapsed group       — pred_spread == 0 → ratio 0.0, in collapsed_frac, excluded from logerr
3. Over-predicting group       — ratio == 2.0 → |log2(2)| == 1.0, within_2x boundary (2.0 included)
4. Degenerate group            — true_spread < 1e-9 → excluded from ALL counts (NaN result when sole group)
5. Mixed degenerate + valid    — degenerate excluded, valid computed normally
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.metrics import _arch_spread_metrics


def _make_frame(groups_spec: list[tuple]) -> pd.DataFrame:
    """Build a minimal sub_frame from (group_id, y_true_list, y_pred_list) tuples."""
    rows = []
    for gid, yt, yp in groups_spec:
        for t, p in zip(yt, yp):
            rows.append({"group": gid, "y_true": float(t), "y_pred": float(p)})
    return pd.DataFrame(rows)


# ── Case 1: perfectly recovered ───────────────────────────────────────────────

def test_perfect_recovery():
    """Single arch3 group where pred range == true range → ratio = 1.0."""
    frame = _make_frame([
        ("g1", [0.0, 1.0, 2.0], [0.5, 1.5, 2.5]),  # spread true=2.0, pred=2.0, ratio=1.0
    ])
    out = _arch_spread_metrics(frame, "_arch3")
    assert out["arch_spread_n_groups_arch3"] == pytest.approx(1.0)
    assert abs(out["arch_spread_ratio_arch3"] - 1.0) < 1e-12
    assert abs(out["arch_spread_true_arch3"] - 2.0) < 1e-12
    assert abs(out["arch_spread_pred_arch3"] - 2.0) < 1e-12
    assert out["arch_spread_collapsed_frac_arch3"] == pytest.approx(0.0)
    assert out["arch_spread_within_2x_frac_arch3"] == pytest.approx(1.0)  # 1.0 in [0.5, 2.0]
    assert not np.isnan(out["arch_spread_logerr_arch3"])
    assert abs(out["arch_spread_logerr_arch3"]) < 1e-12  # |log2(1)| == 0


# ── Case 2: fully collapsed ────────────────────────────────────────────────────

def test_collapsed_group():
    """Single group with pred_spread == 0: ratio=0.0, counted in collapsed_frac, excluded from logerr."""
    frame = _make_frame([
        ("g1", [0.0, 1.0, 2.0], [1.0, 1.0, 1.0]),  # pred constant → spread_pred=0, ratio=0
    ])
    out = _arch_spread_metrics(frame, "_arch3")
    assert out["arch_spread_n_groups_arch3"] == pytest.approx(1.0)
    assert abs(out["arch_spread_ratio_arch3"] - 0.0) < 1e-12
    assert out["arch_spread_collapsed_frac_arch3"] == pytest.approx(1.0)  # 100% collapsed
    assert out["arch_spread_within_2x_frac_arch3"] == pytest.approx(0.0)  # 0.0 not in [0.5, 2.0]
    assert np.isnan(out["arch_spread_logerr_arch3"])  # no nonzero ratios → NaN


# ── Case 3: over-predicting ────────────────────────────────────────────────────

def test_over_predicting():
    """Single group with ratio == 2.0 → |log2(2.0)| == 1.0; boundary included in within_2x."""
    frame = _make_frame([
        ("g1", [0.0, 1.0, 2.0], [0.0, 2.0, 4.0]),  # spread true=2.0, pred=4.0, ratio=2.0
    ])
    out = _arch_spread_metrics(frame, "_arch3")
    assert out["arch_spread_n_groups_arch3"] == pytest.approx(1.0)
    assert abs(out["arch_spread_ratio_arch3"] - 2.0) < 1e-12
    assert out["arch_spread_collapsed_frac_arch3"] == pytest.approx(0.0)  # 2.0 >= 0.25
    assert out["arch_spread_within_2x_frac_arch3"] == pytest.approx(1.0)  # 2.0 included (<=)
    assert abs(out["arch_spread_logerr_arch3"] - 1.0) < 1e-12  # |log2(2)| = 1.0


# ── Case 4: degenerate group (true_spread == 0) ────────────────────────────────

def test_degenerate_group_excluded():
    """Group with true_spread < 1e-9 is excluded from ALL counts → all NaN when it's the only group."""
    frame = _make_frame([
        ("g1", [1.0, 1.0, 1.0], [0.8, 1.0, 1.2]),  # all y_true equal → true_spread=0
    ])
    out = _arch_spread_metrics(frame, "_arch3")
    for key in ["arch_spread_n_groups_arch3", "arch_spread_true_arch3", "arch_spread_pred_arch3",
                "arch_spread_ratio_arch3", "arch_spread_collapsed_frac_arch3",
                "arch_spread_within_2x_frac_arch3", "arch_spread_logerr_arch3"]:
        assert np.isnan(out[key]), f"{key} should be NaN for degenerate-only frame"


# ── Mixed: degenerate + valid ────────────────────────────────────────────────

def test_degenerate_does_not_affect_valid():
    """A degenerate group alongside a valid group: degenerate excluded, valid computed."""
    frame = _make_frame([
        ("degen",  [1.0, 1.0, 1.0], [0.5, 1.5, 2.5]),  # excluded
        ("valid",  [0.0, 1.0, 2.0], [0.0, 0.5, 1.0]),   # spread true=2.0, pred=1.0, ratio=0.5
    ])
    out = _arch_spread_metrics(frame, "_arch3")
    assert abs(out["arch_spread_ratio_arch3"] - 0.5) < 1e-12
    assert out["arch_spread_collapsed_frac_arch3"] == pytest.approx(0.0)  # 0.5 >= 0.25


if __name__ == "__main__":
    test_perfect_recovery()
    test_collapsed_group()
    test_over_predicting()
    test_degenerate_group_excluded()
    test_degenerate_does_not_affect_valid()
    print("All 5 arch-spread tests passed.")
