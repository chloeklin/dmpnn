"""Guardrails for wDMPNN protocol-variant argument validation.

run_wdmpnn_generalization.py's `--protocol_variant original_paper` reproduces
polymer-chemprop 1.4.0's protocol (no early stopping, full epoch budget, best-
validation checkpoint selection). Since
`training_summary["prediction_checkpoint"]` is
`checkpoint_record(best_ckpt_path if frozen_protocol else last_ckpt_path)`,
running `original_paper` without `--frozen_protocol` would silently write
last-epoch predictions instead of best-validation predictions (HANDOFF §4).
These tests ensure `main()` fails fast in that case, and that the existing
`regen_v1` + `--frozen_protocol` + `patience != 15` guard still works.
"""
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "scripts" / "python"))

from run_wdmpnn_generalization import main  # noqa: E402


def test_original_paper_without_frozen_protocol_raises(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "run_wdmpnn_generalization.py",
        "--protocol_variant", "original_paper",
        "--batch_size", "50",
        "--epochs", "30",
        "--patience", "30",
    ])
    with pytest.raises(ValueError, match="original_paper requires --frozen_protocol"):
        main()


def test_regen_v1_frozen_protocol_requires_patience_15(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "run_wdmpnn_generalization.py",
        "--frozen_protocol",
        "--patience", "5",
    ])
    with pytest.raises(ValueError, match="regen_v1 with --frozen_protocol requires patience=15"):
        main()
