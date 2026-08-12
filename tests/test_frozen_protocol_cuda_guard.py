"""Tests for the frozen-protocol CUDA guard in run_hpg_generalization.py."""

import argparse
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "python"))

from run_hpg_generalization import _check_frozen_protocol_accelerator, _smoke_suffix


def _args(frozen_protocol: bool = False, allow_non_cuda: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        frozen_protocol=frozen_protocol,
        allow_non_cuda=allow_non_cuda,
        repeat=None,
        stability_fix="none",
    )


def test_frozen_protocol_rejects_non_cuda():
    args = _args(frozen_protocol=True, allow_non_cuda=False)
    env = {"accelerator": "mps"}
    with pytest.raises(RuntimeError) as exc_info:
        _check_frozen_protocol_accelerator(args, env)
    msg = str(exc_info.value)
    assert "Frozen protocol training must use CUDA" in msg
    assert "2026-08-11 local MPS / CUDA mix" in msg
    assert "__localsmoke" in msg


def test_frozen_protocol_rejects_cpu():
    args = _args(frozen_protocol=True, allow_non_cuda=False)
    env = {"accelerator": "cpu"}
    with pytest.raises(RuntimeError) as exc_info:
        _check_frozen_protocol_accelerator(args, env)
    assert "Detected accelerator is 'cpu'" in str(exc_info.value)


def test_allow_non_cuda_permits_mps():
    args = _args(frozen_protocol=True, allow_non_cuda=True)
    env = {"accelerator": "mps"}
    # should not raise
    _check_frozen_protocol_accelerator(args, env)


def test_unfrozen_protocol_ignores_accelerator():
    args = _args(frozen_protocol=False, allow_non_cuda=False)
    env = {"accelerator": "mps"}
    _check_frozen_protocol_accelerator(args, env)


def test_cuda_never_raises():
    args = _args(frozen_protocol=True, allow_non_cuda=False)
    env = {"accelerator": "cuda"}
    _check_frozen_protocol_accelerator(args, env)


def test_smoke_suffix_when_allowed():
    args = _args(allow_non_cuda=True)
    assert _smoke_suffix(args) == "__localsmoke"


def test_smoke_suffix_when_not_allowed():
    args = _args(allow_non_cuda=False)
    assert _smoke_suffix(args) == ""
