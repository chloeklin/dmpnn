from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np
import torch


def sha256_file(path: str | Path) -> str:
    resolved = Path(path).resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_record(path: str | Path) -> dict[str, str]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def split_indices_sha256(train_indices, val_indices, test_indices) -> str:
    digest = hashlib.sha256()
    for label, values in ((b"train", train_indices), (b"val", val_indices), (b"test", test_indices)):
        array = np.asarray(values, dtype="<i8")
        digest.update(label)
        digest.update(np.asarray([array.size], dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def runtime_environment() -> dict:
    cuda_available = torch.cuda.is_available()
    driver_version = None
    if cuda_available:
        try:
            driver_version = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
            ).strip().splitlines()[0]
        except (OSError, subprocess.CalledProcessError, IndexError):
            driver_version = None
    return {
        "accelerator": "cuda" if cuda_available else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"),
        "device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "device_capability": list(torch.cuda.get_device_capability(0)) if cuda_available else None,
        "driver_version": driver_version,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        # No runner calls torch.use_deterministic_algorithms, so full determinism is not
        # in force and fixed-seed runs are not bit-reproducible.
        "deterministic_algorithms_requested": False,
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
