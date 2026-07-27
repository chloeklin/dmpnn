from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "logs" / "regen_v1" / "r1" / "manifests" / "r1_pilot.manifest"
OUTPUT = ROOT / "analysis" / "model_diagnostics" / "_regen_v1_pilot_check.md"


def main() -> None:
    if not MANIFEST.is_file():
        raise SystemExit(f"Missing pilot manifest: {MANIFEST}")
    rows = []
    split_hashes = {}
    for line in MANIFEST.read_text().splitlines():
        runner, model, target, fold, seed, payload = line.split("\t")
        prediction = Path(payload.split("|", 1)[0])
        sidecar_path = prediction.with_suffix(".config.json")
        if not prediction.is_file() or not sidecar_path.is_file():
            raise SystemExit(f"Pilot incomplete: {prediction}")
        with np.load(prediction, allow_pickle=True) as archive:
            y_pred = archive["y_pred"].astype(float).ravel()
            y_final = archive["y_pred_final"].astype(float).ravel()
            split_hash = str(archive["split_indices_sha256"].item())
            if y_pred.shape != y_final.shape or y_pred.size == 0:
                raise AssertionError(f"Invalid prediction arrays: {prediction}")
        sidecar = json.loads(sidecar_path.read_text())
        if sidecar["epochs_actually_run"] <= 0 or sidecar["wall_time_seconds"] <= 60:
            raise AssertionError(f"Training not demonstrated: {prediction}")
        if sidecar["best_epoch"] is None or sidecar["best_val_loss"] is None:
            raise AssertionError(f"Missing selection provenance: {prediction}")
        for key in ("prediction_checkpoint", "final_prediction_checkpoint"):
            record = sidecar[key]
            if set(record) != {"path", "sha256"} or len(record["sha256"]) != 64:
                raise AssertionError(f"Invalid checkpoint record {key}: {prediction}")
        split_hashes.setdefault((model, target, fold), set()).add(split_hash)
        old_path = ROOT / "predictions" / "ea_ip_lomo" / prediction.name
        differs_from_old = True
        if old_path.is_file():
            with np.load(old_path, allow_pickle=True) as old:
                differs_from_old = not np.array_equal(y_pred, old["y_pred"].astype(float).ravel())
            if not differs_from_old:
                raise AssertionError(f"Regenerated artifact exactly matches predecessor: {prediction}")
        rows.append((model, target, fold, seed, sidecar["epochs_actually_run"], sidecar["best_epoch"], sidecar["best_val_loss"], sidecar["wall_time_seconds"], differs_from_old))
    for key, hashes in split_hashes.items():
        if len(hashes) != 1:
            raise AssertionError(f"Split hashes differ across available pilot seeds for {key}: {hashes}")
    lines = [
        "# Regen v1 R1 pilot verification",
        "",
        "**PASS.** All ten pilot jobs demonstrably trained, saved best and final predictions, recorded complete checkpoint provenance, used consistent frozen split hashes, and differ from their predecessors where an old artifact exists.",
        "",
        "| model | target | fold | seed | epochs | best_epoch | best_val_loss | wall_time_seconds | differs_from_old |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    lines.extend(f"| {model} | {target} | {fold} | {seed} | {epochs} | {best_epoch} | {loss:.8f} | {wall:.1f} | {differs} |" for model, target, fold, seed, epochs, best_epoch, loss, wall, differs in rows)
    OUTPUT.write_text("\n".join(lines) + "\n")
    print(f"PASS: wrote {OUTPUT}")


if __name__ == "__main__":
    main()
