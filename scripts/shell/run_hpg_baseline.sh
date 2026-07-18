#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/venv/bin/python}"
LOG_DIR="${ROOT_DIR}/logs/hpg_baseline"
mkdir -p "${LOG_DIR}"

run() {
  local label="$1"
  shift
  "${PYTHON_BIN}" "$@" 2>&1 | tee "${LOG_DIR}/${label}.log"
}

case "${MODE}" in
  smoke)
    run "smoke_pytest" -m pytest -o addopts='' "${ROOT_DIR}/tests/test_hpg_smoke.py" -q
    run "smoke_runner" "${ROOT_DIR}/scripts/python/run_hpg_generalization.py" \
      --split_types group_disjoint --folds 0 --targets "EA vs SHE (eV)" \
      --models hpg_sum,hpg_frac --seed 0 --epochs 1 --patience 1
    ;;
  full)
    for seed in 42 43 44; do
      run "full_s${seed}" "${ROOT_DIR}/scripts/python/run_hpg_generalization.py" --seed "${seed}"
    done
    ;;
  *)
    printf 'Usage: %s [smoke|full]\n' "$0" >&2
    exit 2
    ;;
esac
