#!/bin/bash
# Download Regen V1 prediction NPZs/configs and PBS task logs from NCI Gadi.
#
# Usage:
#   scripts/shell/download_regen_v1_artifacts.sh
#
# Override defaults via environment:
#   REMOTE_USER=hl4138 REMOTE_HOST=gadi.nci.org.au \
#     REMOTE_DIR=/scratch/um09/hl4138/dmpnn scripts/shell/download_regen_v1_artifacts.sh
#
# The script is safe to re-run: rsync only transfers changed files.
set -euo pipefail

REMOTE_USER="${REMOTE_USER:-hl4138}"
REMOTE_HOST="${REMOTE_HOST:-gadi.nci.org.au}"
REMOTE_DIR="${REMOTE_DIR:-/scratch/um09/hl4138/dmpnn}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Downloading regen_v1 artifacts from ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"

mkdir -p "${LOCAL_DIR}/predictions/regen_v1"
rsync -avP --exclude='*.tmp' \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/predictions/regen_v1/" \
  "${LOCAL_DIR}/predictions/regen_v1/"

for experiment in r1 r3; do
  remote_tasks="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/logs/regen_v1/${experiment}/tasks/"
  local_tasks="${LOCAL_DIR}/logs/regen_v1/${experiment}/tasks"
  echo "Downloading task logs for ${experiment} ..."
  if ssh -o BatchMode=yes "${REMOTE_USER}@${REMOTE_HOST}" test -d "${REMOTE_DIR}/logs/regen_v1/${experiment}/tasks"; then
    mkdir -p "${local_tasks}"
    rsync -avP "${remote_tasks}" "${local_tasks}/"
  else
    echo "Remote task log directory does not exist yet: ${remote_tasks}"
  fi
done

echo "Download complete."
echo "Confirm frozen-split assertion execution by grepping task logs, e.g.:"
echo "  grep -R 'Frozen monomer_b_heldout split assertions passed\|B-identity leakage\|differs from frozen metadata\|frozen_protocol' ${LOCAL_DIR}/logs/regen_v1/r3/tasks/"
