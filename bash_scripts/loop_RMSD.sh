#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY_DIR="$ROOT_DIR/python_scripts"

if [[ $# -lt 2 ]]; then
  echo "Usage: bash loop_RMSD.sh <chain_idx> <expected_length> [pdb_name] [frame_range] [jobs]"
  echo "Example: bash loop_RMSD.sh 1 6552 IgE 0-399 8"
  exit 1
fi

CHAIN="$1"
EXPECTED_LENGTH="$2"
PDB_NAME="${3:-IgE}"
FRAME_RANGE="${4:-0-399}"
JOBS="${5:-8}"

python3 "$PY_DIR/backone_scripts/run_reverse_scaling_batch.py" \
  --pdb-name "$PDB_NAME" \
  --chain-lengths "$((EXPECTED_LENGTH / 12))" \
  --frames "$FRAME_RANGE" \
  --jobs "$JOBS" \
  --pred-template "RAMAPROIR_yhat_frame_{frame}_chain_${CHAIN}.npy" \
  --actual-template "train_LAB_B{frame}_{pdb}_chain${CHAIN}.npy" \
  --custom-min-template "custom_min_B{frame}_{pdb}_chain${CHAIN}.npy" \
  --custom-range-template "custom_range_B{frame}_{pdb}_chain${CHAIN}.npy"
