#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY_DIR="$ROOT_DIR/python_scripts"

PDB_NAME="${1:-IgE}"
MODEL_PATH="${2:-$ROOT_DIR/weights/best_refined26_continued_model9_check_MinMax_Conv3D.keras}"

for chain in 1 2 3 4 5; do
  time python3 "$PY_DIR/evaluate3.py" "$PDB_NAME" "$chain" "$MODEL_PATH"
done
