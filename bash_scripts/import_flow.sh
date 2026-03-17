#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY_DIR="$ROOT_DIR/python_scripts"

echo "pdb to arr"
bash "$SCRIPT_DIR/BB_val_arr.sh" --pdb-name IgE --cg-only 1 --cg-pdb-dir .

python3 "$PY_DIR/AA_subset_ml3.py" IgE ALL
