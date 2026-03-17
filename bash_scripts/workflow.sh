#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY_DIR="$ROOT_DIR/python_scripts"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash workflow.sh <pdb_name> [chain_lengths_csv] [frame_range|auto] [jobs] [model_path] [cg_only] [load_full_model]"
  echo "Example: bash workflow.sh IgE 546,215,546,215,121 auto 12 weights/backbone_model_refined26_multi_input.h5 0 0"
  echo "CG-only example: bash workflow.sh IgE 546,215,546,215,121 auto 12 weights/backbone_model_refined26_multi_input.h5 1 0"
  exit 1
fi

PDB_NAME="$1"
CHAIN_LENGTHS="${2:-546,215,546,215,121}"
FRAME_RANGE="${3:-auto}"
JOBS="${4:-8}"
MODEL_PATH="${5:-$ROOT_DIR/weights/backbone_model_refined26_multi_input.h5}"
CG_ONLY="${6:-0}"
LOAD_FULL_MODEL="${7:-0}"

if [[ "$LOAD_FULL_MODEL" != "0" && "$LOAD_FULL_MODEL" != "1" ]]; then
  echo "load_full_model must be 0 or 1"
  exit 1
fi

if [[ ! -f "$MODEL_PATH" && -f "$ROOT_DIR/weights/$MODEL_PATH" ]]; then
  MODEL_PATH="$ROOT_DIR/weights/$MODEL_PATH"
fi

resolve_frames() {
  local spec="$1"
  if [[ "$spec" == "auto" ]]; then
    if [[ -f cluster_ALL_CG.npy ]]; then
      echo "ALL"
      return
    fi
    local numeric
    numeric="$(ls cluster_*_CG.npy 2>/dev/null | sed -nE 's/cluster_([0-9]+)_CG\.npy/\1/p' | sort -n | uniq)"
    if [[ -n "${numeric// }" ]]; then
      echo "$numeric"
    fi
  elif [[ "$spec" == *"-"* ]]; then
    local start="${spec%-*}"
    local end="${spec#*-}"
    seq "$start" "$end"
  else
    echo "$spec" | tr ',' '\n'
  fi
}

FRAMES="$(resolve_frames "$FRAME_RANGE")"
if [[ -z "${FRAMES// }" ]]; then
  echo "No frames resolved. If using 'auto', make sure cluster_<idx>_CG.npy files exist."
  exit 1
fi
FRAME_ARG="$(echo "$FRAMES" | paste -sd, -)"

echo "[0/4] Cleanup previous intermediates for ${PDB_NAME}"
rm -f \
  train_feat_B*_"$PDB_NAME"_chain*.npy \
  train_LAB_B*_"$PDB_NAME"_chain*.npy \
  custom_min_B*_"$PDB_NAME"_chain*.npy \
  custom_range_B*_"$PDB_NAME"_chain*.npy \
  RAMAPROIR_yhat_frame_*_chain_*.npy \
  pred_"$PDB_NAME"_frame*_chain*_frames*.npy \
  actual_"$PDB_NAME"_frame*_chain*_frames*.npy \
  rmsd_"$PDB_NAME"_frame*_chain*_frames*.npy

echo "[1/4] Build train_feat/train_LAB/custom_* files (parallel)"
if [[ "$CG_ONLY" == "1" ]]; then
  echo "$FRAMES" | xargs -I{} -P "$JOBS" python3 "$PY_DIR/AA_subset_ml3.py" "$PDB_NAME" {} --chain-lengths "$CHAIN_LENGTHS" --cg-only
else
  echo "$FRAMES" | xargs -I{} -P "$JOBS" python3 "$PY_DIR/AA_subset_ml3.py" "$PDB_NAME" {} --chain-lengths "$CHAIN_LENGTHS"
fi

# If ALL mode was used, derive numeric frame indices from generated train_feat files.
if [[ "$FRAME_ARG" == "ALL" ]]; then
  FRAME_ARG="$(ls train_feat_B*_"$PDB_NAME"_chain1.npy 2>/dev/null | sed -nE 's/.*train_feat_B([0-9]+)_.*$/\1/p' | sort -n | uniq | paste -sd, -)"
  if [[ -z "${FRAME_ARG// }" ]]; then
    echo "Could not infer numeric frame list from train_feat outputs after ALL mode."
    exit 1
  fi
fi

echo "[2/4] Optional model inference"
if [[ -f "$PY_DIR/evaluate3.py" ]]; then
  # Build command incrementally to avoid empty-array expansion issues on bash 3.2 + set -u.
  eval_cmd=(python3 "$PY_DIR/evaluate3.py" "$PDB_NAME" ALL "$MODEL_PATH" --batch-size 5000)

  if [[ "$CG_ONLY" == "1" ]]; then
    eval_cmd+=(--cg-only)
  fi
  if [[ "$LOAD_FULL_MODEL" == "1" ]]; then
    eval_cmd+=(--load-full-model)
  fi

  time "${eval_cmd[@]}"
else
  echo "evaluate3.py not found; skipping inference step."
  echo "Expected prediction files from your existing evaluate pipeline."
fi

echo "[3/4] Reverse scaling (parallel)"
if [[ "$CG_ONLY" == "1" ]]; then
  python3 "$PY_DIR/backone_scripts/run_reverse_scaling_batch.py" \
    --pdb-name "$PDB_NAME" \
    --chain-lengths "$CHAIN_LENGTHS" \
    --frames "$FRAME_ARG" \
    --jobs "$JOBS" \
    --cg-only
else
  python3 "$PY_DIR/backone_scripts/run_reverse_scaling_batch.py" \
    --pdb-name "$PDB_NAME" \
    --chain-lengths "$CHAIN_LENGTHS" \
    --frames "$FRAME_ARG" \
    --jobs "$JOBS"
fi

echo "[4/4] Reattach chains and compute RMSD"
BACKBONE_PREFIX="backbone_${PDB_NAME}"
if [[ "$CG_ONLY" == "1" ]]; then
  python3 "$PY_DIR/reAttachment.py" --pdb-name "$PDB_NAME" --output-prefix "$BACKBONE_PREFIX" --pred-only
else
  python3 "$PY_DIR/reAttachment.py" --pdb-name "$PDB_NAME" --output-prefix "$BACKBONE_PREFIX"
fi

echo "Done:"
echo "  ${BACKBONE_PREFIX}_prediction.npy"
if [[ "$CG_ONLY" != "1" ]]; then
  echo "  ${BACKBONE_PREFIX}_actual.npy"
  echo "  ${BACKBONE_PREFIX}_rmsd.npy"
fi
