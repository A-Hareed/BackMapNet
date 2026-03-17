#!/usr/bin/env bash
set -euo pipefail

# End-to-end CG->AA backmapping pipeline:
# 1) local frame normalization
# 2) optional AA normalization
# 3) masking split + model prediction
# 4) denormalization/unpadding

usage() {
  cat <<'EOF'
Usage:
  bash run_backmapping_pipeline.sh \
    --pdb <PDB_NAME> \
    --cg-sc <cluster_X_CG_SC.npy> \
    --cg-bb <cluster_X_CG_BB.npy> \
    --full-seq <sequence_PDB_full.txt> \
    [--cluster <CLUST_ID>] \
    [--segment-starts <csv>] \
    [--segment-starts-file <file.csv>] \
    [--aa-cluster <cluster_X_SC.npy>] \
    [--run-aa] \
    [--sequence <sequence_PDB.txt>] \
    [--pred-file <prediction.npy>] \
    [--weights <weights_file.h5>] \
    [--filter-residue-id <int>] \
    [--gate-col <int>] \
    [--no-filter-residue] \
    [--out <denorm_output.npy>] \
    [--python <python_bin>] \
    [--force]

Notes:
  - --run-aa enables stage 2 (local_frames_AA.py). It requires --aa-cluster.
  - stage 3 runs:
      python3 new_masking_test_train_split_localFrame.py <PDB>
      python3 run_model.py cluster_<CLUST>_SC_CG_RBF_localFrameFeatures.npy \
                           cluster_PD_<CLUST>_SC_LocalFrame.npy \
                           masking_input_<CLUST>.npy <CLUST>
  - If --pred-file is not provided, the script auto-detects one of:
      expert_Yhat_reshaped.npy, expert_17_Yhat.npy, expert_Yhat.npy
EOF
}

PYTHON_BIN="python3"
RUN_AA=0
FORCE=0
PDB=""
CLUST=""
CG_SC=""
CG_BB=""
FULL_SEQ=""
AA_CLUSTER=""
SEQUENCE_FILE=""
PRED_FILE=""
WEIGHTS_FILE="EXPERT_M24_best.weights.h5"
FILTER_RESIDUE_ID="0"
GATE_COL="37"
FILTER_ENABLED=1
KEEP_IDX_FILE="expert_filter_keep_group_idx.npy"
SEGMENT_STARTS=""
SEGMENT_STARTS_FILE=""
OUT_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pdb) PDB="${2:-}"; shift 2 ;;
    --cluster) CLUST="${2:-}"; shift 2 ;;
    --segment-starts) SEGMENT_STARTS="${2:-}"; shift 2 ;;
    --segment-starts-file) SEGMENT_STARTS_FILE="${2:-}"; shift 2 ;;
    --cg-sc) CG_SC="${2:-}"; shift 2 ;;
    --cg-bb) CG_BB="${2:-}"; shift 2 ;;
    --full-seq) FULL_SEQ="${2:-}"; shift 2 ;;
    --aa-cluster) AA_CLUSTER="${2:-}"; shift 2 ;;
    --run-aa) RUN_AA=1; shift 1 ;;
    --sequence) SEQUENCE_FILE="${2:-}"; shift 2 ;;
    --pred-file) PRED_FILE="${2:-}"; shift 2 ;;
    --weights) WEIGHTS_FILE="${2:-}"; shift 2 ;;
    --filter-residue-id) FILTER_RESIDUE_ID="${2:-}"; shift 2 ;;
    --gate-col) GATE_COL="${2:-}"; shift 2 ;;
    --no-filter-residue) FILTER_ENABLED=0; shift 1 ;;
    --out) OUT_FILE="${2:-}"; shift 2 ;;
    --python) PYTHON_BIN="${2:-}"; shift 2 ;;
    --force) FORCE=1; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -n "$PDB" ]] || { echo "Missing --pdb" >&2; usage; exit 2; }
[[ -n "$CG_SC" ]] || { echo "Missing --cg-sc" >&2; usage; exit 2; }
[[ -n "$CG_BB" ]] || { echo "Missing --cg-bb" >&2; usage; exit 2; }
[[ -n "$FULL_SEQ" ]] || { echo "Missing --full-seq" >&2; usage; exit 2; }

if [[ -z "$CLUST" ]]; then
  CLUST="1"
  echo "No --cluster provided. Defaulting cluster id to: $CLUST"
fi

if [[ "$RUN_AA" -eq 1 && -z "$AA_CLUSTER" ]]; then
  echo "--run-aa requires --aa-cluster" >&2
  exit 2
fi

if [[ -z "$SEQUENCE_FILE" ]]; then
  SEQUENCE_FILE="sequence_${PDB}.txt"
fi

if [[ -z "$OUT_FILE" ]]; then
  OUT_FILE="reversed_localframe_${CLUST}.npy"
fi

require_file() {
  local f="$1"
  [[ -f "$f" ]] || { echo "Missing file: $f" >&2; exit 1; }
}

run_step() {
  local name="$1"; shift
  echo ""
  echo "[$(date '+%H:%M:%S')] ${name}"
  echo "CMD: $*"
  "$@"
}

if [[ -n "$SEGMENT_STARTS_FILE" ]]; then
  require_file "$SEGMENT_STARTS_FILE"
  SEGMENT_STARTS="$(tr -d '[:space:]' < "$SEGMENT_STARTS_FILE")"
fi

# Performance-oriented defaults (safe no-op where unsupported)
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(command -v nproc >/dev/null 2>&1 && nproc || echo 1)}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"

require_file "$CG_SC"
require_file "$CG_BB"
require_file "$FULL_SEQ"
require_file "$SEQUENCE_FILE"
if [[ "$RUN_AA" -eq 1 ]]; then
  require_file "$AA_CLUSTER"
fi

FEATURE_FILE="cluster_${CLUST}_SC_CG_RBF_localFrameFeatures.npy"
MASK_FILE="masking_input_${CLUST}.npy"
DEFAULT_F1="cluster_${CLUST}_SC_CG_RBF_localFrameFeatures.npy"
DEFAULT_F2="cluster_PD_${CLUST}_SC_LocalFrame.npy"

echo "Pipeline configuration:"
echo "  PDB: $PDB"
echo "  CLUSTER: $CLUST"
echo "  CG_SC: $CG_SC"
echo "  CG_BB: $CG_BB"
echo "  WEIGHTS: $WEIGHTS_FILE"
if [[ -n "$SEGMENT_STARTS" ]]; then
  echo "  SEGMENT_STARTS: $SEGMENT_STARTS"
else
  echo "  SEGMENT_STARTS: default in local_frames.py"
fi
if [[ "$FILTER_ENABLED" -eq 1 ]]; then
  echo "  FILTER_RESIDUE_ID: $FILTER_RESIDUE_ID (enabled)"
  echo "  GATE_COL: $GATE_COL"
  echo "  KEEP_IDX_FILE: $KEEP_IDX_FILE"
else
  echo "  FILTER_RESIDUE_ID: disabled"
fi

if [[ "$FORCE" -eq 1 || ! -f "$FEATURE_FILE" ]]; then
  if [[ -n "$SEGMENT_STARTS" ]]; then
    run_step "Stage 1/4: local frame normalization (CG)" \
      "$PYTHON_BIN" local_frames.py "$CG_SC" "$PDB" "$CG_BB" "$FULL_SEQ" "$SEGMENT_STARTS"
  else
    run_step "Stage 1/4: local frame normalization (CG)" \
      "$PYTHON_BIN" local_frames.py "$CG_SC" "$PDB" "$CG_BB" "$FULL_SEQ"
  fi
else
  echo "Skipping Stage 1/4 (exists): $FEATURE_FILE"
fi

if [[ "$RUN_AA" -eq 1 ]]; then
  TARGET_FILE="cluster_PD_${CLUST}_SC_LocalFrame.npy"
  if [[ "$FORCE" -eq 1 || ! -f "$TARGET_FILE" || ! -f "$MASK_FILE" ]]; then
    run_step "Stage 2/4: local frame normalization (AA targets)" \
      "$PYTHON_BIN" local_frames_AA.py "$AA_CLUSTER" "$PDB" "$CG_SC" "$CLUST"
  else
    echo "Skipping Stage 2/4 (exists): $TARGET_FILE, $MASK_FILE"
  fi
else
  echo "Skipping Stage 2/4 (optional AA normalization disabled)"
fi

run_step "Stage 3a/4: masking split prep" \
  "$PYTHON_BIN" new_masking_test_train_split_localFrame.py "$PDB"

if [[ ! -f "$DEFAULT_F1" ]]; then
  echo "Warning: $DEFAULT_F1 not found."
fi
if [[ ! -f "$DEFAULT_F2" ]]; then
  echo "Warning: $DEFAULT_F2 not found."
fi
if [[ ! -f "$MASK_FILE" ]]; then
  echo "Warning: $MASK_FILE not found before run_model.py."
fi

run_step "Stage 3b/4: model prediction" \
  "$PYTHON_BIN" run_model.py "$DEFAULT_F1" "$DEFAULT_F2" "$MASK_FILE" "$CLUST" --weights "$WEIGHTS_FILE" \
  $([[ "$FILTER_ENABLED" -eq 1 ]] && printf -- "--filter-residue-id %s --gate-col %s" "$FILTER_RESIDUE_ID" "$GATE_COL")

if [[ -z "$PRED_FILE" ]]; then
  for c in expert_Yhat_reshaped.npy expert_17_Yhat.npy expert_Yhat.npy; do
    if [[ -f "$c" ]]; then
      PRED_FILE="$c"
      break
    fi
  done
fi

[[ -n "$PRED_FILE" ]] || { echo "Could not auto-detect prediction file. Pass --pred-file <file.npy>." >&2; exit 1; }
require_file "$PRED_FILE"
require_file "$MASK_FILE"

if [[ -f "$KEEP_IDX_FILE" ]]; then
  run_step "Stage 4/4: denormalize + unpad" \
    "$PYTHON_BIN" denorm.py "$PRED_FILE" "$SEQUENCE_FILE" "$CG_SC" "$MASK_FILE" "$PDB" "$CLUST" "$OUT_FILE" "$KEEP_IDX_FILE"
else
  run_step "Stage 4/4: denormalize + unpad" \
    "$PYTHON_BIN" denorm.py "$PRED_FILE" "$SEQUENCE_FILE" "$CG_SC" "$MASK_FILE" "$PDB" "$CLUST" "$OUT_FILE"
fi

echo ""
echo "Pipeline complete."
echo "Output: $OUT_FILE"
