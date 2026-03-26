#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SC_PY_DIR="$ROOT_DIR/python_scripts/sidechain"

if [[ $# -lt 2 ]]; then
  echo "Usage: bash sidechain_workflow.sh <pdb_name> <chain_lengths_csv> [sc_cluster_id] [cg_only]"
  echo "Example: bash sidechain_workflow.sh IgE 546,215,546,215,121 2 0"
  exit 1
fi

PDB_NAME="$1"
CHAIN_LENGTHS="$2"
SC_CLUSTER_ID="${3:-2}"
CG_ONLY="${4:-0}"
SC_MODEL="${SIDECHAIN_MODEL_PATH:-$ROOT_DIR/weights/ckpt_epoch_SINGLE_BASE_37_147.keras}"
SC_DEFAULT_BUFFER="${SIDECHAIN_DEFAULT_BUFFER:-}"
SC_AROMATIC_BUFFER="${SIDECHAIN_AROMATIC_BUFFER:-}"

CG_SC_FILE="cluster_${SC_CLUSTER_ID}_CG_SC.npy"
CG_BB_FILE="cluster_${SC_CLUSTER_ID}_CG_BB.npy"
FULL_SEQ_FILE="sequence_${PDB_NAME}_FULL.txt"
SC_SEQ_FILE="sequence_${PDB_NAME}.txt"
AA_SC_FILE="cluster_${SC_CLUSTER_ID}_SC.npy"

require_file() {
  local f="$1"
  [[ -f "$f" ]] || { echo "Missing required file: $f" >&2; exit 1; }
}

require_file "$CG_SC_FILE"
require_file "$CG_BB_FILE"
require_file "$FULL_SEQ_FILE"
require_file "$SC_SEQ_FILE"

SEGMENT_STARTS="$(python3 - "$CHAIN_LENGTHS" <<'PY'
import sys
vals=[int(x.strip()) for x in sys.argv[1].split(',') if x.strip()]
starts=[0]
running=0
for v in vals[:-1]:
    running += v
    starts.append(running)
print(','.join(str(x) for x in starts))
PY
)"

echo "[sidechain] Stage 1/4: local frame normalization (CG)"
lf_cmd=(python3 "$SC_PY_DIR/local_frames.py" "$CG_SC_FILE" "$PDB_NAME" "$CG_BB_FILE" "$FULL_SEQ_FILE" "$SEGMENT_STARTS")
if [[ -n "$SC_DEFAULT_BUFFER" ]]; then
  lf_cmd+=(--default-buffer "$SC_DEFAULT_BUFFER")
fi
if [[ -n "$SC_AROMATIC_BUFFER" ]]; then
  lf_cmd+=(--aromatic-buffer "$SC_AROMATIC_BUFFER")
fi
# denorm.py requires R_localFrame_* in both full and CG-only prediction paths.
lf_cmd+=(--save-local-frames)
"${lf_cmd[@]}"

require_file "$SC_MODEL"
require_file "R_localFrame_${PDB_NAME}_cluster${SC_CLUSTER_ID}.npy"

if [[ "$CG_ONLY" == "1" ]]; then
  echo "[sidechain] Stage 2/4: CG-only mode; skipping AA target normalization."
else
  require_file "$AA_SC_FILE"
  echo "[sidechain] Stage 2/4: local frame normalization (AA targets)"
  python3 "$SC_PY_DIR/local_frames_AA.py" "$AA_SC_FILE" "$PDB_NAME" "$CG_SC_FILE" "$SC_CLUSTER_ID"
fi

if [[ "$CG_ONLY" == "1" ]]; then
  echo "[sidechain] Stage 3/4: model prediction (CG-only, no-eval)"
else
  echo "[sidechain] Stage 3/4: model evaluation + prediction"
fi
rm -f expert_Yhat_reshaped.npy reshaped_YHAT.npy "expert_${SC_CLUSTER_ID}_Yhat.npy" expert_Yhat.npy
python3 "$SC_PY_DIR/new_masking_test_train_split_localFrame.py" "$PDB_NAME" "$SC_CLUSTER_ID"
MASK_NPY_FILE="masking_input_${SC_CLUSTER_ID}.npy"
SC_MASK_FILE="input_Masking_testing_${PDB_NAME}_localFrame.npz"
require_file "$MASK_NPY_FILE"
require_file "$SC_MASK_FILE"
if [[ "$CG_ONLY" == "1" ]]; then
  python3 "$SC_PY_DIR/load_model.py" \
    --x "cluster_${SC_CLUSTER_ID}_SC_CG_RBF_localFrameFeatures.npy" \
    --model "$SC_MODEL" \
    --out "expert_${SC_CLUSTER_ID}_Yhat.npy" \
    --no-eval
else
  python3 "$SC_PY_DIR/load_model.py" \
    "cluster_${SC_CLUSTER_ID}_SC_CG_RBF_localFrameFeatures.npy" \
    "cluster_PD_${SC_CLUSTER_ID}_SC_LocalFrame.npy" \
    "$SC_MASK_FILE" \
    "$SC_CLUSTER_ID" \
    --model "$SC_MODEL"
fi

PRED_FILE=""
for f in expert_Yhat_reshaped.npy reshaped_YHAT.npy "expert_${SC_CLUSTER_ID}_Yhat.npy" expert_Yhat.npy; do
  if [[ -f "$f" ]]; then
    PRED_FILE="$f"
    break
  fi
done

if [[ -z "$PRED_FILE" ]]; then
  echo "Could not locate side-chain prediction file after load_model stage." >&2
  exit 1
fi

if [[ "$CG_ONLY" == "1" ]]; then
  # In CG-only mode, load_model emits bead-wise predictions (N*groups, 15).
  # Denorm expects frame-wise padded predictions (N, 15*groups), so reshape here.
  RESHAPED_PRED="$(python3 - "$PRED_FILE" "$MASK_NPY_FILE" <<'PY'
import os
import sys
import numpy as np

pred_file = sys.argv[1]
mask_file = sys.argv[2]

pred = np.load(pred_file)
mask = np.load(mask_file)

if mask.ndim != 2:
    raise SystemExit(f"mask must be 2D, got {mask.shape}")
if mask.shape[1] % 15 != 0:
    raise SystemExit(f"mask width must be divisible by 15, got {mask.shape[1]}")

n_frames = int(mask.shape[0])
groups = int(mask.shape[1] // 15)
expected_rows = n_frames * groups

if pred.ndim == 2 and pred.shape == (n_frames, mask.shape[1]):
    # Already frame-wise.
    print(pred_file)
    raise SystemExit(0)

if pred.ndim != 2 or pred.shape[1] != 15:
    raise SystemExit(f"unexpected prediction shape for CG-only reshape: {pred.shape}")
if pred.shape[0] != expected_rows:
    raise SystemExit(
        f"cannot reshape pred {pred.shape} using mask {mask.shape}; "
        f"expected rows={expected_rows}"
    )

reshaped = pred.reshape(n_frames, -1).astype(np.float32)
out_file = "expert_Yhat_reshaped.npy"
np.save(out_file, reshaped)
print(out_file)
PY
)"
  PRED_FILE="$RESHAPED_PRED"
fi

echo "[sidechain] Stage 4/4: denormalize"
DENORM_DEFAULT_FLAGS=(
  --no-bond-fix
  --ring-fix
  --ring-template-source ccd
  --ring-template-cache-dir .ring_template_cache
  --ring-fix-alpha 0.5
)

if [[ -f expert_filter_keep_group_idx.npy ]]; then
  python3 "$SC_PY_DIR/denorm.py" "$PRED_FILE" "$SC_SEQ_FILE" "$CG_SC_FILE" "masking_input_${SC_CLUSTER_ID}.npy" "$PDB_NAME" "$SC_CLUSTER_ID" "sidechain_${PDB_NAME}_prediction.npy" expert_filter_keep_group_idx.npy "${DENORM_DEFAULT_FLAGS[@]}"
else
  python3 "$SC_PY_DIR/denorm.py" "$PRED_FILE" "$SC_SEQ_FILE" "$CG_SC_FILE" "masking_input_${SC_CLUSTER_ID}.npy" "$PDB_NAME" "$SC_CLUSTER_ID" "sidechain_${PDB_NAME}_prediction.npy" "${DENORM_DEFAULT_FLAGS[@]}"
fi

echo "[sidechain] Done: sidechain_${PDB_NAME}_prediction.npy"
