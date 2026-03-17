#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
PYTHON_DIR="$ROOT_DIR/python_scripts"
BASH_DIR="$ROOT_DIR/bash_scripts"

resolve_existing_dir() {
  local raw="$1"
  local label="$2"
  local expanded="${raw/#\~/$HOME}"

  if [[ -d "$expanded" ]]; then
    (cd "$expanded" && pwd -P)
    return 0
  fi
  if [[ -d "$ROOT_DIR/$expanded" ]]; then
    (cd "$ROOT_DIR/$expanded" && pwd -P)
    return 0
  fi

  echo "${label} does not exist: $raw" >&2
  return 1
}

resolve_existing_file() {
  local raw="$1"
  local label="$2"
  local expanded="${raw/#\~/$HOME}"

  if [[ -f "$expanded" ]]; then
    (cd "$(dirname "$expanded")" && printf "%s/%s\n" "$(pwd -P)" "$(basename "$expanded")")
    return 0
  fi
  if [[ -f "$ROOT_DIR/$expanded" ]]; then
    (cd "$(dirname "$ROOT_DIR/$expanded")" && printf "%s/%s\n" "$(pwd -P)" "$(basename "$expanded")")
    return 0
  fi
  if [[ -f "$ROOT_DIR/weights/$expanded" ]]; then
    (cd "$ROOT_DIR/weights" && printf "%s/%s\n" "$(pwd -P)" "$(basename "$expanded")")
    return 0
  fi

  echo "${label} does not exist: $raw" >&2
  return 1
}

usage() {
  cat <<USAGE
Usage:
  bash BackMapNet.sh --pdb-name <name> [options]

Core options:
  --pdb-name <name>             Required PDB tag (e.g., IgE)
  --chain-lengths <csv|auto>    Residue chain lengths (default: auto)
  --frame-range <spec|auto>     Frame spec: auto, 0-399, or 0,1,2 (default: auto)
  --jobs <n>                    Parallel workers (default: 8)
  --cg-only <0|1>               1 = CG-only mode (default: 1)
  --fresh-start <0|1>           1 = rebuild cluster files from scratch (default: 1)
  --cg-pdb-dir <dir>            Directory with CG_frame_<idx>.pdb files (default: .)
                                Relative paths are resolved from current directory, then repo root.
  --aa-pdb-dir <dir>            Directory with frame_<idx>.pdb files (auto-switches cg_only=0)

Backbone model:
  --model-path <path>           Backbone model file path
                                (default: weights/backbone_model_refined26_multi_input.h5)
  --load-full-model <0|1>       1 = load a full saved model instead of weights (default: 0)

Side-chain options:
  --run-sidechain <0|1>         Enable side-chain pipeline (default: 1)
  --aa-sc-pdb-dir <dir>         Directory with frame_<idx>_SC.pdb files
                                Required when --aa-pdb-dir is provided and side-chain is enabled.
  --sidechain-cluster-id <id>   Side-chain cluster id token (default: 2)

PDB export options:
  --write-pdb <0|1>             Write PDB frame(s) from reconstructed combined array (default: 0)
  --pdb-output-dir <dir>        Output directory for PDB files (default: pdb_frames_<pdb_name>)
  --pdb-frame-spec <spec>       all, single index (5), range (0-99), or list (0,5,10) (default: all)
  --pdb-filename-template <t>   Filename template, e.g. frame_BackMapNet_V3_{frame}.pdb

  -h, --help                    Show this help

Examples:
  # CG-only backbone + side-chain CG feature generation
  bash BackMapNet.sh --pdb-name IgE --cg-pdb-dir /data/cg

  # Full backbone + full side-chain
  bash BackMapNet.sh \
    --pdb-name IgE \
    --cg-pdb-dir /data/cg \
    --aa-pdb-dir /data/aa_bb \
    --aa-sc-pdb-dir /data/aa_sc \
    --model-path /models/backbone.weights.h5

Legacy positional mode (still supported):
  bash BackMapNet.sh <pdb_name> [chain_lengths_csv|auto] [frame_range|auto] [jobs] [model_path] [cg_only] [fresh_start] [cg_pdb_dir]
USAGE
}

PDB_NAME=""
CHAIN_LENGTHS="auto"
FRAME_RANGE="auto"
JOBS="8"
MODEL_PATH="$ROOT_DIR/weights/backbone_model_refined26_multi_input.h5"
LOAD_FULL_MODEL="0"
CG_ONLY="1"
FRESH_START="1"
CG_PDB_DIR="."
AA_PDB_DIR=""

RUN_SIDECHAIN="1"
AA_SC_PDB_DIR=""
SC_CLUSTER_ID="2"
WRITE_PDB="0"
PDB_OUTPUT_DIR=""
PDB_FRAME_SPEC="all"
PDB_FILENAME_TEMPLATE="frame_BackMapNet_V3_{frame}.pdb"

# Backward compatibility: positional invocation.
if [[ $# -gt 0 && "$1" != -* ]]; then
  PDB_NAME="$1"
  CHAIN_LENGTHS="${2:-auto}"
  FRAME_RANGE="${3:-auto}"
  JOBS="${4:-8}"
  MODEL_PATH="${5:-$MODEL_PATH}"
  CG_ONLY="${6:-1}"
  FRESH_START="${7:-1}"
  CG_PDB_DIR="${8:-.}"
else
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --pdb-name)
        PDB_NAME="$2"; shift 2 ;;
      --chain-lengths)
        CHAIN_LENGTHS="$2"; shift 2 ;;
      --frame-range)
        FRAME_RANGE="$2"; shift 2 ;;
      --jobs)
        JOBS="$2"; shift 2 ;;
      --model-path)
        MODEL_PATH="$2"; shift 2 ;;
      --load-full-model)
        LOAD_FULL_MODEL="$2"; shift 2 ;;
      --cg-only)
        CG_ONLY="$2"; shift 2 ;;
      --fresh-start)
        FRESH_START="$2"; shift 2 ;;
      --cg-pdb-dir)
        CG_PDB_DIR="$2"; shift 2 ;;
      --aa-pdb-dir)
        AA_PDB_DIR="$2"; shift 2 ;;
      --run-sidechain)
        RUN_SIDECHAIN="$2"; shift 2 ;;
      --aa-sc-pdb-dir)
        AA_SC_PDB_DIR="$2"; shift 2 ;;
      --sidechain-cluster-id)
        SC_CLUSTER_ID="$2"; shift 2 ;;
      --write-pdb)
        WRITE_PDB="$2"; shift 2 ;;
      --pdb-output-dir)
        PDB_OUTPUT_DIR="$2"; shift 2 ;;
      --pdb-frame-spec)
        PDB_FRAME_SPEC="$2"; shift 2 ;;
      --pdb-filename-template)
        PDB_FILENAME_TEMPLATE="$2"; shift 2 ;;
      -h|--help)
        usage; exit 0 ;;
      *)
        echo "Unknown option: $1"
        usage
        exit 1 ;;
    esac
  done
fi

if [[ -z "$PDB_NAME" ]]; then
  echo "Missing required --pdb-name"
  usage
  exit 1
fi

if [[ "$CG_ONLY" != "0" && "$CG_ONLY" != "1" ]]; then
  echo "cg_only must be 0 or 1"
  exit 1
fi

if [[ "$FRESH_START" != "0" && "$FRESH_START" != "1" ]]; then
  echo "fresh_start must be 0 or 1"
  exit 1
fi

if [[ "$RUN_SIDECHAIN" != "0" && "$RUN_SIDECHAIN" != "1" ]]; then
  echo "run-sidechain must be 0 or 1"
  exit 1
fi

if [[ "$LOAD_FULL_MODEL" != "0" && "$LOAD_FULL_MODEL" != "1" ]]; then
  echo "load-full-model must be 0 or 1"
  exit 1
fi

if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || [[ "$JOBS" -lt 1 ]]; then
  echo "jobs must be a positive integer"
  exit 1
fi

if ! [[ "$SC_CLUSTER_ID" =~ ^[0-9]+$ ]] || [[ "$SC_CLUSTER_ID" -lt 0 ]]; then
  echo "sidechain-cluster-id must be a non-negative integer"
  exit 1
fi

if [[ "$WRITE_PDB" != "0" && "$WRITE_PDB" != "1" ]]; then
  echo "write-pdb must be 0 or 1"
  exit 1
fi

if ! CG_PDB_DIR="$(resolve_existing_dir "$CG_PDB_DIR" "cg_pdb_dir")"; then
  exit 1
fi

if [[ -n "$AA_PDB_DIR" ]]; then
  if ! AA_PDB_DIR="$(resolve_existing_dir "$AA_PDB_DIR" "aa_pdb_dir")"; then
    exit 1
  fi
  if [[ "$CG_ONLY" != "0" ]]; then
    echo "[BackMapNet] --aa-pdb-dir provided; switching to cg_only=0"
  fi
  CG_ONLY="0"
fi

if [[ "$CG_ONLY" == "0" && -z "$AA_PDB_DIR" ]]; then
  echo "Full mode requires --aa-pdb-dir (directory containing frame_<idx>.pdb)."
  exit 1
fi

if [[ "$RUN_SIDECHAIN" == "1" && "$CG_ONLY" == "0" && -z "$AA_SC_PDB_DIR" ]]; then
  echo "Side-chain full mode requires --aa-sc-pdb-dir when --aa-pdb-dir is provided."
  exit 1
fi

if [[ -n "$AA_SC_PDB_DIR" ]]; then
  if ! AA_SC_PDB_DIR="$(resolve_existing_dir "$AA_SC_PDB_DIR" "aa_sc_pdb_dir")"; then
    exit 1
  fi
fi

# Resolve model path from current directory, repo root, or repo/weights.
if ! MODEL_PATH="$(resolve_existing_file "$MODEL_PATH" "model_path")"; then
  exit 1
fi

resolve_first_frame() {
  local spec="$1"
  if [[ "$spec" == "auto" ]]; then
    find "$CG_PDB_DIR" -maxdepth 1 -type f -name 'CG_frame_*.pdb' \
      | sed -E 's#.*/CG_frame_([0-9]+)\.pdb#\1#' | sort -n | sed -n '1p'
  elif [[ "$spec" == *"-"* ]]; then
    local start="${spec%-*}"
    echo "$start"
  else
    echo "$spec" | tr ',' '\n' | sed '/^$/d' | sort -n | sed -n '1p'
  fi
}

if [[ "$CHAIN_LENGTHS" == "auto" ]]; then
  FIRST_FRAME="$(resolve_first_frame "$FRAME_RANGE")"
  if [[ -z "${FIRST_FRAME// }" ]]; then
    echo "Could not resolve a frame for chain-length auto inference."
    echo "Expected files like ${CG_PDB_DIR}/CG_frame_<idx>.pdb"
    exit 1
  fi

  REF_CG_PDB="${CG_PDB_DIR}/CG_frame_${FIRST_FRAME}.pdb"
  REF_AA_PDB="${AA_PDB_DIR}/frame_${FIRST_FRAME}.pdb"

  if [[ "$CG_ONLY" == "1" ]]; then
    REF_PDB="$REF_CG_PDB"
  else
    REF_PDB="$REF_AA_PDB"
  fi

  if [[ ! -f "$REF_PDB" ]]; then
    echo "Chain-length auto inference failed: missing reference PDB $REF_PDB"
    exit 1
  fi

  CHAIN_LENGTHS="$(python3 - "$PYTHON_DIR" "$REF_PDB" <<'PY'
import sys

python_dir = sys.argv[1]
pdb_path = sys.argv[2]
sys.path.insert(0, python_dir)

from AA_subset_ml3 import infer_chain_lengths_from_pdb

lengths = infer_chain_lengths_from_pdb(pdb_path)
print(",".join(str(x) for x in lengths))
PY
)"

  if [[ -z "${CHAIN_LENGTHS// }" ]]; then
    echo "Chain-length auto inference returned empty result for $REF_PDB"
    exit 1
  fi

  echo "[BackMapNet] Auto-inferred chain lengths from ${REF_PDB}: ${CHAIN_LENGTHS}"
fi

echo "[BackMapNet] Step 1/5: PDB -> cluster arrays (backbone + optional side-chain)"
bash "$BASH_DIR/BB_val_arr.sh" \
  --pdb-name "$PDB_NAME" \
  --frame-range "$FRAME_RANGE" \
  --cg-only "$CG_ONLY" \
  --fresh-start "$FRESH_START" \
  --cg-pdb-dir "$CG_PDB_DIR" \
  --aa-pdb-dir "$AA_PDB_DIR" \
  --aa-sc-pdb-dir "$AA_SC_PDB_DIR" \
  --run-sidechain "$RUN_SIDECHAIN" \
  --sc-cluster-id "$SC_CLUSTER_ID" \
  --jobs "$JOBS" \
  --python-dir "$PYTHON_DIR" \
  --python-sidechain-dir "$PYTHON_DIR/sidechain"

echo "[BackMapNet] Step 2/5: backbone pipeline"
bash "$BASH_DIR/workflow.sh" "$PDB_NAME" "$CHAIN_LENGTHS" auto "$JOBS" "$MODEL_PATH" "$CG_ONLY" "$LOAD_FULL_MODEL"

if [[ "$RUN_SIDECHAIN" == "1" ]]; then
  echo "[BackMapNet] Step 3/5: side-chain pipeline"
  bash "$BASH_DIR/sidechain_workflow.sh" "$PDB_NAME" "$CHAIN_LENGTHS" "$SC_CLUSTER_ID" "$CG_ONLY"
else
  echo "[BackMapNet] Step 3/5: side-chain pipeline skipped (--run-sidechain 0)"
fi

if [[ "$RUN_SIDECHAIN" == "1" ]]; then
  RECON_SCRIPT="$PYTHON_DIR/reconstruct_arr.py"
  if [[ ! -f "$RECON_SCRIPT" ]]; then
    echo "Missing reconstruct script: $RECON_SCRIPT"
    exit 1
  fi

  if [[ "$CG_ONLY" == "1" ]]; then
    RECON_MODE="cg-only"
    RECON_OUT="combined_${PDB_NAME}_prediction.npy"
  else
    RECON_MODE="full"
    RECON_OUT="combined_${PDB_NAME}_actual.npy"
  fi

  echo "[BackMapNet] Step 4/5: reconstruct backbone + side-chain arrays (${RECON_MODE})"
  python3 "$RECON_SCRIPT" \
    --mode "$RECON_MODE" \
    --pdb-name "$PDB_NAME" \
    --sc-cluster-id "$SC_CLUSTER_ID"
  echo "[BackMapNet] Reconstructed array: $RECON_OUT"
else
  echo "[BackMapNet] Step 4/5: reconstruction skipped (requires --run-sidechain 1)"
fi

if [[ "$WRITE_PDB" == "1" ]]; then
  if [[ "$RUN_SIDECHAIN" != "1" ]]; then
    echo "[BackMapNet] Step 5/5: PDB export skipped (requires --run-sidechain 1 for combined array)"
  else
    MAKEPDB_SCRIPT="$PYTHON_DIR/MakePDB_temp.py"
    if [[ ! -f "$MAKEPDB_SCRIPT" ]]; then
      echo "Missing PDB writer script: $MAKEPDB_SCRIPT"
      exit 1
    fi

    if [[ "$CG_ONLY" == "1" ]]; then
      COMBINED_FILE="combined_${PDB_NAME}_prediction.npy"
    else
      COMBINED_FILE="combined_${PDB_NAME}_actual.npy"
    fi

    if [[ ! -f "$COMBINED_FILE" ]]; then
      echo "Combined array for PDB export not found: $COMBINED_FILE"
      exit 1
    fi

    SEQ_FILE="sequence_${PDB_NAME}_FULL.txt"
    if [[ ! -f "$SEQ_FILE" ]]; then
      SEQ_FILE="sequence_${PDB_NAME}.txt"
    fi
    if [[ ! -f "$SEQ_FILE" ]]; then
      echo "Sequence file for PDB export not found: sequence_${PDB_NAME}_FULL.txt or sequence_${PDB_NAME}.txt"
      exit 1
    fi

    if [[ -z "$PDB_OUTPUT_DIR" ]]; then
      PDB_OUTPUT_DIR="pdb_frames_${PDB_NAME}"
    fi

    echo "[BackMapNet] Step 5/5: PDB export from ${COMBINED_FILE}"
    python3 "$MAKEPDB_SCRIPT" \
      --coords-file "$COMBINED_FILE" \
      --sequence-file "$SEQ_FILE" \
      --output-dir "$PDB_OUTPUT_DIR" \
      --frame-spec "$PDB_FRAME_SPEC" \
      --filename-template "$PDB_FILENAME_TEMPLATE" \
      --chain-lengths "$CHAIN_LENGTHS"
    echo "[BackMapNet] PDB frames written to: $PDB_OUTPUT_DIR"
  fi
else
  echo "[BackMapNet] Step 5/5: PDB export skipped (--write-pdb 0)"
fi

echo "[BackMapNet] Completed."
