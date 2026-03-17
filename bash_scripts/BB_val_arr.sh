#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<USAGE
Usage:
  bash BB_val_arr.sh --pdb-name <name> [options]

Options:
  --pdb-name <name>         Required PDB tag (e.g., IgE)
  --frame-range <spec>      auto | 0-399 | 0,1,2 (default: auto)
  --cg-only <0|1>           1 = only CG conversion (default: 0)
  --fresh-start <0|1>       1 = rebuild output clusters from scratch (default: 0)
  --cg-pdb-dir <dir>        Dir with CG_frame_<idx>.pdb (default: .)
  --aa-pdb-dir <dir>        Dir with frame_<idx>.pdb (required if cg_only=0)
  --aa-sc-pdb-dir <dir>     Dir with frame_<idx>_SC.pdb (required if cg_only=0 and sidechain enabled)
  --run-sidechain <0|1>     Build side-chain arrays too (default: 1)
  --sc-cluster-id <id>      Side-chain cluster id token (default: 2)
  --jobs <n>                Parallel workers (default: 1)
  --python-dir <dir>        Directory with backbone conversion scripts
  --python-sidechain-dir <dir> Directory with side-chain conversion scripts
  -h, --help                Show help

Legacy positional mode (still supported):
  bash BB_val_arr.sh <pdb_name> [frame_range|auto] [cg_only] [fresh_start] [cg_pdb_dir] [jobs]
USAGE
}

PDB_NAME=""
FRAME_SPEC="auto"
CG_ONLY="0"
FRESH_START="0"
CG_PDB_DIR="."
AA_PDB_DIR=""
AA_SC_PDB_DIR=""
RUN_SIDECHAIN="1"
SC_CLUSTER_ID="2"
JOBS="1"
PYTHON_DIR="$ROOT_DIR/python_scripts"
PY_SC_DIR="$ROOT_DIR/python_scripts/sidechain"

# Backward compatibility: positional invocation
if [[ $# -gt 0 && "$1" != -* ]]; then
  PDB_NAME="$1"
  FRAME_SPEC="${2:-auto}"
  CG_ONLY="${3:-0}"
  FRESH_START="${4:-0}"
  CG_PDB_DIR="${5:-.}"
  JOBS="${6:-1}"
else
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --pdb-name)
        PDB_NAME="$2"; shift 2 ;;
      --frame-range)
        FRAME_SPEC="$2"; shift 2 ;;
      --cg-only)
        CG_ONLY="$2"; shift 2 ;;
      --fresh-start)
        FRESH_START="$2"; shift 2 ;;
      --cg-pdb-dir)
        CG_PDB_DIR="$2"; shift 2 ;;
      --aa-pdb-dir)
        AA_PDB_DIR="$2"; shift 2 ;;
      --aa-sc-pdb-dir)
        AA_SC_PDB_DIR="$2"; shift 2 ;;
      --run-sidechain)
        RUN_SIDECHAIN="$2"; shift 2 ;;
      --sc-cluster-id)
        SC_CLUSTER_ID="$2"; shift 2 ;;
      --jobs)
        JOBS="$2"; shift 2 ;;
      --python-dir)
        PYTHON_DIR="$2"; shift 2 ;;
      --python-sidechain-dir)
        PY_SC_DIR="$2"; shift 2 ;;
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

if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || [[ "$JOBS" -lt 1 ]]; then
  echo "jobs must be a positive integer"
  exit 1
fi

if ! [[ "$SC_CLUSTER_ID" =~ ^[0-9]+$ ]] || [[ "$SC_CLUSTER_ID" -lt 0 ]]; then
  echo "sc-cluster-id must be a non-negative integer"
  exit 1
fi

if [[ ! -d "$CG_PDB_DIR" ]]; then
  echo "cg_pdb_dir does not exist: $CG_PDB_DIR"
  exit 1
fi

if [[ "$CG_ONLY" == "0" ]]; then
  if [[ -z "$AA_PDB_DIR" ]]; then
    echo "Full mode requires --aa-pdb-dir"
    exit 1
  fi
  if [[ ! -d "$AA_PDB_DIR" ]]; then
    echo "aa_pdb_dir does not exist: $AA_PDB_DIR"
    exit 1
  fi
fi

if [[ ! -f "$PYTHON_DIR/NEW_pdb2arr_CG.py" || ! -f "$PYTHON_DIR/NEW_pdb2arr_All_BB.py" ]]; then
  echo "Backbone conversion scripts not found in: $PYTHON_DIR"
  exit 1
fi

if [[ "$RUN_SIDECHAIN" == "1" ]]; then
  if [[ ! -f "$PY_SC_DIR/NEW_pdb2arr_CG_SC.py" || ! -f "$PY_SC_DIR/NEW_pdb2arr_sideChain.py" ]]; then
    echo "Side-chain conversion scripts not found in: $PY_SC_DIR"
    exit 1
  fi
  if [[ "$CG_ONLY" == "0" ]]; then
    if [[ -z "$AA_SC_PDB_DIR" ]]; then
      echo "Side-chain full mode requires --aa-sc-pdb-dir"
      exit 1
    fi
    if [[ ! -d "$AA_SC_PDB_DIR" ]]; then
      echo "aa_sc_pdb_dir does not exist: $AA_SC_PDB_DIR"
      exit 1
    fi
  fi
fi

collect_frames_auto() {
  find "$CG_PDB_DIR" -maxdepth 1 -type f -name 'CG_frame_*.pdb' \
    | sed -E 's#.*/CG_frame_([0-9]+)\.pdb#\1#' | sort -n | uniq
}

collect_frames_from_spec() {
  local spec="$1"
  if [[ "$spec" == *"-"* ]]; then
    local start="${spec%-*}"
    local end="${spec#*-}"
    seq "$start" "$end"
  else
    echo "$spec" | tr ',' '\n'
  fi
}

if [[ "$FRAME_SPEC" == "auto" ]]; then
  FRAMES="$(collect_frames_auto)"
else
  FRAMES="$(collect_frames_from_spec "$FRAME_SPEC")"
fi

if [[ -z "${FRAMES// }" ]]; then
  echo "No frames found. Expected files like CG_frame_<idx>.pdb"
  exit 1
fi

declare -a VALID_FRAMES=()
for frame_idx in $FRAMES; do
  cg_pdb="${CG_PDB_DIR}/CG_frame_${frame_idx}.pdb"
  aa_pdb="${AA_PDB_DIR}/frame_${frame_idx}.pdb"
  aa_sc_pdb="${AA_SC_PDB_DIR}/frame_${frame_idx}_SC.pdb"

  if [[ ! -f "$cg_pdb" ]]; then
    echo "Skipping frame ${frame_idx}: missing ${cg_pdb}"
    continue
  fi
  if [[ "$CG_ONLY" != "1" && ! -f "$aa_pdb" ]]; then
    echo "Skipping frame ${frame_idx}: missing ${aa_pdb}"
    continue
  fi
  if [[ "$RUN_SIDECHAIN" == "1" && "$CG_ONLY" != "1" && ! -f "$aa_sc_pdb" ]]; then
    echo "Skipping frame ${frame_idx}: missing ${aa_sc_pdb}"
    continue
  fi
  VALID_FRAMES+=("$frame_idx")
done

if [[ "${#VALID_FRAMES[@]}" -eq 0 ]]; then
  echo "No valid frames to process after existence checks."
  exit 1
fi

if [[ "$FRESH_START" == "1" ]]; then
  rm -f cluster_ALL_CG.npy
  if [[ "$CG_ONLY" != "1" ]]; then
    rm -f cluster_ALL.npy
  fi
  if [[ "$RUN_SIDECHAIN" == "1" ]]; then
    rm -f "cluster_${SC_CLUSTER_ID}_CG_SC.npy" "cluster_${SC_CLUSTER_ID}_SC.npy" "cluster_${SC_CLUSTER_ID}_CG_BB.npy"
  fi
fi

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/bb_val_arr.XXXXXX")"
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

fix_aa_pdb() {
  local in_pdb="$1"
  local out_pdb="$2"
  # Normalize AA PDB residue names using fixed-width columns:
  # 1) remove terminal prefixes in altLoc column (N/C + standard residue)
  # 2) normalize common histidine/cysteine aliases (HID/HIE/... -> HIS, CYX -> CYS)
  awk '
    BEGIN {
      split("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL", aa_list, " ")
      for (i in aa_list) aa[aa_list[i]] = 1
    }
    function pad80(s, n) {
      n = length(s)
      if (n < 80) return s sprintf("%" (80 - n) "s", "")
      return s
    }
    {
      line = $0
      if ($0 ~ /^(ATOM  |HETATM)/) {
        line = pad80(line)

        alt = toupper(substr(line, 17, 1))
        res = toupper(substr(line, 18, 3))
        if ((alt == "N" || alt == "C") && (res in aa)) {
          line = substr(line, 1, 16) " " res substr(line, 21)
        }

        res = toupper(substr(line, 18, 3))
        if (res == "HID" || res == "HIE" || res == "HIP" || res == "HSD" || res == "HSE" || res == "HSP") {
          line = substr(line, 1, 17) "HIS" substr(line, 21)
        } else if (res == "CYX") {
          line = substr(line, 1, 17) "CYS" substr(line, 21)
        }

        # Fallback alias normalization for non-strict formatting variants.
        gsub(/HID/, "HIS", line)
        gsub(/HIE/, "HIS", line)
        gsub(/HIP/, "HIS", line)
        gsub(/HSD/, "HIS", line)
        gsub(/HSE/, "HIS", line)
        gsub(/HSP/, "HIS", line)
        gsub(/CYX/, "CYS", line)
      }
      print line
    }
  ' "$in_pdb" > "$out_pdb"
}

convert_frame() {
  local frame_idx="$1"
  local skip_sequence="${2:-0}"

  local cg_pdb="${CG_PDB_DIR}/CG_frame_${frame_idx}.pdb"
  local aa_pdb="${AA_PDB_DIR}/frame_${frame_idx}.pdb"
  local aa_sc_pdb="${AA_SC_PDB_DIR}/frame_${frame_idx}_SC.pdb"

  local cg_out="${tmp_dir}/cg_${frame_idx}.npy"

  if [[ "$skip_sequence" == "1" ]]; then
    PDB2ARR_SKIP_SEQUENCE=1 python3 "$PYTHON_DIR/NEW_pdb2arr_CG.py" "$cg_pdb" "$cg_out" "$PDB_NAME" >/dev/null
  else
    python3 "$PYTHON_DIR/NEW_pdb2arr_CG.py" "$cg_pdb" "$cg_out" "$PDB_NAME" >/dev/null
  fi

  if [[ "$CG_ONLY" != "1" ]]; then
    local fixed_pdb="${tmp_dir}/fixed_${frame_idx}.pdb"
    local aa_out="${tmp_dir}/aa_${frame_idx}.npy"
    fix_aa_pdb "$aa_pdb" "$fixed_pdb"
    python3 "$PYTHON_DIR/NEW_pdb2arr_All_BB.py" "$fixed_pdb" "$aa_out" >/dev/null
  fi

  if [[ "$RUN_SIDECHAIN" == "1" ]]; then
    local sc_cg_out="${tmp_dir}/sc_cg_${frame_idx}.npy"
    if [[ "$skip_sequence" == "1" ]]; then
      PDB2ARR_SKIP_SEQUENCE=1 python3 "$PY_SC_DIR/NEW_pdb2arr_CG_SC.py" "$cg_pdb" "$sc_cg_out" "$PDB_NAME" >/dev/null
    else
      python3 "$PY_SC_DIR/NEW_pdb2arr_CG_SC.py" "$cg_pdb" "$sc_cg_out" "$PDB_NAME" >/dev/null
    fi

    if [[ "$CG_ONLY" != "1" ]]; then
      local sc_aa_out="${tmp_dir}/sc_aa_${frame_idx}.npy"
      python3 "$PY_SC_DIR/NEW_pdb2arr_sideChain.py" "$aa_sc_pdb" "$sc_aa_out" >/dev/null
    fi
  fi
}

export PDB_NAME CG_PDB_DIR AA_PDB_DIR AA_SC_PDB_DIR CG_ONLY RUN_SIDECHAIN tmp_dir PYTHON_DIR PY_SC_DIR
export -f fix_aa_pdb
export -f convert_frame

echo "Processing ${#VALID_FRAMES[@]} frame(s) with jobs=${JOBS} ..."

# Run first frame serially to generate sequence files exactly once.
convert_frame "${VALID_FRAMES[0]}" "0"

if [[ "${#VALID_FRAMES[@]}" -gt 1 ]]; then
  REMAINING=("${VALID_FRAMES[@]:1}")
  if [[ "$JOBS" -eq 1 ]]; then
    for frame_idx in "${REMAINING[@]}"; do
      convert_frame "$frame_idx" "1"
    done
  else
    printf '%s\n' "${REMAINING[@]}" \
      | xargs -I{} -P "$JOBS" bash -c 'convert_frame "$1" "1"' _ {}
  fi
fi

concat_temp_arrays() {
  local prefix="$1"
  local out_path="$2"
  python3 - "$tmp_dir" "$prefix" "$out_path" "$FRESH_START" "${VALID_FRAMES[@]}" <<'PY'
import os
import sys
import numpy as np

tmp_dir, prefix, out_path, fresh_start, *frames = sys.argv[1:]

parts = []
expected_width = None

for frame in frames:
    p = os.path.join(tmp_dir, f"{prefix}_{frame}.npy")
    if not os.path.exists(p):
        continue
    arr = np.load(p)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Unexpected array rank for {p}: {arr.ndim}")
    if expected_width is None:
        expected_width = arr.shape[1]
    elif arr.shape[1] != expected_width:
        raise ValueError(
            f"Inconsistent widths for prefix={prefix}: {arr.shape[1]} vs {expected_width} (frame={frame})"
        )
    parts.append(arr)

if not parts:
    raise SystemExit(f"No arrays found for prefix={prefix}")

batch = np.concatenate(parts, axis=0)

if fresh_start != "1" and os.path.exists(out_path):
    existing = np.load(out_path)
    if existing.ndim == 1:
        existing = existing.reshape(1, -1)
    if existing.shape[1] != batch.shape[1]:
        raise ValueError(
            f"Cannot append to {out_path}: existing width {existing.shape[1]} != new width {batch.shape[1]}"
        )
    result = np.concatenate((existing, batch), axis=0)
else:
    result = batch

np.save(out_path, result)
print(f"{out_path}: {result.shape}")
PY
}

concat_temp_arrays "cg" "cluster_ALL_CG.npy"
if [[ "$CG_ONLY" != "1" ]]; then
  concat_temp_arrays "aa" "cluster_ALL.npy"
fi

if [[ "$RUN_SIDECHAIN" == "1" ]]; then
  concat_temp_arrays "sc_cg" "cluster_${SC_CLUSTER_ID}_CG_SC.npy"
  if [[ "$CG_ONLY" != "1" ]]; then
    concat_temp_arrays "sc_aa" "cluster_${SC_CLUSTER_ID}_SC.npy"
  fi

  # Side-chain local-frame scripts expect a backbone CG file named cluster_<id>_CG_BB.npy
  cp -f cluster_ALL_CG.npy "cluster_${SC_CLUSTER_ID}_CG_BB.npy"
fi
