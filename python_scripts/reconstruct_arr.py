#!/usr/bin/env python3
"""
Reconstruct full all-atom coordinate arrays by interleaving backbone + side-chain arrays.

Modes
- cg-only:
    default bb = backbone_<PDB>_prediction.npy
    default sc = sidechain_<PDB>_prediction.npy
    default out = combined_<PDB>_prediction.npy
- full:
    default bb = backbone_<PDB>_actual.npy
    default sc = cluster_<SC_CLUSTER_ID>_SC.npy
    default out = combined_<PDB>_actual.npy

You can override any default input/output path using flags.
"""

import argparse
import os
from typing import List, Tuple

import numpy as np


# Side-chain heavy-atom counts per residue.
NO_ATOMS = {
    "CYS": 2,
    "ALA": 1,
    "MET": 4,
    "ASP": 4,
    "ASN": 4,
    "ARG": 7,
    "GLN": 5,
    "GLU": 5,
    "HIS": 6,
    "ILE": 4,
    "LEU": 4,
    "LYS": 5,
    "PHE": 7,
    "PRO": 3,
    "SER": 2,
    "THR": 3,
    "TRP": 10,
    "TYR": 8,
    "VAL": 3,
    "GLY": 0,
}

BB_ATOMS_PER_RES = 4  # N, CA, C, O
COORDS_PER_ATOM = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct full all-atom array by combining backbone + side-chain arrays."
    )
    parser.add_argument(
        "--mode",
        choices=["cg-only", "full"],
        default="cg-only",
        help="cg-only => prediction+prediction, full => actual+AA-sidechain defaults",
    )
    parser.add_argument(
        "--pdb-name",
        default="",
        help="PDB tag used for default file names (e.g., IgE). Required if using defaults.",
    )
    parser.add_argument(
        "--bb-file",
        default="",
        help="Backbone array path (.npy). Overrides mode default.",
    )
    parser.add_argument(
        "--sc-file",
        default="",
        help="Side-chain array path (.npy). Overrides mode default.",
    )
    parser.add_argument(
        "--sequence-file",
        default="",
        help="Sequence file path. Accepts comma-separated residues and optional chain '|' separators.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output file path (.npy). Overrides mode default naming.",
    )
    parser.add_argument(
        "--sc-cluster-id",
        type=int,
        default=2,
        help="Used by full-mode default side-chain AA file: cluster_<id>_SC.npy",
    )
    return parser.parse_args()


def _must_exist(path: str, label: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def _load_2d(path: str, label: str) -> np.ndarray:
    _must_exist(path, label)
    arr = np.load(path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"{label} must be 2D after load, got shape {arr.shape} ({path})")
    return np.asarray(arr, dtype=np.float32)


def _parse_sequence(path: str) -> List[str]:
    _must_exist(path, "Sequence file")
    with open(path, "r", encoding="utf-8") as fh:
        raw = fh.read().strip()
    if not raw:
        raise ValueError(f"Sequence file is empty: {path}")

    sequence: List[str] = []
    for chain_block in raw.split("|"):
        for tok in chain_block.split(","):
            res = tok.strip().upper()
            if not res:
                continue
            if res not in NO_ATOMS:
                raise KeyError(f"Unknown residue '{res}' in sequence file: {path}")
            sequence.append(res)

    if not sequence:
        raise ValueError(f"No residues parsed from sequence file: {path}")
    return sequence


def _resolve_sequence_file(args: argparse.Namespace) -> str:
    if args.sequence_file:
        return args.sequence_file

    if not args.pdb_name:
        raise ValueError("Provide --sequence-file or --pdb-name so defaults can be resolved.")

    full_path = f"sequence_{args.pdb_name}_FULL.txt"
    sc_path = f"sequence_{args.pdb_name}.txt"
    if os.path.isfile(full_path):
        return full_path
    if os.path.isfile(sc_path):
        return sc_path
    raise FileNotFoundError(
        f"Could not find default sequence files: {full_path} or {sc_path}. "
        "Provide --sequence-file explicitly."
    )


def _resolve_io(args: argparse.Namespace) -> Tuple[str, str, str, str]:
    seq_file = _resolve_sequence_file(args)

    if args.mode == "cg-only":
        if not args.pdb_name and (not args.bb_file or not args.sc_file or not args.output):
            raise ValueError(
                "cg-only mode defaults require --pdb-name. "
                "Alternatively provide --bb-file --sc-file --output."
            )
        bb_default = f"backbone_{args.pdb_name}_prediction.npy" if args.pdb_name else ""
        sc_default = f"sidechain_{args.pdb_name}_prediction.npy" if args.pdb_name else ""
        out_default = f"combined_{args.pdb_name}_prediction.npy" if args.pdb_name else ""
    else:
        if not args.pdb_name and (not args.bb_file or not args.output):
            raise ValueError(
                "full mode defaults require --pdb-name. "
                "Alternatively provide --bb-file and --output."
            )
        bb_default = f"backbone_{args.pdb_name}_actual.npy" if args.pdb_name else ""
        sc_default = f"cluster_{args.sc_cluster_id}_SC.npy"
        out_default = f"combined_{args.pdb_name}_actual.npy" if args.pdb_name else ""

    bb_file = args.bb_file or bb_default
    sc_file = args.sc_file or sc_default
    out_file = args.output or out_default

    if not bb_file or not sc_file or not out_file:
        raise ValueError(
            "Could not resolve bb/sc/output file names. "
            "Provide --bb-file --sc-file --output (or --pdb-name for defaults)."
        )

    return bb_file, sc_file, seq_file, out_file


def reconstruct_full_array(
    bb_arr: np.ndarray,
    sc_arr: np.ndarray,
    sequence: List[str],
) -> np.ndarray:
    if bb_arr.shape[0] != sc_arr.shape[0]:
        raise ValueError(
            f"Frame mismatch: backbone has {bb_arr.shape[0]} frames, "
            f"side-chain has {sc_arr.shape[0]} frames."
        )

    expected_bb_cols = len(sequence) * BB_ATOMS_PER_RES * COORDS_PER_ATOM
    expected_sc_cols = sum(NO_ATOMS[res] * COORDS_PER_ATOM for res in sequence)

    if bb_arr.shape[1] != expected_bb_cols:
        raise ValueError(
            f"Backbone width mismatch: got {bb_arr.shape[1]}, "
            f"expected {expected_bb_cols} from sequence length {len(sequence)}."
        )
    if sc_arr.shape[1] != expected_sc_cols:
        raise ValueError(
            f"Side-chain width mismatch: got {sc_arr.shape[1]}, "
            f"expected {expected_sc_cols} from residue map + sequence."
        )

    bb_cursor = 0
    sc_cursor = 0
    blocks = []

    bb_block_size = BB_ATOMS_PER_RES * COORDS_PER_ATOM
    for res in sequence:
        bb_window = slice(bb_cursor, bb_cursor + bb_block_size)
        bb_block = bb_arr[:, bb_window]
        bb_cursor += bb_block_size

        sc_atom_count = NO_ATOMS[res]
        if sc_atom_count > 0:
            sc_block_size = sc_atom_count * COORDS_PER_ATOM
            sc_window = slice(sc_cursor, sc_cursor + sc_block_size)
            sc_block = sc_arr[:, sc_window]
            sc_cursor += sc_block_size
            blocks.append(np.concatenate([bb_block, sc_block], axis=1))
        else:
            blocks.append(bb_block)

    if bb_cursor != bb_arr.shape[1]:
        raise ValueError(
            f"Backbone cursor mismatch after reconstruction: {bb_cursor} vs {bb_arr.shape[1]}"
        )
    if sc_cursor != sc_arr.shape[1]:
        raise ValueError(
            f"Side-chain cursor mismatch after reconstruction: {sc_cursor} vs {sc_arr.shape[1]}"
        )

    return np.concatenate(blocks, axis=1).astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    bb_file, sc_file, seq_file, out_file = _resolve_io(args)

    sequence = _parse_sequence(seq_file)
    bb_arr = _load_2d(bb_file, "Backbone array")
    sc_arr = _load_2d(sc_file, "Side-chain array")

    full_arr = reconstruct_full_array(bb_arr, sc_arr, sequence)

    out_dir = os.path.dirname(os.path.abspath(out_file))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(out_file, full_arr)

    print("Mode:", args.mode)
    print("Backbone:", bb_file, bb_arr.shape)
    print("Sidechain:", sc_file, sc_arr.shape)
    print("Sequence:", seq_file, f"residues={len(sequence)}")
    print("Saved:", out_file, full_arr.shape)


if __name__ == "__main__":
    main()
