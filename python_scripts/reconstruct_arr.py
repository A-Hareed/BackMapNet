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
from typing import Dict, List, Tuple

import numpy as np


# Side-chain atom order must match MakePDB_temp.py / reorder_sidechain_pdbs2.py.
SIDECHAIN_ATOMS: Dict[str, List[str]] = {
    "LYS": ["CB", "CG", "CD", "CE", "NZ"],
    "ALA": ["CB"],
    "CYS": ["CB", "SG"],
    "GLN": ["CB", "CG", "CD", "OE1", "NE2"],
    "VAL": ["CB", "CG1", "CG2"],
    "ASN": ["CB", "CG", "OD1", "ND2"],
    "LEU": ["CB", "CG", "CD1", "CD2"],
    "THR": ["CB", "CG2", "OG1"],
    "PHE": ["CB", "CG", "CD1", "CE1", "CZ", "CE2", "CD2"],
    "SER": ["CB", "OG"],
    "PRO": ["CD", "CG", "CB"],
    "TYR": ["CB", "CG", "CD1", "CE1", "CZ", "OH", "CE2", "CD2"],
    "HIS": ["CB", "CG", "ND1", "CE1", "NE2", "CD2"],
    "ARG": ["CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "TRP": ["CB", "CG", "CD1", "NE1", "CE2", "CZ2", "CH2", "CZ3", "CE3", "CD2"],
    "ILE": ["CB", "CG2", "CG1", "CD"],
    "GLU": ["CB", "CG", "CD", "OE1", "OE2"],
    "ASP": ["CB", "CG", "OD1", "OD2"],
    "MET": ["CB", "CG", "SD", "CE"],
    "GLY": [],
}

NO_ATOMS = {res: len(atoms) for res, atoms in SIDECHAIN_ATOMS.items()}

BB_ATOMS_PER_RES = 4  # N, CA, C, O
COORDS_PER_ATOM = 3
IDEAL_CA_CB = 1.526
IDEAL_C_O = 1.229


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
    parser.add_argument(
        "--fix-anchor-bonds",
        action="store_true",
        help="Apply mild CA-CB and C-O bond-length correction after reconstruction.",
    )
    parser.add_argument(
        "--ca-cb-mode",
        choices=["move-ca", "move-sidechain", "split"],
        default="move-ca",
        help=(
            "CA-CB correction mode. move-ca anchors CB and moves CA; "
            "move-sidechain anchors CA and moves the full sidechain; split moves both."
        ),
    )
    parser.add_argument(
        "--ca-cb-threshold",
        type=float,
        default=0.3,
        help="Only correct CA-CB when absolute length error is above this Angstrom threshold.",
    )
    parser.add_argument(
        "--ca-cb-alpha",
        type=float,
        default=0.2,
        help="CA-CB correction strength in [0,1].",
    )
    parser.add_argument(
        "--c-o-threshold",
        type=float,
        default=0.3,
        help="Only correct C-O when absolute length error is above this Angstrom threshold.",
    )
    parser.add_argument(
        "--c-o-alpha",
        type=float,
        default=0.5,
        help="C-O correction strength in [0,1].",
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


def _validate_alpha_threshold(alpha: float, threshold: float, label: str) -> None:
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError(f"{label} alpha must be in [0,1], got {alpha}")
    if threshold < 0.0:
        raise ValueError(f"{label} threshold must be >= 0, got {threshold}")


def _pull_atom_to_ideal_distance(
    coords3: np.ndarray,
    anchor_idx: int,
    moving_idx: int,
    ideal_len: float,
    threshold: float,
    alpha: float,
    eps: float = 1e-8,
) -> int:
    vec = coords3[:, moving_idx, :] - coords3[:, anchor_idx, :]
    dist = np.sqrt(np.sum(vec * vec, axis=1, keepdims=True))
    err = np.abs(dist - np.float32(ideal_len))
    apply = (dist > eps) & (err > np.float32(threshold))
    if not np.any(apply):
        return 0

    target = coords3[:, anchor_idx, :] + vec * (np.float32(ideal_len) / (dist + np.float32(eps)))
    delta = np.float32(alpha) * (target - coords3[:, moving_idx, :])
    coords3[:, moving_idx, :] = np.where(apply, coords3[:, moving_idx, :] + delta, coords3[:, moving_idx, :])
    return int(np.count_nonzero(apply))


def _move_group_to_ideal_distance(
    coords3: np.ndarray,
    anchor_idx: int,
    moving_idx: int,
    group_idxs: List[int],
    ideal_len: float,
    threshold: float,
    alpha: float,
    eps: float = 1e-8,
) -> int:
    vec = coords3[:, moving_idx, :] - coords3[:, anchor_idx, :]
    dist = np.sqrt(np.sum(vec * vec, axis=1, keepdims=True))
    err = np.abs(dist - np.float32(ideal_len))
    apply = (dist > eps) & (err > np.float32(threshold))
    if not np.any(apply):
        return 0

    target = coords3[:, anchor_idx, :] + vec * (np.float32(ideal_len) / (dist + np.float32(eps)))
    delta = np.float32(alpha) * (target - coords3[:, moving_idx, :])
    for idx in group_idxs:
        coords3[:, idx, :] = np.where(apply, coords3[:, idx, :] + delta, coords3[:, idx, :])
    return int(np.count_nonzero(apply))


def _split_ca_cb_correction(
    coords3: np.ndarray,
    ca_idx: int,
    cb_idx: int,
    sidechain_idxs: List[int],
    ideal_len: float,
    threshold: float,
    alpha: float,
    eps: float = 1e-8,
) -> int:
    vec = coords3[:, ca_idx, :] - coords3[:, cb_idx, :]
    dist = np.sqrt(np.sum(vec * vec, axis=1, keepdims=True))
    err = np.abs(dist - np.float32(ideal_len))
    apply = (dist > eps) & (err > np.float32(threshold))
    if not np.any(apply):
        return 0

    midpoint = 0.5 * (coords3[:, ca_idx, :] + coords3[:, cb_idx, :])
    unit = vec / (dist + np.float32(eps))
    target_ca = midpoint + 0.5 * np.float32(ideal_len) * unit
    target_cb = midpoint - 0.5 * np.float32(ideal_len) * unit
    ca_delta = np.float32(alpha) * (target_ca - coords3[:, ca_idx, :])
    sc_delta = np.float32(alpha) * (target_cb - coords3[:, cb_idx, :])

    coords3[:, ca_idx, :] = np.where(apply, coords3[:, ca_idx, :] + ca_delta, coords3[:, ca_idx, :])
    for idx in sidechain_idxs:
        coords3[:, idx, :] = np.where(apply, coords3[:, idx, :] + sc_delta, coords3[:, idx, :])
    return int(np.count_nonzero(apply))


def fix_anchor_bonds(
    full_arr: np.ndarray,
    sequence: List[str],
    ca_cb_mode: str,
    ca_cb_threshold: float,
    ca_cb_alpha: float,
    c_o_threshold: float,
    c_o_alpha: float,
) -> Tuple[np.ndarray, int, int]:
    _validate_alpha_threshold(ca_cb_alpha, ca_cb_threshold, "CA-CB")
    _validate_alpha_threshold(c_o_alpha, c_o_threshold, "C-O")

    coords3 = full_arr.reshape(full_arr.shape[0], -1, 3).copy()
    atom_offset = 0
    ca_cb_fixed = 0
    c_o_fixed = 0

    for res in sequence:
        ca_idx = atom_offset + 1
        c_idx = atom_offset + 2
        o_idx = atom_offset + 3
        sc_atoms = SIDECHAIN_ATOMS[res]
        sidechain_idxs = [atom_offset + BB_ATOMS_PER_RES + i for i in range(len(sc_atoms))]

        c_o_fixed += _pull_atom_to_ideal_distance(
            coords3=coords3,
            anchor_idx=c_idx,
            moving_idx=o_idx,
            ideal_len=IDEAL_C_O,
            threshold=c_o_threshold,
            alpha=c_o_alpha,
        )

        if "CB" in sc_atoms:
            cb_idx = atom_offset + BB_ATOMS_PER_RES + sc_atoms.index("CB")
            if ca_cb_mode == "move-ca":
                ca_cb_fixed += _pull_atom_to_ideal_distance(
                    coords3=coords3,
                    anchor_idx=cb_idx,
                    moving_idx=ca_idx,
                    ideal_len=IDEAL_CA_CB,
                    threshold=ca_cb_threshold,
                    alpha=ca_cb_alpha,
                )
            elif ca_cb_mode == "move-sidechain":
                ca_cb_fixed += _move_group_to_ideal_distance(
                    coords3=coords3,
                    anchor_idx=ca_idx,
                    moving_idx=cb_idx,
                    group_idxs=sidechain_idxs,
                    ideal_len=IDEAL_CA_CB,
                    threshold=ca_cb_threshold,
                    alpha=ca_cb_alpha,
                )
            elif ca_cb_mode == "split":
                ca_cb_fixed += _split_ca_cb_correction(
                    coords3=coords3,
                    ca_idx=ca_idx,
                    cb_idx=cb_idx,
                    sidechain_idxs=sidechain_idxs,
                    ideal_len=IDEAL_CA_CB,
                    threshold=ca_cb_threshold,
                    alpha=ca_cb_alpha,
                )
            else:
                raise ValueError(f"Unknown CA-CB correction mode: {ca_cb_mode}")

        atom_offset += BB_ATOMS_PER_RES + len(sc_atoms)

    return coords3.reshape(full_arr.shape).astype(np.float32, copy=False), ca_cb_fixed, c_o_fixed


def main() -> None:
    args = parse_args()
    bb_file, sc_file, seq_file, out_file = _resolve_io(args)

    sequence = _parse_sequence(seq_file)
    bb_arr = _load_2d(bb_file, "Backbone array")
    sc_arr = _load_2d(sc_file, "Side-chain array")

    full_arr = reconstruct_full_array(bb_arr, sc_arr, sequence)
    ca_cb_fixed = 0
    c_o_fixed = 0
    if args.fix_anchor_bonds:
        full_arr, ca_cb_fixed, c_o_fixed = fix_anchor_bonds(
            full_arr=full_arr,
            sequence=sequence,
            ca_cb_mode=args.ca_cb_mode,
            ca_cb_threshold=args.ca_cb_threshold,
            ca_cb_alpha=args.ca_cb_alpha,
            c_o_threshold=args.c_o_threshold,
            c_o_alpha=args.c_o_alpha,
        )

    out_dir = os.path.dirname(os.path.abspath(out_file))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(out_file, full_arr)

    print("Mode:", args.mode)
    print("Backbone:", bb_file, bb_arr.shape)
    print("Sidechain:", sc_file, sc_arr.shape)
    print("Sequence:", seq_file, f"residues={len(sequence)}")
    if args.fix_anchor_bonds:
        print(
            "Anchor-bond correction:",
            f"CA-CB fixed={ca_cb_fixed}",
            f"C-O fixed={c_o_fixed}",
            f"CA-CB mode={args.ca_cb_mode}",
        )
    print("Saved:", out_file, full_arr.shape)


if __name__ == "__main__":
    main()
