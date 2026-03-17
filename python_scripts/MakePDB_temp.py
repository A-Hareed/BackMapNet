#!/usr/bin/env python3
"""
Write multi-chain all-atom PDB frames from a flattened coordinate array.

This writer expects coordinate columns in residue order:
  backbone (N,CA,C,O) + sidechain atoms
where sidechain atom order matches reorder_sidechain_pdbs2.py training order.
"""

import argparse
import os
from typing import List, Tuple

import numpy as np


BACKBONE_ATOMS = ("N", "CA", "C", "O")

# Must stay aligned with reorder_sidechain_pdbs2.py ATOM_ORDER flattening.
SIDECHAIN_ATOMS = {
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

RES_ALIASES = {
    "HSD": "HIS",
    "HSE": "HIS",
    "HSP": "HIS",
    "HID": "HIS",
    "HIE": "HIS",
    "HIP": "HIS",
    "MSE": "MET",
}

CHAIN_ID_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

PDB_ATOM_TMPL = (
    "{record:<6s}{serial:>5d} {name:^4s}{altLoc:1s}{resName:>3s} {chainID:1s}"
    "{resSeq:>4d}{iCode:1s}   {x:8.3f}{y:8.3f}{z:8.3f}"
    "{occupancy:6.2f}{tempFactor:6.2f}          {element:>2s}{charge:2s}"
)
PDB_TER_TMPL = "TER   {serial:>5d}      {resName:>3s} {chainID:1s}{resSeq:>4d}\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write all-atom PDB frames from flattened coordinate arrays."
    )
    parser.add_argument(
        "--coords-file",
        required=True,
        help="Input npy coordinates with shape (n_frames, 3*total_atoms).",
    )
    parser.add_argument(
        "--sequence-file",
        required=True,
        help="Sequence text; residues comma-separated, chains separated by '|'.",
    )
    parser.add_argument(
        "--output-dir",
        default="pdb_frames",
        help="Directory for generated PDB files.",
    )
    parser.add_argument(
        "--frame-spec",
        default="all",
        help="Frame selection: all, single index (5), range (0-99), or list (0,5,10).",
    )
    parser.add_argument(
        "--filename-template",
        default="frame_BackMapNet_V3_{frame}.pdb",
        help="Output filename template. Available fields: {frame}, {i}.",
    )
    parser.add_argument(
        "--chain-lengths",
        default="",
        help=(
            "Optional CSV chain lengths to enforce chain boundaries when the sequence "
            "file is a single flat list (no '|'). Example: 546,215,546,215,121"
        ),
    )
    return parser.parse_args()


def normalize_residue_name(res_name: str) -> str:
    name = res_name.strip().upper()
    return RES_ALIASES.get(name, name)


def parse_sequence(seq_path: str) -> List[List[str]]:
    with open(seq_path, "r", encoding="utf-8") as handle:
        raw = handle.read().strip()
    if not raw:
        raise ValueError(f"Sequence file is empty: {seq_path}")

    chain_blocks = [blk.strip() for blk in raw.split("|") if blk.strip()]
    if not chain_blocks:
        raise ValueError(f"No residues found in sequence file: {seq_path}")

    chains: List[List[str]] = []
    for block in chain_blocks:
        residues = []
        for tok in block.split(","):
            tok = tok.strip()
            if not tok:
                continue
            res = normalize_residue_name(tok)
            if res not in SIDECHAIN_ATOMS:
                raise KeyError(f"Unknown residue '{tok}' (normalized '{res}') in {seq_path}")
            residues.append(res)
        if residues:
            chains.append(residues)

    if not chains:
        raise ValueError(f"No valid residues parsed from sequence file: {seq_path}")
    if len(chains) > len(CHAIN_ID_ALPHABET):
        raise ValueError(
            f"Too many chains ({len(chains)}) for one-character chain IDs."
        )
    return chains


def parse_chain_lengths(chain_lengths_csv: str) -> List[int]:
    text = (chain_lengths_csv or "").strip()
    if not text:
        return []
    lengths = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        value = int(tok)
        if value <= 0:
            raise ValueError(f"Invalid chain length '{tok}' in --chain-lengths.")
        lengths.append(value)
    return lengths


def apply_chain_lengths(chains: List[List[str]], chain_lengths: List[int]) -> List[List[str]]:
    if not chain_lengths:
        return chains

    flat = [res for chain in chains for res in chain]
    if sum(chain_lengths) != len(flat):
        raise ValueError(
            "Chain-length sum does not match sequence residue count: "
            f"sum(chain_lengths)={sum(chain_lengths)} residues={len(flat)}."
        )

    out: List[List[str]] = []
    start = 0
    for n in chain_lengths:
        out.append(flat[start:start + n])
        start += n
    return out


def build_atom_metadata(chains: List[List[str]]) -> Tuple[List[str], List[int], List[str], List[str], List[int]]:
    atom_names: List[str] = []
    residue_indices: List[int] = []
    residue_names: List[str] = []
    chain_ids: List[str] = []
    chain_end_idxs: List[int] = []

    for c_idx, residues in enumerate(chains):
        chain_id = CHAIN_ID_ALPHABET[c_idx]
        for res_seq, res in enumerate(residues, start=1):
            for atom in BACKBONE_ATOMS:
                atom_names.append(atom)
                residue_indices.append(res_seq)
                residue_names.append(res)
                chain_ids.append(chain_id)
            for atom in SIDECHAIN_ATOMS[res]:
                atom_names.append(atom)
                residue_indices.append(res_seq)
                residue_names.append(res)
                chain_ids.append(chain_id)
        chain_end_idxs.append(len(atom_names) - 1)

    return atom_names, residue_indices, residue_names, chain_ids, chain_end_idxs


def parse_frame_spec(spec: str, n_frames: int) -> List[int]:
    spec = spec.strip().lower()
    if spec == "all":
        return list(range(n_frames))
    if "-" in spec:
        start_s, end_s = spec.split("-", 1)
        start = int(start_s)
        end = int(end_s)
        if end < start:
            raise ValueError(f"Invalid range '{spec}': end < start.")
        frames = list(range(start, end + 1))
    elif "," in spec:
        frames = [int(x.strip()) for x in spec.split(",") if x.strip()]
    else:
        frames = [int(spec)]

    for idx in frames:
        if idx < 0 or idx >= n_frames:
            raise IndexError(f"Requested frame {idx} outside valid range [0, {n_frames - 1}]")
    return frames


def atom_element(atom_name: str) -> str:
    if not atom_name:
        return ""
    # PDB atom names may include digits; keep only letters for element parsing.
    letters = "".join(ch for ch in atom_name.strip().upper() if ch.isalpha())
    if not letters:
        return ""
    # Preserve two-letter ions/elements when present; keep CA reserved for alpha carbon.
    if len(letters) >= 2 and letters[:2] in {"CL", "BR", "FE", "MG", "ZN", "NA", "SE"}:
        return letters[:2]
    return letters[0]


def write_frame_pdb(
    coords: np.ndarray,
    out_path: str,
    atom_names: List[str],
    residue_indices: List[int],
    residue_names: List[str],
    chain_ids: List[str],
    chain_end_idxs: List[int],
) -> None:
    with open(out_path, "w", encoding="utf-8") as handle:
        atom_serial = 1
        total_atoms = len(atom_names)

        for i in range(total_atoms):
            x, y, z = coords[3 * i: 3 * i + 3]
            atom_name = atom_names[i]
            handle.write(
                PDB_ATOM_TMPL.format(
                    record="ATOM",
                    serial=atom_serial,
                    name=atom_name,
                    altLoc=" ",
                    resName=residue_names[i],
                    chainID=chain_ids[i],
                    resSeq=residue_indices[i],
                    iCode=" ",
                    x=float(x),
                    y=float(y),
                    z=float(z),
                    occupancy=1.00,
                    tempFactor=0.00,
                    element=atom_element(atom_name),
                    charge="",
                )
                + "\n"
            )
            atom_serial += 1

            if i in chain_end_idxs:
                handle.write(
                    PDB_TER_TMPL.format(
                        serial=atom_serial,
                        resName=residue_names[i],
                        chainID=chain_ids[i],
                        resSeq=residue_indices[i],
                    )
                )
                atom_serial += 1

        handle.write("END\n")


def main() -> None:
    args = parse_args()

    chains = parse_sequence(args.sequence_file)
    chain_lengths = parse_chain_lengths(args.chain_lengths)
    chains = apply_chain_lengths(chains, chain_lengths)
    atom_names, residue_indices, residue_names, chain_ids, chain_end_idxs = build_atom_metadata(chains)
    total_atoms = len(atom_names)

    coords_all = np.load(args.coords_file)
    if coords_all.ndim == 1:
        coords_all = coords_all.reshape(1, -1)
    if coords_all.ndim != 2:
        raise ValueError(f"coords array must be 2D, got shape {coords_all.shape}")

    n_frames, n_cols = coords_all.shape
    expected_cols = total_atoms * 3
    if n_cols != expected_cols:
        raise ValueError(
            f"Coordinate width mismatch for {args.coords_file}: got {n_cols}, expected {expected_cols} "
            f"(total_atoms={total_atoms})."
        )

    frame_indices = parse_frame_spec(args.frame_spec, n_frames)
    os.makedirs(args.output_dir, exist_ok=True)

    for out_i, frame_idx in enumerate(frame_indices):
        file_name = args.filename_template.format(frame=frame_idx, i=out_i)
        out_path = os.path.join(args.output_dir, file_name)
        write_frame_pdb(
            coords=coords_all[frame_idx],
            out_path=out_path,
            atom_names=atom_names,
            residue_indices=residue_indices,
            residue_names=residue_names,
            chain_ids=chain_ids,
            chain_end_idxs=chain_end_idxs,
        )
        print(f"Wrote {out_path}")

    print(f"Done writing {len(frame_indices)} frame(s).")


if __name__ == "__main__":
    main()
