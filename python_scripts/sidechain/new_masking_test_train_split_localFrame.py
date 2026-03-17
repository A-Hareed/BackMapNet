import sys
import os

import numpy as np


def save_mask_compressed(path_no_ext, arr):
    arr = np.asarray(arr, dtype=np.uint8)
    np.savez_compressed(f"{path_no_ext}.npz", arr=arr)


NO_ATOMS_CG = {
    "CYS": [1],
    "ALA": [1],
    "MET": [1],
    "ASP": [1],
    "ASN": [1],
    "ARG": [2],
    "GLN": [1],
    "GLU": [1],
    "GLY": [0],
    "HIS": [3],
    "ILE": [1],
    "LEU": [1],
    "LYS": [2],
    "PHE": [3],
    "PRO": [1],
    "SER": [1],
    "THR": [1],
    "TRP": [4],
    "TYR": [3],
    "VAL": [1],
}

NO_ATOMS_AA = {
    "CYS": [2, [0]],
    "ALA": [1, [0]],
    "MET": [4, [0]],
    "ASP": [4, [0]],
    "ASN": [4, [0]],
    "ARG": [7, ["0,1,2,3,4,5,6,7,8", "9,10,11,12,13,14,15,16,17,18,19,20"]],
    "GLN": [5, [0]],
    "GLU": [5, [0]],
    "GLY": [0, [0]],
    "HIS": [6, ["0,1,2,3,4,5", "6,7,8,9,10,11", "12,13,14,15,16,17"]],
    "ILE": [4, [0]],
    "LEU": [4, [0]],
    "LYS": [5, ["0,1,2,3,4,5,6,7,8", "9,10,11,12,13,14"]],
    "PHE": [7, ["0,1,2,3,4,5,6,7,8", "9,10,11,12,13,14", "15,16,17,18,19,20"]],
    "PRO": [3, [0]],
    "SER": [2, [0]],
    "THR": [3, [0]],
    "TRP": [10, ["0,1,2,3,4,5,27,28,29", "6,7,8,9,10,11,12,13,14", "21,22,23,24,25,26", "15,16,17,18,19,20"]],
    "TYR": [8, ["0,1,2,3,4,5,6,7,8", "9,10,11,12,13,14,15,16,17", "18,19,20,21,22,23"]],
    "VAL": [3, [0]],
}


def _token_width(token):
    if "," in token:
        return len([x for x in token.split(",") if x.strip() != ""])
    if "_" in token:
        a, b = [int(x) for x in token.split("_")]
        return max(0, b - a)
    raise ValueError(f"Unsupported AA group token format: {token}")


def _group_widths_from_sequence(sequence):
    widths = []
    for res in sequence:
        if res not in NO_ATOMS_AA:
            raise KeyError(f"Residue '{res}' not present in NO_ATOMS_AA map")
        n_atoms, groups = NO_ATOMS_AA[res]
        if n_atoms == 0:
            continue
        if isinstance(groups[0], int):
            widths.append(3 * int(n_atoms))
        else:
            for token in groups:
                widths.append(_token_width(token))
    return widths


def _ensure_masking_input(mask_file, pdb, cluster_id):
    seq_file = f"sequence_{pdb}.txt"
    cg_file = f"cluster_{cluster_id}_CG_SC.npy"
    if not os.path.exists(seq_file):
        raise FileNotFoundError(f"Cannot synthesize mask; missing sequence file: {seq_file}")
    if not os.path.exists(cg_file):
        raise FileNotFoundError(f"Cannot synthesize mask; missing CG sidechain file: {cg_file}")

    with open(seq_file, "r", encoding="utf-8") as fh:
        sequence = [x for x in fh.read().strip().split(",") if x]
    if not sequence:
        raise ValueError(f"Empty sequence file: {seq_file}")

    cg = np.load(cg_file, mmap_mode="r")
    if cg.ndim != 2 or cg.shape[1] % 3 != 0:
        raise ValueError(f"Unexpected CG array shape in {cg_file}: {cg.shape}")
    n_frames = int(cg.shape[0])
    m_from_cg = int(cg.shape[1] // 3)

    expected_m = sum(NO_ATOMS_CG[res][0] for res in sequence)
    if expected_m != m_from_cg:
        raise ValueError(
            f"CG bead mismatch: sequence expects {expected_m}, CG file has {m_from_cg} beads."
        )

    widths = _group_widths_from_sequence(sequence)
    if len(widths) != m_from_cg:
        raise ValueError(
            f"AA grouping mismatch: got {len(widths)} groups, expected {m_from_cg} (from CG)."
        )

    expected_shape = (n_frames, 15 * m_from_cg)
    if os.path.exists(mask_file):
        existing = np.load(mask_file, mmap_mode="r")
        if tuple(existing.shape) == expected_shape:
            return
        print(
            f"Regenerating {mask_file}: existing shape {tuple(existing.shape)} != expected {expected_shape}"
        )

    mask = np.full(expected_shape, -2.0, dtype=np.float32)
    for g_idx, width in enumerate(widths):
        if width < 0 or width > 15:
            raise ValueError(f"Invalid group width {width} at group {g_idx}")
        start = g_idx * 15
        mask[:, start:start + width] = 1.0

    np.save(mask_file, mask)
    print(f"Synthesized {mask_file} for CG-only mode with shape {mask.shape}")


def main():
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python new_masking_test_train_split_localFrame.py <pdb_name> [cluster_id]"
        )

    pdb = sys.argv[1]
    cluster_id = int(sys.argv[2]) if len(sys.argv) >= 3 else 2

    mask_file = f"masking_input_{cluster_id}.npy"
    _ensure_masking_input(mask_file, pdb, cluster_id)

    data = np.load(mask_file)
    data = (data != -2).astype(np.uint8)
    data = data.reshape(-1, 15)

    print(f"testing mask shape: {data.shape}")
    save_mask_compressed(f"input_Masking_testing_{pdb}_localFrame", data)


if __name__ == "__main__":
    main()
