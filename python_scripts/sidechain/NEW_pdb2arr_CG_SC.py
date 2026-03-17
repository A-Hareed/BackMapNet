import os
import sys

import numpy as np


def parse_cg_sc_pdb(cg_pdb_file_path):
    """
    Parse side-chain CG coordinates and sequences from a CG PDB.

    Coordinate filter (legacy-compatible):
    - keep atoms where atom_name != "BB" OR residue_name == "ALA"

    Sequence extraction (from BB atoms, first model only):
    - full sequence: all residues (including GLY)
    - sidechain sequence: residues excluding GLY
    """
    current_model_coords = []
    full_data = []

    full_sequence = []
    sc_sequence = []

    first_model_done = False
    prev_res_key_first_model = None

    with open(cg_pdb_file_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("ENDMDL"):
                if current_model_coords:
                    full_data.append(current_model_coords)
                    current_model_coords = []
                first_model_done = True
                continue

            if not line.startswith("ATOM") or len(line) < 54:
                continue

            atom_name = line[12:16].strip()
            residue_name = line[17:21].strip()
            chain_id = line[21].strip() or "_"
            resseq_text = line[22:26].strip()
            ins_code = line[26].strip()
            if not resseq_text:
                continue
            try:
                resseq = int(resseq_text)
            except ValueError:
                continue

            # Side-chain CG coordinate extraction
            if atom_name != "BB" or residue_name == "ALA":
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    current_model_coords.extend([x, y, z])
                except ValueError:
                    continue

            # Sequence extraction from BB atoms for the first model only.
            # Track residues in file order instead of de-duplicating by
            # (chain, resseq), because some CG PDBs reuse blank chain IDs
            # and restart residue numbers at chain boundaries.
            if (not first_model_done) and atom_name == "BB":
                res_key = (chain_id, resseq, ins_code)
                if prev_res_key_first_model is None or res_key != prev_res_key_first_model:
                    prev_res_key_first_model = res_key
                    full_sequence.append(residue_name)
                    if residue_name != "GLY":
                        sc_sequence.append(residue_name)

    if current_model_coords:
        full_data.append(current_model_coords)

    return full_data, full_sequence, sc_sequence


def append_or_save(output_array_path, arr):
    if os.path.exists(output_array_path):
        existing = np.load(output_array_path)
        if existing.ndim == 1:
            existing = existing.reshape(1, -1)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if existing.shape[1] != arr.shape[1]:
            raise ValueError(
                f"Cannot concatenate arrays with different widths: "
                f"{existing.shape[1]} vs {arr.shape[1]}"
            )
        arr = np.concatenate((existing, arr), axis=0)
    np.save(output_array_path, arr)


def maybe_write_sequences(sequence_name, full_sequence, sc_sequence):
    if not sequence_name:
        return
    if os.environ.get("PDB2ARR_SKIP_SEQUENCE", "0") == "1":
        return

    full_out = f"sequence_{sequence_name}_FULL.txt"
    sc_out = f"sequence_{sequence_name}.txt"

    if full_sequence and not os.path.exists(full_out):
        with open(full_out, "w", encoding="utf-8") as fh:
            fh.write(",".join(full_sequence))

    if sc_sequence and not os.path.exists(sc_out):
        with open(sc_out, "w", encoding="utf-8") as fh:
            fh.write(",".join(sc_sequence))


def main():
    if len(sys.argv) not in (3, 4):
        raise SystemExit(
            "Usage: python NEW_pdb2arr_CG_SC.py <cg_pdb_file_path> <output_array_path> [sequence_name]"
        )

    cg_pdb_file_path = sys.argv[1]
    output_array_path = sys.argv[2]
    sequence_name = sys.argv[3] if len(sys.argv) == 4 else None

    if not os.path.exists(cg_pdb_file_path):
        raise FileNotFoundError(f"Coarse-grained PDB file not found: {cg_pdb_file_path}")

    full_data, full_sequence, sc_sequence = parse_cg_sc_pdb(cg_pdb_file_path)
    if not full_data:
        raise SystemExit(
            f"No side-chain CG bead coordinates found in {cg_pdb_file_path} after filtering."
        )

    arr = np.asarray(full_data, dtype=np.float32)
    append_or_save(output_array_path, arr)
    maybe_write_sequences(sequence_name, full_sequence, sc_sequence)


if __name__ == "__main__":
    main()
