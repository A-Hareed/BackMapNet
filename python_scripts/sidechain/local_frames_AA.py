#!/usr/bin/env python3
import numpy as np
import sys
import re

"""
Option B: Normalize ALL-ATOM targets using local frames R + CG bead coords,
WITHOUT requiring bead_to_res/bead_to_num in meta.

Uses:
  - R_localFrame_{pdb}_cluster{cluster_id}.npy     (N, L, 3, 3)
  - localFrame_META_{pdb}_cluster{cluster_id}.npz  (optional: delta_range, delta_half_range, sequence)
  - CG bead coords file (N, 3*M)                   (sidechain beads)
  - ALL-ATOM target file                           (cluster_{id}_SC.npy)

Keeps your all-atom ordering and padding exactly (15 per bead-group, pad=0 and mask=-2).

Usage:
  python normalize_targets_localframe_optionB.py <all_atom_file.npy> <PDBNAME> <cg_bead_file.npy> [cluster_id]

Example:
  python normalize_targets_localframe_optionB.py cluster_12_SC.npy 1J4N Feature_array/cluster_12_CG_SC.npy 12
"""

CLIP_TO_01 = True

# -------------------------
# Inputs
# -------------------------
all_atom_file = sys.argv[1]
pdb = sys.argv[2]
cg_bead_file = sys.argv[3]

# cluster id
if len(sys.argv) >= 5 and sys.argv[4].strip().isdigit():
    cluster_id = int(sys.argv[4])
else:
    m = re.search(r"cluster_(\d+)_SC\.npy", all_atom_file)
    if m is None:
        raise ValueError(f"Could not parse cluster id from: {all_atom_file}")
    cluster_id = int(m.group(1))

# sequence (3-letter)
with open(f"sequence_{pdb}.txt", "r") as f:
    sequence = [x for x in f.read().strip().split(",") if x]
L = len(sequence)

# load arrays
all_atoms = np.load(all_atom_file).astype(np.float32)  # (N, 3*total_atoms_flat)
cg_beads = np.load(cg_bead_file).astype(np.float32)    # (N, 3*M)
N = all_atoms.shape[0]
if cg_beads.shape[0] != N:
    raise ValueError(f"N mismatch: all_atoms N={N}, cg_beads N={cg_beads.shape[0]}")

if cg_beads.shape[1] % 3 != 0:
    raise ValueError(f"CG bead file second dim must be divisible by 3, got {cg_beads.shape[1]}")

M_from_file = cg_beads.shape[1] // 3  # number of sidechain beads in CG file

# -------------------------
# Load R frames
# -------------------------
R_path = f"R_localFrame_{pdb}_cluster{cluster_id}.npy"
R_all = np.load(R_path, mmap_mode="r")  # (N, L, 3, 3)
if R_all.shape[0] != N or R_all.shape[1] != L:
    raise ValueError(f"R_all shape mismatch: got {R_all.shape}, expected ({N},{L},3,3)")

# -------------------------
# Load meta (optional fields)
# -------------------------
delta_half_range = 3.5
delta_range = 7.0
meta_seq = None

meta_path = f"localFrame_META_{pdb}_cluster{cluster_id}.npz"
try:
    meta = np.load(meta_path, allow_pickle=True)
    files = set(meta.files)

    if "delta_half_range" in files:
        delta_half_range = float(meta["delta_half_range"][0])
    if "delta_range" in files:
        delta_range = float(meta["delta_range"][0])
    if "sequence" in files:
        meta_seq = meta["sequence"].astype(str).tolist()

    if meta_seq is not None and meta_seq != sequence:
        raise ValueError("Sequence mismatch between sequence file and localFrame_META.")

except FileNotFoundError:
    print(f"WARNING: meta file not found: {meta_path}. Using default delta_range=7.0, delta_half_range=3.5")

# ==========================================================
# CG bead layout map (must match cg_bead_file ordering)
# ==========================================================
no_Atoms_CG = {
 'CYS':[1,['0,1,2'],[1.0,1]],
 'ALA':[1,['0,1,2'],[1.0,2]],
 'MET':[1,['0,1,2'],[1.0,1]],
 'ASP':[1,['0,1,2'],[1.0,3]],
 'ASN':[1,['0,1,2'],[1.0,3]],
 'ARG':[2,['0,1,2','3,4,5'],[1,[1,3]]],
 'GLN':[1,['0,1,2'],[1.0,2]],
 'GLU':[1,['0,1,2'],[1.0,3]],
 'GLY':[0,[0]],
 'HIS':[3,['0,1,2','3,4,5','6,7,8'],[1,[0,2,2]]],
 'ILE':[1,['0,1,2'],[1.0,0]],
 'LEU':[1,['0,1,2'],[1.0,0]],
 'LYS':[2,['0,1,2','3,4,5'],[1,[0,3]]],
 'PHE':[3,['0,1,2','3,4,5','6,7,8'],[1,[0,0,0]]],
 'PRO':[1,['0,1,2'],[1.0,0]],
 'SER':[1,['0,1,2'],[1.0,2]],
 'THR':[1,['0,1,2'],[1.0,2]],
 'TRP':[4,['0,1,2','3,4,5','6,7,8','9,10,11'],[1,[0,2,0,0]]],
 'TYR':[3,['0,1,2','3,4,5','6,7,8'],[1,[0,0,2]]],
 'VAL':[1,['0,1,2'],[1.0,0]]
}

# ==========================================================
# All-atom grouping map (must match your OLD target script ordering)
# ALA is kept correct.
# ==========================================================
no_Atoms_AA = {
 'CYS':[2,[0]],
 'ALA':[1,[0]],   # correct ALA
 'MET':[4,[0]],
 'ASP':[4,[0]],
 'ASN':[4,[0]],
 'ARG':[7,['0,1,2,3,4,5,6,7,8','9,10,11,12,13,14,15,16,17,18,19,20']],
 'GLN':[5,[0]],
 'GLU':[5,[0]],
 'GLY':[0,[0]],
 'HIS':[6,['0,1,2,3,4,5','6,7,8,9,10,11','12,13,14,15,16,17']],
 'ILE':[4,[0]],
 'LEU':[4,[0]],
 'LYS':[5,['0,1,2,3,4,5,6,7,8','9,10,11,12,13,14']],
 'PHE':[7,['0,1,2,3,4,5,6,7,8','9,10,11,12,13,14','15,16,17,18,19,20']],
 'PRO':[3,[0]],
 'SER':[2,[0]],
 'THR':[3,[0]],
 'TRP':[10,['0,1,2,3,4,5,27,28,29','6,7,8,9,10,11,12,13,14','21,22,23,24,25,26','15,16,17,18,19,20']],
 'TYR':[8,['0,1,2,3,4,5,6,7,8','9,10,11,12,13,14,15,16,17','18,19,20,21,22,23']],
 'VAL':[3,[0]]
}

# -------------------------
# Infer expected bead count from CG map and compare to CG file
# -------------------------
expected_M = sum(no_Atoms_CG[res][0] for res in sequence)
if expected_M != M_from_file:
    raise ValueError(
        f"CG bead file implies M={M_from_file} beads but sequence+CG map expects {expected_M} beads.\n"
        "This means your CG file ordering/layout does not match the CG bead map."
    )
M = expected_M

# -------------------------
# Infer bead_to_res and bead_to_num (Option B)
# -------------------------
bead_to_res = np.empty((M,), dtype=np.int32)
bead_to_num = np.empty((M,), dtype=np.int32)

k = 0
for i, res in enumerate(sequence):
    n_beads = no_Atoms_CG[res][0]
    if n_beads == 0:
        continue
    for num in range(n_beads):
        bead_to_res[k] = i
        bead_to_num[k] = num
        k += 1
if k != M:
    raise RuntimeError(f"Internal error building bead_to_res: filled {k}, expected {M}")

# -------------------------
# Build bead_xyz_all (N, M, 3) from cg_beads in residue order
# -------------------------
bead_xyz_all = np.empty((N, M, 3), dtype=np.float32)
cg_ptr = 0
k = 0

for i, res in enumerate(sequence):
    n_beads = no_Atoms_CG[res][0]
    if n_beads == 0:
        continue

    block = cg_beads[:, cg_ptr:cg_ptr + 3*n_beads]  # (N, 3*n_beads)
    cg_ptr += 3*n_beads

    if n_beads == 1:
        bead_xyz_all[:, k, :] = block.reshape(N, 3)
        k += 1
    else:
        splicer_list = no_Atoms_CG[res][1]  # list of "0,1,2", "3,4,5", ...
        for num, sp_str in enumerate(splicer_list):
            sp = [int(x) for x in sp_str.split(",")]
            bead_xyz_all[:, k, :] = block[:, sp]
            k += 1

if cg_ptr != cg_beads.shape[1]:
    raise ValueError(f"Did not consume all CG bead cols. Used {cg_ptr}, total {cg_beads.shape[1]}")
if k != M:
    raise ValueError(f"Did not fill all bead_xyz. Filled {k}, expected {M}")

# -------------------------
# Total AA bead-groups must match M (alignment check)
# -------------------------
groups_total = 0
for res in sequence:
    n_atoms, groups = no_Atoms_AA[res]
    if n_atoms == 0:
        continue
    if isinstance(groups[0], int):
        groups_total += 1
    else:
        groups_total += len(groups)

if groups_total != M:
    raise ValueError(
        f"AA grouping produces {groups_total} groups but CG bead count is M={M}.\n"
        "This means your AA grouping order is not aligned with your CG bead order."
    )

# -------------------------
# Local-frame normalization helper
# -------------------------
def normalize_atoms_localframe(atoms_flat, bead_xyz, R, delta_range):
    # atoms_flat: (N, 3*n_atoms_group)
    atoms = atoms_flat.reshape(N, -1, 3)
    d_global = atoms - bead_xyz[:, None, :]
    d_local = np.einsum("nij,nkj->nki", R.transpose(0, 2, 1), d_global)  # R^T * d
    norm = 0.5 + (d_local / delta_range)
#    if CLIP_TO_01:
#        norm = np.clip(norm, 0.0, 1.0)
    return norm.reshape(atoms_flat.shape).astype(np.float32)

# -------------------------
# Allocate output arrays (15 coords per bead-group)
# -------------------------
final_arr = np.zeros((N, 15 * M), dtype=np.float32)
masking_arr = np.full((N, 15 * M), -2.0, dtype=np.float32)

# -------------------------
# Normalize all-atoms with EXACT SAME ordering as your old script
# -------------------------
aa_ptr = 0
k = 0
col = 0

for i, res in enumerate(sequence):
    n_atoms, groups = no_Atoms_AA[res]
    end_ptr = aa_ptr + 3 * n_atoms

    if n_atoms == 0:
        aa_ptr = end_ptr
        continue

    residue_atoms = all_atoms[:, aa_ptr:end_ptr]  # (N, 3*n_atoms)

    if isinstance(groups[0], int):
        res_i = bead_to_res[k]
        R = R_all[:, res_i, :, :]
        bead_xyz = bead_xyz_all[:, k, :]

        norm = normalize_atoms_localframe(residue_atoms, bead_xyz, R, delta_range)

        pad = 15 - norm.shape[1]
        if pad < 0:
            raise ValueError(f"Group wider than 15 at residue {res} index {i} (width={norm.shape[1]})")

        out = slice(col, col + 15)
        final_arr[:, out] = np.pad(norm, ((0,0),(0,pad)), constant_values=0)
        masking_arr[:, out] = np.pad(norm, ((0,0),(0,pad)), constant_values=-2)

        k += 1
        col += 15

    else:
        for bead_group in groups:
            res_i = bead_to_res[k]
            R = R_all[:, res_i, :, :]
            bead_xyz = bead_xyz_all[:, k, :]

            if "," in bead_group:
                idxs = [int(x) for x in bead_group.split(",")]
                atoms_slice = residue_atoms[:, idxs]
            elif "_" in bead_group:
                a, b = [int(x) for x in bead_group.split("_")]
                atoms_slice = residue_atoms[:, a:b]
            else:
                raise ValueError(f"Unexpected bead_group format: {bead_group}")

            norm = normalize_atoms_localframe(atoms_slice, bead_xyz, R, delta_range)

            pad = 15 - norm.shape[1]
            if pad < 0:
                raise ValueError(
                    f"Group wider than 15 at residue {res} index {i}, group={bead_group} (width={norm.shape[1]})"
                )

            out = slice(col, col + 15)
            final_arr[:, out] = np.pad(norm, ((0,0),(0,pad)), constant_values=0)
            masking_arr[:, out] = np.pad(norm, ((0,0),(0,pad)), constant_values=-2)

            k += 1
            col += 15

    aa_ptr = end_ptr

if k != M:
    raise ValueError(f"Consumed bead-groups k={k}, expected M={M}.")
if col != 15 * M:
    raise ValueError(f"Output col={col}, expected {15*M}.")

# -------------------------
# Save
# -------------------------
out_file = f"cluster_PD_{cluster_id}_SC_LocalFrame.npy"
mask_file = f"masking_input_{cluster_id}.npy"

np.save(out_file, final_arr)
np.save(mask_file, masking_arr)

print("Saved:", out_file, final_arr.shape, "min/max:", float(final_arr.min()), float(final_arr.max()))
print("Saved:", mask_file, masking_arr.shape, "min/max:", float(masking_arr.min()), float(masking_arr.max()))
print("Used delta_range:", delta_range, "delta_half_range:", delta_half_range)
print("Clipping enabled [0,1]:", CLIP_TO_01)

# Diagnostics: report out-of-range percentage on valid (non-padding) coordinates.
valid_mask = (masking_arr != -2)
valid_vals = final_arr[valid_mask]
if valid_vals.size > 0:
    below0 = np.sum(valid_vals < 0.0)
    above1 = np.sum(valid_vals > 1.0)
    outside = below0 + above1
    total = valid_vals.size
    print(
        "Valid coords outside [0,1]:",
        f"{outside}/{total} ({100.0 * outside / total:.3f}%)",
        f"| below0: {below0} ({100.0 * below0 / total:.3f}%)",
        f"| above1: {above1} ({100.0 * above1 / total:.3f}%)",
    )
else:
    print("Valid coords outside [0,1]: no valid coordinates found.")

print("\nUndo later (per atom vector):")
print("  d_local = (atom_norm - 0.5) * delta_range")
print("  atom_xyz = bead_xyz + R @ d_local")
print("Where bead_xyz is from CG beads, and R = R_all[:, bead_to_res[k]].")
