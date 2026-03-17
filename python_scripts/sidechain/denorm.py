#!/usr/bin/env python3
import numpy as np
import sys
import re

"""
UNPAD + DENORMALISE for Option B local-frame normalisation.

Inputs:
  argv[1] pred_or_norm_padded.npy : (N, 15*M) predicted/normalised+padded array
  argv[2] sequence_file.txt       : comma-separated 3-letter sequence (same as sequence_{pdb}.txt)
  argv[3] cg_bead_file.npy        : (N, 3*M) CG sidechain bead coords (same ordering as your normaliser)
  argv[4] mask_file.npy           : (N, 15*M) mask array with -2 at padded positions (from your normaliser)
  argv[5] pdb_name                : e.g. 1J4N
  argv[6] cluster_id              : integer

Optional:
  argv[7] out_file.npy            : output filename (default: reversed_localframe_{cluster_id}.npy)
  argv[8] keep_group_idx.npy      : optional 1D group index file from filtered run_model output

Outputs:
  out_file.npy : (N, 3*total_atoms) all-atom coords in ORIGINAL all-atom ordering
"""

# -------------------------
# Parse args
# -------------------------
pred_file   = sys.argv[1]
seq_file    = sys.argv[2]
cg_file     = sys.argv[3]
mask_file   = sys.argv[4]
pdb         = sys.argv[5]
cluster_id  = int(sys.argv[6])
out_file    = sys.argv[7] if len(sys.argv) >= 8 else f"reversed_localframe_{cluster_id}.npy"
keep_idx_file = sys.argv[8] if len(sys.argv) >= 9 else None

# -------------------------
# Load sequence
# -------------------------
with open(seq_file, "r") as f:
    sequence = [x for x in f.read().strip().split(",") if x]
L = len(sequence)
print("Sequence length:", L)

# ==========================================================
# CG bead layout map (MUST match cg_bead_file ordering)
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
# All-atom grouping map (MUST match your normaliser ordering)
# ==========================================================
no_Atoms_AA = {
 'CYS':[2,[0]],
 'ALA':[1,[0]],
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
 'PHE':[7,['0,1,2,3,4,5,6,7,8','9,10,11,12,13,14','15,16,17,18,19,20']],  # fixed typo
 'PRO':[3,[0]],
 'SER':[2,[0]],
 'THR':[3,[0]],
 'TRP':[10,['0,1,2,3,4,5,27,28,29','6,7,8,9,10,11,12,13,14','21,22,23,24,25,26','15,16,17,18,19,20']],
 'TYR':[8,['0,1,2,3,4,5,6,7,8','9,10,11,12,13,14,15,16,17','18,19,20,21,22,23']],
 'VAL':[3,[0]]
}

# -------------------------
# Load arrays
# -------------------------
pred = np.load(pred_file)
mask = np.load(mask_file)
cg_beads = np.load(cg_file)

# Handle flattened pred/mask
if pred.ndim == 1 and mask.ndim == 2:
    pred = pred.reshape(mask.shape[0], -1)
if mask.ndim == 1 and pred.ndim == 2:
    mask = mask.reshape(pred.shape[0], -1)

# If prediction is filtered (fewer group-columns), expand back to full width.
if pred.ndim == 2 and mask.ndim == 2 and pred.shape[1] != mask.shape[1]:
    if keep_idx_file is None:
        raise ValueError(
            f"pred/mask width mismatch: pred={pred.shape}, mask={mask.shape}. "
            "Provide keep_group_idx.npy as argv[8] (from run_model filter output) "
            "or run without residue filtering."
        )

    keep_idx = np.load(keep_idx_file)
    if keep_idx.ndim != 1:
        raise ValueError(f"keep idx must be 1D, got shape {keep_idx.shape}")

    if mask.shape[1] % 15 != 0:
        raise ValueError(f"mask width must be divisible by 15, got {mask.shape[1]}")
    full_groups = mask.shape[1] // 15

    if keep_idx.dtype == np.bool_:
        if keep_idx.size != full_groups:
            raise ValueError(
                f"bool keep idx length {keep_idx.size} != full_groups {full_groups}"
            )
        keep_idx = np.where(keep_idx)[0].astype(np.int32)
    else:
        keep_idx = keep_idx.astype(np.int32)

    if np.any(keep_idx < 0) or np.any(keep_idx >= full_groups):
        raise ValueError("keep idx contains out-of-range group indices")

    expected_pred_width = keep_idx.size * 15
    if pred.shape[1] != expected_pred_width:
        raise ValueError(
            f"pred width {pred.shape[1]} does not match keep_idx groups ({keep_idx.size}) * 15 = {expected_pred_width}"
        )

    pred_full = np.full((pred.shape[0], full_groups * 15), 0.5, dtype=np.float32)
    for j, g in enumerate(keep_idx):
        pred_full[:, g * 15:(g + 1) * 15] = pred[:, j * 15:(j + 1) * 15]
    pred = pred_full
    print(
        "Expanded filtered prediction using keep idx:",
        keep_idx_file,
        "->",
        pred.shape,
    )

N = pred.shape[0]
if mask.shape[0] != N:
    raise ValueError(f"N mismatch: pred {pred.shape}, mask {mask.shape}")
if cg_beads.shape[0] != N:
    raise ValueError(f"N mismatch: pred N={N}, cg_beads N={cg_beads.shape[0]}")

# -------------------------
# Unpad using mask == -2 convention (same logic as your old script)
# -------------------------
valid = (mask != -2)
col_mask = valid.any(axis=0)          # columns that are ever non-pad
pred_unpadded = pred[:, col_mask].astype(np.float32)

print("pred padded shape :", pred.shape)
print("pred unpadded shape:", pred_unpadded.shape)

# -------------------------
# Load R frames
# -------------------------
R_path = f"R_localFrame_{pdb}_cluster{cluster_id}.npy"
R_all = np.load(R_path, mmap_mode="r")  # (N, L, 3, 3)
if R_all.shape[0] != N or R_all.shape[1] != L:
    raise ValueError(f"R_all shape mismatch: got {R_all.shape}, expected ({N},{L},3,3)")

# -------------------------
# Load meta for delta_range
# -------------------------
delta_range = 7.0
meta_path = f"localFrame_META_{pdb}_cluster{cluster_id}.npz"
try:
    meta = np.load(meta_path, allow_pickle=True)
    if "delta_range" in meta.files:
        delta_range = float(meta["delta_range"][0])
except FileNotFoundError:
    print(f"WARNING: meta not found: {meta_path} (using delta_range=7.0)")

# -------------------------
# Infer bead count M + build bead_to_res/bead_to_num
# -------------------------
if cg_beads.shape[1] % 3 != 0:
    raise ValueError("cg_beads second dim must be divisible by 3")
M_from_file = cg_beads.shape[1] // 3

expected_M = sum(no_Atoms_CG[res][0] for res in sequence)
if expected_M != M_from_file:
    raise ValueError(f"CG M mismatch: cg_file has {M_from_file}, sequence+map expects {expected_M}")

M = expected_M

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
    raise RuntimeError(f"bead_to_res filled {k}, expected {M}")

# -------------------------
# Build bead_xyz_all (N, M, 3) same as forward script
# -------------------------
bead_xyz_all = np.empty((N, M, 3), dtype=np.float32)
cg_ptr = 0
k = 0
for i, res in enumerate(sequence):
    n_beads = no_Atoms_CG[res][0]
    if n_beads == 0:
        continue

    block = cg_beads[:, cg_ptr:cg_ptr + 3*n_beads].astype(np.float32)
    cg_ptr += 3*n_beads

    if n_beads == 1:
        bead_xyz_all[:, k, :] = block.reshape(N, 3)
        k += 1
    else:
        splicer_list = no_Atoms_CG[res][1]
        for sp_str in splicer_list:
            sp = [int(x) for x in sp_str.split(",")]
            bead_xyz_all[:, k, :] = block[:, sp]
            k += 1

if k != M:
    raise ValueError(f"Filled bead_xyz k={k}, expected M={M}")

# -------------------------
# Denormalise helper
# -------------------------
def denorm_localframe(norm_flat, bead_xyz, R, delta_range):
    # norm_flat: (N, 3*n_atoms_group)
    norm = norm_flat.reshape(N, -1, 3).astype(np.float32)
    d_local = (norm - 0.5) * delta_range
    # d_global = R @ d_local
    d_global = np.einsum("nij,nkj->nki", R.astype(np.float32), d_local)
    atoms = bead_xyz[:, None, :].astype(np.float32) + d_global
    return atoms.reshape(norm_flat.shape).astype(np.float32)

# -------------------------
# Rebuild ORIGINAL all-atom flat ordering
# -------------------------
total_atoms = sum(no_Atoms_AA[res][0] for res in sequence)
all_atoms_out = np.zeros((N, 3 * total_atoms), dtype=np.float32)

ptr = 0          # pointer into pred_unpadded
aa_ptr = 0       # pointer into all_atoms_out
k = 0            # bead-group index (0..M-1)

for i, res in enumerate(sequence):
    n_atoms, groups = no_Atoms_AA[res]
    end_aa = aa_ptr + 3*n_atoms

    if n_atoms == 0:
        aa_ptr = end_aa
        continue

    residue_out = np.zeros((N, 3*n_atoms), dtype=np.float32)

    if isinstance(groups[0], int):
        # single group: entire residue is one bead-group
        ncoords = 3 * n_atoms
        norm_slice = pred_unpadded[:, ptr:ptr+ncoords]
        ptr += ncoords

        res_i = bead_to_res[k]
        R = R_all[:, res_i, :, :]
        bead_xyz = bead_xyz_all[:, k, :]

        atoms_flat = denorm_localframe(norm_slice, bead_xyz, R, delta_range)
        residue_out[:, :] = atoms_flat

        k += 1

    else:
        # multi-group residue: fill indices back into residue_out
        for bead_group in groups:
            if "," in bead_group:
                idxs = [int(x) for x in bead_group.split(",")]
            elif "_" in bead_group:
                a, b = [int(x) for x in bead_group.split("_")]
                idxs = list(range(a, b))
            else:
                raise ValueError(f"Unexpected bead_group format: {bead_group}")

            ncoords = len(idxs)
            norm_slice = pred_unpadded[:, ptr:ptr+ncoords]
            ptr += ncoords

            res_i = bead_to_res[k]
            R = R_all[:, res_i, :, :]
            bead_xyz = bead_xyz_all[:, k, :]

            atoms_flat = denorm_localframe(norm_slice, bead_xyz, R, delta_range)
            residue_out[:, idxs] = atoms_flat

            k += 1

    all_atoms_out[:, aa_ptr:end_aa] = residue_out
    aa_ptr = end_aa

# -------------------------
# Sanity checks + save
# -------------------------
if ptr != pred_unpadded.shape[1]:
    raise ValueError(f"Did not consume all unpadded cols: ptr={ptr}, total={pred_unpadded.shape[1]}")
if k != M:
    raise ValueError(f"Did not consume all bead-groups: k={k}, M={M}")

np.save(out_file, all_atoms_out)
print("Saved:", out_file, all_atoms_out.shape, "min/max:", float(all_atoms_out.min()), float(all_atoms_out.max()))
print("Used delta_range:", delta_range)
