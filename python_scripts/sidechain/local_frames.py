#!/usr/bin/env python3
import numpy as np
import sys
import re
from numpy.lib.format import open_memmap

"""
LOCAL-FRAME FEATURE SCRIPT (handles GLY-missing SC sequence, saves R/O)

UPDATED: adds extra features (NO normalisation):
  - bead_num   : 0..3 (bead index within residue)
  - residue_id : integer ID using your mapping (0..18)

Current output feature dimension is 38.

Inputs:
  argv[1] cluster_file    : arr_data .npy (sidechain bead coords, ordered by SC sequence)
  argv[2] pdb_name        : used to load sequence_{pdb}.txt  (SC sequence, GLY removed)
  argv[3] backbone_file   : bb_data .npy  (N, 3*L_bb) backbone beads INCLUDING GLY

Optional positional arguments:
  - If L_bb != L_sc: alignment source (required), either:
      * sc_to_bb mapping .npy (shape (L_sc,), int indices into backbone residues), OR
      * full sequence .txt containing ALL residues including GLY, comma-separated 3-letter codes
  - segment_starts in BACKBONE indexing, comma-separated, e.g. "0,747,1494,2241"
    (default: "0,747,1494,2241")

Optional flags:
  --default-buffer <float>   buffer for non-aromatic residues when exporting custom_min/custom_range
  --aromatic-buffer <float>  buffer for TYR/PHE/TRP when exporting custom_min/custom_range
  --save-local-frames        also save R_localFrame_* and O_localFrame_* (disabled by default)

Outputs:
  - cluster_{cluster}_SC_CG_RBF_localFrameFeatures.npy  (N*total_beads, 38)
  - custom_range_{pdb}_{cluster}.npy                    (N, 3*total_beads)
  - custom_min_{pdb}_{cluster}.npy                      (N, 3*total_beads)
  - R_localFrame_{pdb}_cluster{cluster}.npy             (N, L_sc, 3, 3) [only with --save-local-frames]
  - O_localFrame_{pdb}_cluster{cluster}.npy             (N, L_sc, 3)    [only with --save-local-frames]
  - localFrame_META_{pdb}_cluster{cluster}.npz          (mapping arrays + segment info)

Later (all-atoms target normalization for an atom attached to bead k):
  sc_i = bead_to_scres[k]
  R = R_sc[frame, sc_i]                # (3,3) local->global
  bead_xyz = your bead coord (global)  # (3,)
  atom_norm = 0.5 + (R^T @ (atom_xyz - bead_xyz)) / 7.0

Undo:
  atom_xyz = bead_xyz + R @ ((atom_norm - 0.5) * 7.0)
"""


if len(sys.argv) < 4:
    raise SystemExit(
        "Usage: python local_frames.py <cluster_CG_SC.npy> <PDB> <backbone.npy> "
        "[alignment_source_if_needed] [segment_starts] [--default-buffer FLOAT] [--aromatic-buffer FLOAT] [--save-local-frames]"
    )


def parse_optional_args(argv_tail):
    positional = []
    default_buffer = 2.3
    aromatic_buffer = 3.5
    save_local_frames = False

    i = 0
    while i < len(argv_tail):
        tok = argv_tail[i]
        if tok == "--default-buffer":
            if i + 1 >= len(argv_tail):
                raise ValueError("--default-buffer requires a value")
            default_buffer = float(argv_tail[i + 1])
            i += 2
        elif tok == "--aromatic-buffer":
            if i + 1 >= len(argv_tail):
                raise ValueError("--aromatic-buffer requires a value")
            aromatic_buffer = float(argv_tail[i + 1])
            i += 2
        elif tok == "--save-local-frames":
            save_local_frames = True
            i += 1
        else:
            positional.append(tok)
            i += 1

    if default_buffer <= 0.0 or aromatic_buffer <= 0.0:
        raise ValueError("default/aromatic buffers must be > 0.")

    return positional, default_buffer, aromatic_buffer, save_local_frames

# ----------------------------
# Your residue-id mapping (inverse of INT_TO_AA)
# ----------------------------
AA_TO_INT = {
    'ALA': 0,  'ARG': 1,  'ASN': 2,  'ASP': 3,  'CYS': 4,
    'GLN': 5,  'GLU': 6,  'HIS': 7,  'ILE': 8,  'LEU': 9,
    'LYS': 10, 'MET': 11, 'PHE': 12, 'PRO': 13, 'SER': 14,
    'THR': 15, 'TRP': 16, 'TYR': 17, 'VAL': 18
}

# ----------------------------
# Load SC sequence (GLY removed)
# ----------------------------
pdb_name = sys.argv[2]
with open(f"sequence_{pdb_name}.txt", "r") as f:
    sc_sequence = f.readline().strip().split(",")

L_sc = len(sc_sequence)

# ----------------------------
# Load arrays
# ----------------------------
cluster_file = sys.argv[1]
backbone_file = sys.argv[3]

arr_data = np.load(cluster_file).astype(np.float32)
bb_data  = np.load(backbone_file).astype(np.float32)

N = arr_data.shape[0]
if bb_data.shape[0] != N:
    raise ValueError(f"N mismatch: arr_data has {N}, bb_data has {bb_data.shape[0]}")

if bb_data.shape[1] % 3 != 0:
    raise ValueError(f"bb_data.shape[1] must be multiple of 3, got {bb_data.shape[1]}")

L_bb = bb_data.shape[1] // 3
optional_positional, default_buffer, aromatic_buffer, save_local_frames = parse_optional_args(sys.argv[4:])

align_src = None
segment_starts_str = None
if L_bb == L_sc:
    if len(optional_positional) > 2:
        raise ValueError(
            "Too many positional args. With L_bb == L_sc, provide at most [alignment_source] [segment_starts]."
        )
    if len(optional_positional) >= 1:
        first = optional_positional[0]
        if re.fullmatch(r"[0-9]+(?:,[0-9]+)*", first):
            segment_starts_str = first
        else:
            align_src = first  # accepted for backward compatibility, not required when L_bb == L_sc
    if len(optional_positional) == 2:
        segment_starts_str = optional_positional[1]
else:
    if len(optional_positional) == 0:
        raise ValueError(
            f"bb length (L_bb={L_bb}) != sc length (L_sc={L_sc}).\n"
            "Provide alignment source (.npy mapping or full-sequence .txt including GLY)."
        )
    if len(optional_positional) > 2:
        raise ValueError(
            "Too many positional args. Use [alignment_source] [segment_starts] optionally."
        )
    align_src = optional_positional[0]
    if len(optional_positional) == 2:
        segment_starts_str = optional_positional[1]

# ----------------------------
# Parse cluster number
# ----------------------------
pattern = r"cluster_(\d+)_CG_SC\.npy"
m = re.search(pattern, cluster_file)
if m is None:
    raise ValueError(f"Could not parse cluster number from: {cluster_file}")
cluster_number = int(m.group(1))

# ----------------------------
# Segment starts/ends are in BACKBONE indexing
# ----------------------------
if segment_starts_str is not None:
    segment_starts = [int(x) for x in segment_starts_str.split(",") if x.strip() != ""]
else:
    segment_starts = [0, 747, 1494, 2241]

segment_starts = sorted(set([s for s in segment_starts if 0 <= s < L_bb]))
if 0 not in segment_starts:
    segment_starts = [0] + segment_starts
segment_starts = sorted(set(segment_starts))

segment_ends = []
for idx, s in enumerate(segment_starts):
    if idx < len(segment_starts) - 1:
        segment_ends.append(segment_starts[idx + 1] - 1)
    else:
        segment_ends.append(L_bb - 1)

start_set = set(segment_starts)
end_set   = set(segment_ends)

# ----------------------------
# Build sc_to_bb mapping
# ----------------------------
def load_full_sequence_txt(path: str):
    with open(path, "r") as f:
        return f.readline().strip().split(",")

def build_sc_to_bb_from_full(full_seq, sc_seq):
    """
    full_seq includes GLY; sc_seq excludes GLY.
    We map each element of sc_seq to the backbone index in full_seq
    by taking all non-GLY indices in order and verifying residue names match.
    """
    non_gly_idx = [i for i, r in enumerate(full_seq) if r != "GLY"]
    non_gly_res = [full_seq[i] for i in non_gly_idx]

    if len(non_gly_res) != len(sc_seq):
        raise ValueError(
            f"Alignment failed: non-GLY count in full sequence ({len(non_gly_res)}) "
            f"!= SC sequence length ({len(sc_seq)})."
        )

    for j, (a, b) in enumerate(zip(non_gly_res, sc_seq)):
        if a != b:
            raise ValueError(
                f"Alignment failed at SC index {j}: full_seq non-GLY residue '{a}' != sc_seq '{b}'."
            )

    return np.array(non_gly_idx, dtype=np.int32)

# If L_bb == L_sc, mapping is trivial; else use provided alignment source
if L_bb == L_sc:
    sc_to_bb = np.arange(L_sc, dtype=np.int32)
else:
    if align_src.endswith(".npy"):
        sc_to_bb = np.load(align_src).astype(np.int32)
        if sc_to_bb.shape[0] != L_sc:
            raise ValueError(f"sc_to_bb.npy length {sc_to_bb.shape[0]} != L_sc {L_sc}")
        if sc_to_bb.min() < 0 or sc_to_bb.max() >= L_bb:
            raise ValueError("sc_to_bb contains indices outside backbone range.")
    else:
        full_seq = load_full_sequence_txt(align_src)
        if len(full_seq) != L_bb:
            raise ValueError(f"Full sequence length {len(full_seq)} != L_bb {L_bb} from bb_data.")
        sc_to_bb = build_sc_to_bb_from_full(full_seq, sc_sequence)

# ----------------------------
# AA maps / BLOSUM (same as yours)
# ----------------------------
three_to_one = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V',
    'TRP': 'W', 'TYR': 'Y'
}

BLOSUM_60 = {
"A": [4, -1, -1, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0] ,
"R": [-1, 5, 0, -1, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -2] ,
"N": [-1, 0, 6, 1, -2, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3] ,
"D": [-2, -1, 1, 6, -3, 0, 2, -1, -1, -3, -3, -1, -3, -3, -1, 0, -1, -4, -3, -3] ,
"C": [0, -3, -2, -3, 9, -3, -3, -2, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1] ,
"Q": [-1, 1, 0, 0, -3, 5, 2, -2, 1, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2] ,
"E": [-1, 0, 0, 2, -3, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2] ,
"G": [0, -2, 0, -1, -2, -2, -2, 6, -2, -3, -4, -1, -2, -3, -2, 0, -2, -2, -3, -3] ,
"H": [-2, 0, 1, -1, -3, 1, 0, -2, 7, -3, -3, -1, -1, -1, -2, -1, -2, -2, 2, -3] ,
"I": [-1, -3, -3, -3, -1, -3, -3, -3, -3, 4, 2, -3, 1, 0, -3, -2, -1, -2, -1, 3] ,
"L": [-1, -2, -3, -3, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1] ,
"K": [-1, 2, 0, -1, -3, 1, 1, -1, -1, -3, -2, 4, -1, -3, -1, 0, -1, -3, -2, -2] ,
"M": [-1, -1, -2, -3, -1, 0, -2, -2, -1, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1] ,
"F": [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1] ,
"P": [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2] ,
"S": [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2] ,
"T": [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 4, -2, -2, 0] ,
"W": [-3, -3, -4, -4, -2, -2, -3, -2, -2, -2, -2, -3, -1, 1, -4, -3, -2, 10, 2, -3] ,
"Y": [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 6, -1] ,
"V": [0, -2, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4]
}

def normalize_score(score: float) -> float:
    return (score + 4) / 14

for k, v in BLOSUM_60.items():
    BLOSUM_60[k] = [normalize_score(i) for i in v]

no_Atoms= {
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

AROMATIC_RESIDUES = {"TYR", "PHE", "TRP"}


def get_residue_buffer(res_name: str) -> float:
    if res_name in AROMATIC_RESIDUES:
        return aromatic_buffer
    return default_buffer

# ----------------------------
# Geometry helpers
# ----------------------------
def _normalize_vec(v, eps=1e-8):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + eps)

def calculate_bond_angles_vectorized_normalized(A, B, C, eps=1e-8):
    v1 = A - B
    v2 = C - B
    dot = np.sum(v1 * v2, axis=1)
    n1 = np.linalg.norm(v1, axis=1) + eps
    n2 = np.linalg.norm(v2, axis=1) + eps
    cos_theta = dot / (n1 * n2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angles = np.degrees(np.arccos(cos_theta))
    return (angles / 180.0).reshape(-1, 1)

def calculate_rbf(array1, array2, gamma=0.1):
    squared_distances = np.sum((array1 - array2) ** 2, axis=1)
    return np.exp(-gamma * squared_distances).reshape(-1, 1)

def _angle_rad(A, B, C, eps=1e-8):
    v1 = A - B
    v2 = C - B
    dot = np.sum(v1 * v2, axis=1)
    n1 = np.linalg.norm(v1, axis=1) + eps
    n2 = np.linalg.norm(v2, axis=1) + eps
    cos_theta = np.clip(dot / (n1 * n2), -1.0, 1.0)
    return np.arccos(cos_theta)

def _dihedral_rad(P0, P1, P2, P3, eps=1e-8):
    b0 = P1 - P0
    b1 = P2 - P1
    b2 = P3 - P2
    b1u = b1 / (np.linalg.norm(b1, axis=1, keepdims=True) + eps)
    v = b0 - np.sum(b0 * b1u, axis=1, keepdims=True) * b1u
    w = b2 - np.sum(b2 * b1u, axis=1, keepdims=True) * b1u
    x = np.sum(v * w, axis=1)
    y = np.sum(np.cross(b1u, v) * w, axis=1)
    return np.arctan2(y, x)

def compute_bb_shape_features(bb_xyz, bb_i_idx, seg_start, seg_end):
    i_m1 = max(bb_i_idx - 1, seg_start)
    i_p1 = min(bb_i_idx + 1, seg_end)
    i_p2 = min(bb_i_idx + 2, seg_end)

    p_m1 = bb_xyz[:, i_m1, :]
    p_0  = bb_xyz[:, bb_i_idx, :]
    p_p1 = bb_xyz[:, i_p1, :]
    p_p2 = bb_xyz[:, i_p2, :]

    kappa = _angle_rad(p_m1, p_0, p_p1)
    tau = _dihedral_rad(p_m1, p_0, p_p1, p_p2)

    kappa_sin = np.clip(0.5 + 0.5 * np.sin(kappa), 0.0, 1.0).reshape(-1, 1).astype(np.float32)
    kappa_cos = np.clip(0.5 + 0.5 * np.cos(kappa), 0.0, 1.0).reshape(-1, 1).astype(np.float32)
    tau_sin = np.clip(0.5 + 0.5 * np.sin(tau), 0.0, 1.0).reshape(-1, 1).astype(np.float32)
    tau_cos = np.clip(0.5 + 0.5 * np.cos(tau), 0.0, 1.0).reshape(-1, 1).astype(np.float32)
    return np.concatenate([kappa_sin, kappa_cos, tau_sin, tau_cos], axis=1)

def compute_packing_features(c_point, bb_window):
    if bb_window is None or bb_window.shape[1] == 0:
        return np.zeros((c_point.shape[0], 4), dtype=np.float32)
    d = np.linalg.norm(bb_window - c_point[:, None, :], axis=2)
    f6 = np.mean(d <= 6.0, axis=1, keepdims=True).astype(np.float32)
    f8 = np.mean(d <= 8.0, axis=1, keepdims=True).astype(np.float32)
    f10 = np.mean(d <= 10.0, axis=1, keepdims=True).astype(np.float32)
    dmin = np.min(d, axis=1, keepdims=True)
    dmin_scaled = np.clip(dmin, 0.0, 20.0) / 20.0
    return np.clip(np.concatenate([f6, f8, f10, dmin_scaled.astype(np.float32)], axis=1), 0.0, 1.0).astype(np.float32)

# ----------------------------
# Robust local frame builder
# ----------------------------
def build_local_frame(bb_im1, bb_i, bb_ip1, has_im1: bool, has_ip1: bool):
    """
    Returns:
      R: (N,3,3) columns are [t,b,n] in GLOBAL coords (local->global)
      O: (N,3) origin = bb_i
    """
    O = bb_i

    if has_im1 and has_ip1:
        t = _normalize_vec(bb_ip1 - bb_im1)
    elif has_ip1:
        t = _normalize_vec(bb_ip1 - bb_i)
    else:
        t = _normalize_vec(bb_i - bb_im1)

    ux = np.array([1.0, 0.0, 0.0], dtype=np.float32)[None, :]
    uy = np.array([0.0, 1.0, 0.0], dtype=np.float32)[None, :]
    uz = np.array([0.0, 0.0, 1.0], dtype=np.float32)[None, :]

    dotx = np.abs(np.sum(t * ux, axis=1, keepdims=True))
    u = np.where(dotx < 0.9, ux, uy)
    dotu = np.abs(np.sum(t * u, axis=1, keepdims=True))
    u = np.where(dotu < 0.9, u, uz)

    n = _normalize_vec(np.cross(t, u))
    b = _normalize_vec(np.cross(n, t))
    n = _normalize_vec(np.cross(t, b))

    R = np.stack([t, b, n], axis=-1)
    return R, O


def safe_unit(v, eps=1e-8):
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + eps)



def pick_backbone_neighbor(bb_im1, bb_i, bb_ip1, has_im1, has_ip1):
    if has_im1:
        return bb_im1
    if has_ip1:
        return bb_ip1
    return bb_i

# ----------------------------
# Local coordinate feature mapping
# ----------------------------
DELTA_HALF_RANGE = 3.5
DELTA_RANGE = 2.0 * DELTA_HALF_RANGE  # 7.0

def local_coord_feature(point_global, O, R):
    d_global = point_global - O
    d_local = np.einsum("nij,nj->ni", R.transpose(0,2,1), d_global)  # R^T * d
    return 0.5 + d_local / DELTA_RANGE

# ----------------------------
# Count total beads in SC sequence
# ----------------------------
total_beads = 0
for res in sc_sequence:
    total_beads += no_Atoms[res][0]

# Backbone xyz as (N, L_bb, 3), computed once
bb_xyz = bb_data.reshape(N, L_bb, 3)

# Feature output: one row per bead per frame
# UPDATED: 37 -> 38 (adds 1 BB-RBF feature after current scalar/RBF col 20)
FEATURE_DIM = 38
feat_out = np.empty((N * total_beads, FEATURE_DIM), dtype=np.float32)
custom_range_arr = np.empty((N, 3 * total_beads), dtype=np.float32)
custom_min_arr = np.empty((N, 3 * total_beads), dtype=np.float32)

# Mapping for later (still saved)
bead_to_scres = np.empty((total_beads,), dtype=np.int32)  # which SC residue
bead_to_num   = np.empty((total_beads,), dtype=np.int32)  # bead index within residue
bead_to_bbres = np.empty((total_beads,), dtype=np.int32)  # backbone residue index (via sc_to_bb)

# Save R and O per SC residue (optional)
R_file = f"R_localFrame_{pdb_name}_cluster{cluster_number}.npy"
O_file = f"O_localFrame_{pdb_name}_cluster{cluster_number}.npy"
R_sc = None
O_sc = None
if save_local_frames:
    R_sc = open_memmap(R_file, mode="w+", dtype=np.float32, shape=(N, L_sc, 3, 3))
    O_sc = open_memmap(O_file, mode="w+", dtype=np.float32, shape=(N, L_sc, 3))

# ----------------------------
# Main loop over SC residues (aligned to backbone by sc_to_bb)
# ----------------------------
start_indx = 0
bead_counter = 0
stats_col = 0

for sc_i in range(L_sc):
    resname = sc_sequence[sc_i]
    n_beads = no_Atoms[resname][0]

    # residue id feature (NO normalization)
    if resname not in AA_TO_INT:
        raise KeyError(f"Residue {resname} not found in AA_TO_INT mapping.")
    res_id_val = float(AA_TO_INT[resname])  # store as float32 later
    residue_buffer = get_residue_buffer(resname)

    bb_i_idx = int(sc_to_bb[sc_i])
    bb_indx = 3 * bb_i_idx
    seg_idx = np.searchsorted(segment_starts, bb_i_idx, side="right") - 1
    seg_start = segment_starts[seg_idx]
    seg_end = segment_ends[seg_idx]

    is_start = (bb_i_idx in start_set)
    is_end   = (bb_i_idx in end_set)
    has_im1  = not is_start
    has_ip1  = not is_end

    bb_i = bb_data[:, bb_indx:bb_indx+3]
    bb_im1 = bb_data[:, bb_indx-3:bb_indx] if has_im1 else bb_i
    bb_ip1 = bb_data[:, bb_indx+3:bb_indx+6] if has_ip1 else bb_i

    R, O = build_local_frame(bb_im1, bb_i, bb_ip1, has_im1=has_im1, has_ip1=has_ip1)

    if save_local_frames:
        R_sc[:, sc_i, :, :] = R
        O_sc[:, sc_i, :] = O

    aa1 = three_to_one[resname]
    B_score = np.array(BLOSUM_60[aa1], dtype=np.float32)
    bb_shape_feat = compute_bb_shape_features(bb_xyz, bb_i_idx, seg_start, seg_end)  # (N,4)
    n_beads_scaled = np.full((N, 1), float(n_beads) / 4.0, dtype=np.float32)          # (N,1)

    w = 16
    win_start = max(seg_start, bb_i_idx - w)
    win_end = min(seg_end, bb_i_idx + w)
    neigh_idx = [j for j in range(win_start, win_end + 1) if j != bb_i_idx]
    bb_window = bb_xyz[:, neigh_idx, :] if len(neigh_idx) > 0 else None

    end_indx = start_indx + (n_beads * 3)
    if n_beads == 0:
        start_indx = end_indx
        continue

    residue_arr = arr_data[:, start_indx:end_indx]
    bb_nei = pick_backbone_neighbor(bb_im1, bb_i, bb_ip1, has_im1, has_ip1)

    if n_beads == 1:
        scalar = float(no_Atoms[resname][2][0])
        bead_info = np.concatenate([B_score, np.array([scalar], dtype=np.float32)], axis=0)  # (21,)
        expanded = np.tile(bead_info[None, :], (N, 1))  # (N,21)

        c_point = residue_arr.reshape(N, 3)
        custom_min_arr[:, stats_col : stats_col + 3] = c_point - residue_buffer
        custom_range_arr[:, stats_col : stats_col + 3] = (2.0 * residue_buffer)
        stats_col += 3

        if resname == "ALA":
            # backbone-only angle (varies with backbone)
            bond_angle = calculate_bond_angles_vectorized_normalized(bb_im1, bb_i, bb_ip1)  # (N,1)

            # use a backbone axis from R in GLOBAL coords and map to [0,1]
            axis = R[:, :, 2]                 # n_global; try R[:,:,1] if you prefer b_global
            axis = safe_unit(axis)
            coord_feat = np.clip(0.5 + 0.5 * axis, 0.0, 1.0).astype(np.float32)
        else:
            bond_angle = calculate_bond_angles_vectorized_normalized(bb_nei, bb_i, c_point)
            coord_feat = local_coord_feature(c_point, O, R)
        bb_rbf = calculate_rbf(bb_i, c_point)  # (N,1) BB-to-current-bead RBF

#--------------------------------------------------------------------------------------------------------------------

      #  bond_angle = calculate_bond_angles_vectorized_normalized(bb_nei, bb_i, c_point)  # (N,1)
      #  coord_feat = local_coord_feature(c_point, O, R)  # (N,3)

        bead_num_feat = np.zeros((N, 1), dtype=np.float32)          # single-bead => 0
        res_id_feat   = np.full((N, 1), res_id_val, dtype=np.float32)
        # NEW features (NO normalization):
        bead_fraction_feat = np.zeros((N, 1), dtype=np.float32)        # single-bead => 0
        packing_feat = compute_packing_features(c_point, bb_window)     # (N,4)
        new_cont_feat = np.concatenate(
            [bb_shape_feat, packing_feat, n_beads_scaled, bead_fraction_feat],
            axis=1
        )  # (N,10)

        rows = slice(bead_counter * N, (bead_counter + 1) * N)
        feat_out[rows, :] = np.concatenate(
            [expanded, bb_rbf, bond_angle, coord_feat, new_cont_feat, bead_num_feat, res_id_feat],
            axis=1
        )

        bead_to_scres[bead_counter] = sc_i
        bead_to_num[bead_counter] = 0
        bead_to_bbres[bead_counter] = bb_i_idx
        bead_counter += 1

    else:
        splicer_list = no_Atoms[resname][1]
        c_prev = None
        A = None
        B = None

        for num, splicer_str in enumerate(splicer_list):
            splicer = [int(j) for j in splicer_str.split(",")]
            c_point = residue_arr[:, splicer]
            custom_min_arr[:, stats_col : stats_col + 3] = c_point - residue_buffer
            custom_range_arr[:, stats_col : stats_col + 3] = (2.0 * residue_buffer)
            stats_col += 3

            bead_num_feat = np.full((N, 1), float(num), dtype=np.float32)   # NO normalization
            res_id_feat   = np.full((N, 1), res_id_val, dtype=np.float32)  # NO normalization
            bead_fraction = float(num) / float(max(1, n_beads - 1))
            bead_fraction_feat = np.full((N, 1), bead_fraction, dtype=np.float32)
            packing_feat = compute_packing_features(c_point, bb_window)  # (N,4)
            new_cont_feat = np.concatenate(
                [bb_shape_feat, packing_feat, n_beads_scaled, bead_fraction_feat],
                axis=1
            )  # (N,10)

            if num == 0:
                bead_info = np.concatenate([B_score, np.array([1.0], dtype=np.float32)], axis=0)  # (21,)
                expanded = np.tile(bead_info[None, :], (N, 1))

                bond_angle = calculate_bond_angles_vectorized_normalized(bb_nei, bb_i, c_point)  # (N,1)
                coord_feat = local_coord_feature(c_point, O, R)  # (N,3)
                bb_rbf = calculate_rbf(bb_i, c_point)  # (N,1) BB-to-current-bead RBF

                rows = slice(bead_counter * N, (bead_counter + 1) * N)
                feat_out[rows, :] = np.concatenate(
                    [expanded, bb_rbf, bond_angle, coord_feat, new_cont_feat, bead_num_feat, res_id_feat],
                    axis=1
                )

                c_prev = c_point.copy()
                A = bb_i.copy()
                B = c_point.copy()

            else:
                expanded = np.tile(B_score[None, :], (N, 1))   # (N,20)
                rbf = calculate_rbf(c_prev, c_point)           # (N,1)
                bond_angle = calculate_bond_angles_vectorized_normalized(A, B, c_point)  # (N,1)
                coord_feat = local_coord_feature(c_point, O, R)  # (N,3)
                bb_rbf = calculate_rbf(bb_i, c_point)  # (N,1) BB-to-current-bead RBF

                rows = slice(bead_counter * N, (bead_counter + 1) * N)
                feat_out[rows, :] = np.concatenate(
                    [expanded, rbf, bb_rbf, bond_angle, coord_feat, new_cont_feat, bead_num_feat, res_id_feat],
                    axis=1
                )

                A = B.copy()
                B = c_point.copy()
                c_prev = c_point.copy()

            bead_to_scres[bead_counter] = sc_i
            bead_to_num[bead_counter] = num
            bead_to_bbres[bead_counter] = bb_i_idx
            bead_counter += 1

    start_indx = end_indx

if bead_counter != total_beads:
    raise RuntimeError(f"Internal error: bead_counter={bead_counter} != total_beads={total_beads}")
if stats_col != custom_range_arr.shape[1]:
    raise RuntimeError(
        f"Internal error: stats_col={stats_col} != expected {custom_range_arr.shape[1]}"
    )

# ----------------------------
# Save outputs
# ----------------------------
feature_file = f"cluster_{cluster_number}_SC_CG_RBF_localFrameFeatures.npy"
custom_range_file = f"custom_range_{pdb_name}_{cluster_number}.npy"
custom_min_file = f"custom_min_{pdb_name}_{cluster_number}.npy"
# ---- REORDER to match target reshape (frame-major) ----
# current feat_out rows are bead-major: (bead0 all frames), (bead1 all frames), ...
# convert to frame-major: (frame0 all beads), (frame1 all beads), ...
M = total_beads
feat_out = feat_out.reshape(M, N, FEATURE_DIM).transpose(1, 0, 2).reshape(N * M, FEATURE_DIM)


np.save(feature_file, feat_out)
np.save(custom_range_file, custom_range_arr.astype(np.float32, copy=False))
np.save(custom_min_file, custom_min_arr.astype(np.float32, copy=False))

meta_file = f"localFrame_META_{pdb_name}_cluster{cluster_number}.npz"
np.savez_compressed(
    meta_file,
    feature_dim=np.array([FEATURE_DIM], dtype=np.int32),
    feature_layout=np.array([
        "BLOSUM20 + (scalar_or_RBF) + bb_rbf + bond_angle + local_coord(xyz) + bb_shape(sin/cos kappa,tau) + packing(6/8/10A,min_d) + n_beads_scaled + bead_fraction + bead_num + residue_id"
    ]),
    delta_half_range=np.array([DELTA_HALF_RANGE], dtype=np.float32),
    delta_range=np.array([DELTA_RANGE], dtype=np.float32),
    segment_starts=np.array(segment_starts, dtype=np.int32),
    segment_ends=np.array(segment_ends, dtype=np.int32),
    sc_to_bb=sc_to_bb,
    bead_to_scres=bead_to_scres,
    bead_to_num=bead_to_num,
    bead_to_bbres=bead_to_bbres,
    sequence=np.array(sc_sequence),
    sc_sequence=np.array(sc_sequence),
    L_bb=np.array([L_bb], dtype=np.int32),
)

print("Saved features:", feature_file, feat_out.shape)   # (N*total_beads, 38)
print(
    "Saved target scaling stats:",
    custom_range_file,
    custom_min_file,
    custom_range_arr.shape,
    f"default_buffer={default_buffer}",
    f"aromatic_buffer={aromatic_buffer}",
)
if save_local_frames:
    print("Saved R (SC-aligned):", R_file, R_sc.shape)
    print("Saved O (SC-aligned):", O_file, O_sc.shape)
else:
    print("Skipped R/O local-frame file export (enable with --save-local-frames).")
print("Saved meta:", meta_file)

if save_local_frames:
    print("\nLater (all-atoms normalization) for atom attached to bead k:")
    print("  sc_i = bead_to_scres[k]")
    print("  R = R_sc[frame, sc_i]")
    print("  bead_xyz = your bead coordinate (global)")
    print("  atom_norm = 0.5 + (R^T @ (atom_xyz - bead_xyz)) / 7.0")
    print("Undo:")
    print("  atom_xyz = bead_xyz + R @ ((atom_norm - 0.5) * 7.0)")
