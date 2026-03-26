#!/usr/bin/env python3
import os
import numpy as np
import sys
import re
import shlex
import urllib.request
import urllib.error
from bond_lookup import ATOM_ORDER
from ff14sb_bond_lengths import ff14sb_sidechain_bond_lengths

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
  argv[7+] out_file.npy           : output filename (default: reversed_localframe_{cluster_id}.npy)
  argv[7+] keep_group_idx.npy     : optional 1D group index file from filtered run_model output
  argv[7+] --no-bond-fix          : disable ff14 hard bond-length correction
  argv[7+] --bond-fix-threshold X : only correct when |pred_len - ideal_len| > X Angstrom
  argv[7+] --bond-fix-soft        : use restrained soft correction instead of hard snapping
  argv[7+] --bond-fix-alpha X     : soft correction strength (0..1), default 0.2
  argv[7+] --bond-fix-smooth-width X : smooth turn-on width (Angstrom), default 0.05
  argv[7+] --bond-fix-non-ring-only  : skip ring-containing sidechains (HIS/PHE/TYR/TRP/PRO)
  argv[7+] --ring-fix                : ring-shape correction using rigid template fit
  argv[7+] --ring-template-pdb PATH  : PDB used to build ring templates (default: frame_BackMapNet_V3_13.pdb)
  argv[7+] --ring-template-source X  : one of {pdb, ccd}; default pdb
                                       ccd uses online RCSB CCD ideal coordinates
                                       (https://files.rcsb.org/ligands/download/<RES>.cif)
  argv[7+] --ring-template-cache-dir : cache dir for downloaded CCD CIF files
                                       (default: .ring_template_cache)
  argv[7+] --ring-fix-alpha X        : blend toward fitted ring template (0..1), default 0.25

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
out_file_default = f"reversed_localframe_{cluster_id}.npy"
out_file = out_file_default
keep_idx_file = None
apply_bond_fix = True
bond_fix_threshold = 0.0
bond_fix_soft = False
bond_fix_alpha = 0.2
bond_fix_smooth_width = 0.05
bond_fix_non_ring_only = False
apply_ring_fix = False
ring_template_pdb = "frame_BackMapNet_V3_13.pdb"
ring_template_source = "pdb"
ring_template_cache_dir = ".ring_template_cache"
ring_fix_alpha = 0.25

extra_positional = []
extra_args = sys.argv[7:]
i_arg = 0
while i_arg < len(extra_args):
    tok = extra_args[i_arg]
    if tok == "--no-bond-fix":
        apply_bond_fix = False
        i_arg += 1
    elif tok == "--bond-fix-soft":
        bond_fix_soft = True
        i_arg += 1
    elif tok == "--bond-fix-threshold":
        if i_arg + 1 >= len(extra_args):
            raise ValueError("--bond-fix-threshold requires a numeric value")
        bond_fix_threshold = float(extra_args[i_arg + 1])
        if bond_fix_threshold < 0.0:
            raise ValueError("--bond-fix-threshold must be >= 0")
        i_arg += 2
    elif tok == "--bond-fix-alpha":
        if i_arg + 1 >= len(extra_args):
            raise ValueError("--bond-fix-alpha requires a numeric value")
        bond_fix_alpha = float(extra_args[i_arg + 1])
        if bond_fix_alpha < 0.0 or bond_fix_alpha > 1.0:
            raise ValueError("--bond-fix-alpha must be in [0,1]")
        i_arg += 2
    elif tok == "--bond-fix-smooth-width":
        if i_arg + 1 >= len(extra_args):
            raise ValueError("--bond-fix-smooth-width requires a numeric value")
        bond_fix_smooth_width = float(extra_args[i_arg + 1])
        if bond_fix_smooth_width < 0.0:
            raise ValueError("--bond-fix-smooth-width must be >= 0")
        i_arg += 2
    elif tok == "--bond-fix-non-ring-only":
        bond_fix_non_ring_only = True
        i_arg += 1
    elif tok == "--ring-fix":
        apply_ring_fix = True
        i_arg += 1
    elif tok == "--ring-template-pdb":
        if i_arg + 1 >= len(extra_args):
            raise ValueError("--ring-template-pdb requires a path")
        ring_template_pdb = extra_args[i_arg + 1]
        i_arg += 2
    elif tok == "--ring-template-source":
        if i_arg + 1 >= len(extra_args):
            raise ValueError("--ring-template-source requires a value: pdb or ccd")
        ring_template_source = extra_args[i_arg + 1].strip().lower()
        if ring_template_source not in ("pdb", "ccd"):
            raise ValueError("--ring-template-source must be one of: pdb, ccd")
        i_arg += 2
    elif tok == "--ring-template-cache-dir":
        if i_arg + 1 >= len(extra_args):
            raise ValueError("--ring-template-cache-dir requires a path")
        ring_template_cache_dir = extra_args[i_arg + 1]
        i_arg += 2
    elif tok == "--ring-fix-alpha":
        if i_arg + 1 >= len(extra_args):
            raise ValueError("--ring-fix-alpha requires a numeric value")
        ring_fix_alpha = float(extra_args[i_arg + 1])
        if ring_fix_alpha < 0.0 or ring_fix_alpha > 1.0:
            raise ValueError("--ring-fix-alpha must be in [0,1]")
        i_arg += 2
    elif tok.startswith("-"):
        raise ValueError(f"Unknown option: {tok}")
    else:
        extra_positional.append(tok)
        i_arg += 1

if len(extra_positional) >= 1:
    out_file = extra_positional[0]
if len(extra_positional) >= 2:
    keep_idx_file = extra_positional[1]
if len(extra_positional) > 2:
    raise ValueError(
        "Too many positional args. Expected at most: [out_file.npy] [keep_group_idx.npy]"
    )

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


def _bond_key(a, b):
    return (a, b) if a <= b else (b, a)


def build_atom_bond_plan(atom_order, ff14_table, non_ring_only=False):
    """
    Build per-residue atom-parent bond plans from ff14 sidechain bonds.

    Returns
    -------
    dict
        bond_plan_by_res[resname] = {
            "parent_idx": int32 (n_atoms,),
            "ideal_len": float32 (n_atoms,),
            "apply_order": int32 (n_apply,),
            "bond_count": int
        }
    """
    ff14_lookup = {
        res: {_bond_key(a, b): float(v) for (a, b), v in pairs.items()}
        for res, pairs in ff14_table.items()
    }
    ring_residues = {"HIS", "PHE", "TYR", "TRP", "PRO"}

    bond_plan_by_res = {}
    for resname, bead_map in atom_order.items():
        ordered_beads = sorted(bead_map.keys())
        atom_names = []
        for bead_num in ordered_beads:
            atom_names.extend(bead_map[bead_num])

        n_atoms = len(atom_names)
        if non_ring_only and resname in ring_residues:
            bond_plan_by_res[resname] = {
                "parent_idx": -np.ones((n_atoms,), dtype=np.int32),
                "ideal_len": np.full((n_atoms,), np.nan, dtype=np.float32),
                "apply_order": np.asarray([], dtype=np.int32),
                "bond_count": 0,
            }
            continue

        atom_to_idx = {name: idx for idx, name in enumerate(atom_names)}
        res_lookup = ff14_lookup.get(resname, {})

        # Undirected adjacency of sidechain atom graph with ideal bond lengths.
        neigh = [[] for _ in range(n_atoms)]
        for (a, b), ideal in res_lookup.items():
            if a in atom_to_idx and b in atom_to_idx:
                ia = atom_to_idx[a]
                ib = atom_to_idx[b]
                neigh[ia].append((ib, ideal))
                neigh[ib].append((ia, ideal))

        parent_idx = -np.ones((n_atoms,), dtype=np.int32)
        ideal_len = np.full((n_atoms,), np.nan, dtype=np.float32)
        apply_order = []

        # Build a deterministic spanning forest (BFS) to avoid ring over-constraints.
        visited = np.zeros((n_atoms,), dtype=bool)
        for root in range(n_atoms):
            if visited[root]:
                continue
            visited[root] = True
            queue = [root]
            while queue:
                u = queue.pop(0)
                for v, ideal in sorted(neigh[u], key=lambda x: x[0]):
                    if visited[v]:
                        continue
                    visited[v] = True
                    parent_idx[v] = u
                    ideal_len[v] = np.float32(ideal)
                    apply_order.append(v)
                    queue.append(v)

        bond_plan_by_res[resname] = {
            "parent_idx": parent_idx,
            "ideal_len": ideal_len,
            "apply_order": np.asarray(apply_order, dtype=np.int32),
            "bond_count": int(np.isfinite(ideal_len).sum()),
        }

    return bond_plan_by_res


def hard_correct_atom_bonds_in_angstrom(
    atom_xyz,
    parent_idx,
    ideal_len,
    apply_order,
    eps=1e-8,
    deviation_threshold=0.0,
    soft=False,
    alpha=0.2,
    smooth_width=0.05,
):
    """
    Hard-correct atom-atom bond lengths in Angstrom while preserving directions.

    Parameters
    ----------
    atom_xyz : np.ndarray, shape (N, n_atoms, 3)
        Atom coordinates in global Angstrom.
    parent_idx : np.ndarray, shape (n_atoms,)
        Parent atom index for each atom, or -1 for roots.
    ideal_len : np.ndarray, shape (n_atoms,)
        Target bond length for atom-parent edge; NaN entries are skipped.
    apply_order : np.ndarray, shape (n_apply,)
        Atom indices to correct in dependency order.
    eps : float
        Numerical stabilizer for near-zero vectors.
    deviation_threshold : float
        Apply correction only where |pred_len - ideal_len| > threshold (Angstrom).
        Use 0.0 to always correct valid slots.
    soft : bool
        If True, apply restrained blending toward ideal length instead of hard snap.
    alpha : float
        Soft correction strength in [0,1].
    smooth_width : float
        Width for smooth turn-on after the dead-zone (Angstrom). If <=0, uses hard gate.
    """
    n_atoms = atom_xyz.shape[1]
    if parent_idx.shape[0] != n_atoms or ideal_len.shape[0] != n_atoms:
        raise ValueError(
            f"Bond plan size mismatch: n_atoms={n_atoms}, "
            f"parent_idx={parent_idx.shape}, ideal_len={ideal_len.shape}"
        )

    for child in apply_order:
        p = int(parent_idx[child])
        if p < 0:
            continue
        ideal = float(ideal_len[child])
        if not np.isfinite(ideal):
            continue

        vec = atom_xyz[:, child, :] - atom_xyz[:, p, :]
        dist = np.sqrt(np.sum(vec * vec, axis=1, keepdims=True))

        if soft:
            err = np.abs(dist - np.float32(ideal))
            if smooth_width > 0.0:
                u = np.clip((err - np.float32(deviation_threshold)) / np.float32(smooth_width), 0.0, 1.0)
                w = 3.0 * u * u - 2.0 * u * u * u
            else:
                w = (err > np.float32(deviation_threshold)).astype(np.float32)

            alpha_eff = np.float32(alpha) * w
            dist_new = (1.0 - alpha_eff) * dist + alpha_eff * np.float32(ideal)
            vec_new = dist_new * vec / (dist + np.float32(eps))
            atom_xyz[:, child, :] = atom_xyz[:, p, :] + vec_new
        else:
            factor = np.float32(ideal) / (dist + np.float32(eps))
            child_new = atom_xyz[:, p, :] + vec * factor
            if deviation_threshold > 0.0:
                needs_fix = np.abs(dist - np.float32(ideal)) > np.float32(deviation_threshold)
                atom_xyz[:, child, :] = np.where(needs_fix, child_new, atom_xyz[:, child, :])
            else:
                atom_xyz[:, child, :] = child_new

    return atom_xyz


# Ring atom sets: sidechain ring atoms only (explicitly excludes CA).
RING_ATOMS_BY_RES = {
    "PHE": ["CG", "CD1", "CE1", "CZ", "CE2", "CD2"],
    "HIS": ["CG", "ND1", "CE1", "NE2", "CD2"],
    "TYR": ["CG", "CD1", "CE1", "CZ", "CE2", "CD2"],
    # Fused indole system in sidechain-only ordering.
    "TRP": ["CG", "CD1", "NE1", "CE2", "CZ2", "CH2", "CZ3", "CE3", "CD2"],
}


def build_ring_index_map(atom_order, ring_atoms_by_res):
    """
    Build residue-local ring atom index lists from ATOM_ORDER.

    Returns
    -------
    tuple(dict, dict)
        ring_idx_by_res[resname] = int32 index array in residue-local atom ordering.
        missing_by_res[resname] = list of missing atom names (if any).
    """
    ring_idx_by_res = {}
    missing_by_res = {}

    for resname, ring_atoms in ring_atoms_by_res.items():
        bead_map = atom_order.get(resname)
        if bead_map is None:
            missing_by_res[resname] = list(ring_atoms)
            continue

        atom_names = []
        for bead_num in sorted(bead_map.keys()):
            atom_names.extend(bead_map[bead_num])
        atom_to_idx = {a: i for i, a in enumerate(atom_names)}

        missing = [a for a in ring_atoms if a not in atom_to_idx]
        if missing:
            missing_by_res[resname] = missing
            continue

        ring_idx_by_res[resname] = np.asarray(
            [atom_to_idx[a] for a in ring_atoms], dtype=np.int32
        )

    return ring_idx_by_res, missing_by_res


def _kabsch_fit_single(source_xyz, target_xyz):
    """
    Rigidly fit source -> target (both shape (m,3)) and return fitted source.
    """
    src_cent = source_xyz.mean(axis=0, keepdims=True)
    tgt_cent = target_xyz.mean(axis=0, keepdims=True)
    src0 = source_xyz - src_cent
    tgt0 = target_xyz - tgt_cent

    h = src0.T @ tgt0
    u, _s, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0.0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T

    return src0 @ r + tgt_cent


def build_ring_templates_from_pdb(pdb_path, ring_atoms_by_res):
    """
    Build idealized residue-specific ring templates from a reference PDB.

    For each requested residue type, all occurrences with complete ring atom sets
    are collected, centered, rigidly aligned to a reference occurrence, and averaged.

    Parameters
    ----------
    pdb_path : str
        Path to a PDB file.
    ring_atoms_by_res : dict
        Mapping resname -> ordered ring atom names.

    Returns
    -------
    tuple(dict, dict)
        templates[resname] = (m,3) float32 centered template.
        counts[resname] = number of residue examples used.
    """
    wanted = set(ring_atoms_by_res.keys())
    examples = {res: [] for res in ring_atoms_by_res}
    wanted_atoms = {res: set(atoms) for res, atoms in ring_atoms_by_res.items()}

    current_key = None
    current_res = None
    atom_map = {}

    def flush_current():
        if current_res not in wanted:
            return
        ring_atoms = ring_atoms_by_res[current_res]
        if all(a in atom_map for a in ring_atoms):
            xyz = np.asarray([atom_map[a] for a in ring_atoms], dtype=np.float32)
            examples[current_res].append(xyz)

    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue

            altloc = line[16].strip()
            if altloc not in ("", "A"):
                continue

            atom_name = line[12:16].strip()
            resname = line[17:20].strip()
            chain = line[21].strip()
            resseq = line[22:26].strip()
            icode = line[26].strip()
            key = (chain, resseq, icode, resname)

            if key != current_key:
                if current_key is not None:
                    flush_current()
                current_key = key
                current_res = resname
                atom_map = {}

            if resname in wanted and atom_name in wanted_atoms[resname]:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                atom_map[atom_name] = np.asarray([x, y, z], dtype=np.float32)

    if current_key is not None:
        flush_current()

    templates = {}
    counts = {}
    for resname, arr_list in examples.items():
        counts[resname] = len(arr_list)
        if not arr_list:
            continue

        centered = [arr - arr.mean(axis=0, keepdims=True) for arr in arr_list]
        ref = centered[0]
        aligned = []
        for arr in centered:
            aligned.append(_kabsch_fit_single(arr, ref))

        template = np.mean(np.stack(aligned, axis=0), axis=0)
        template = template - template.mean(axis=0, keepdims=True)
        templates[resname] = template.astype(np.float32)

    return templates, counts


def _extract_chem_comp_atom_coords_from_cif(cif_text):
    """
    Parse chem_comp atom coordinates from an RCSB CCD CIF text block.

    Returns
    -------
    dict
        atom_id -> np.ndarray([x,y,z], dtype=float32)
        Prefers ideal coordinates if available.
    """
    lines = cif_text.splitlines()
    coords_ideal = {}
    coords_model = {}
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].strip()
        if line != "loop_":
            i += 1
            continue

        i += 1
        tags = []
        while i < n and lines[i].strip().startswith("_"):
            tags.append(lines[i].strip())
            i += 1

        if not tags or not any(t.startswith("_chem_comp_atom.") for t in tags):
            while i < n:
                s = lines[i].strip()
                if s in ("#", "loop_") or s.startswith("_") or s.startswith("data_"):
                    break
                i += 1
            continue

        tag_to_idx = {t: j for j, t in enumerate(tags)}
        atom_tag = None
        for cand in ("_chem_comp_atom.atom_id", "_chem_comp_atom.pdbx_component_atom_id"):
            if cand in tag_to_idx:
                atom_tag = cand
                break
        if atom_tag is None:
            continue

        ideal_tags = (
            "_chem_comp_atom.pdbx_model_Cartn_x_ideal",
            "_chem_comp_atom.pdbx_model_Cartn_y_ideal",
            "_chem_comp_atom.pdbx_model_Cartn_z_ideal",
        )
        model_tags = (
            "_chem_comp_atom.model_Cartn_x",
            "_chem_comp_atom.model_Cartn_y",
            "_chem_comp_atom.model_Cartn_z",
        )
        has_ideal = all(t in tag_to_idx for t in ideal_tags)
        has_model = all(t in tag_to_idx for t in model_tags)
        if not has_ideal and not has_model:
            continue

        ntags = len(tags)
        while i < n:
            s = lines[i].strip()
            if not s:
                i += 1
                continue
            if s in ("#", "loop_") or s.startswith("_") or s.startswith("data_"):
                break

            toks = shlex.split(s, posix=True)
            if len(toks) < ntags:
                i += 1
                continue

            atom_name = toks[tag_to_idx[atom_tag]]

            if has_ideal:
                xi = toks[tag_to_idx[ideal_tags[0]]]
                yi = toks[tag_to_idx[ideal_tags[1]]]
                zi = toks[tag_to_idx[ideal_tags[2]]]
                if xi not in (".", "?") and yi not in (".", "?") and zi not in (".", "?"):
                    coords_ideal[atom_name] = np.asarray(
                        [float(xi), float(yi), float(zi)], dtype=np.float32
                    )

            if has_model:
                xm = toks[tag_to_idx[model_tags[0]]]
                ym = toks[tag_to_idx[model_tags[1]]]
                zm = toks[tag_to_idx[model_tags[2]]]
                if xm not in (".", "?") and ym not in (".", "?") and zm not in (".", "?"):
                    coords_model[atom_name] = np.asarray(
                        [float(xm), float(ym), float(zm)], dtype=np.float32
                    )

            i += 1

    # Prefer ideal, fallback to model.
    out = dict(coords_model)
    out.update(coords_ideal)
    return out


def _load_ccd_cif_text(resname, cache_dir):
    """
    Load a residue CCD CIF from cache or RCSB.

    Returns
    -------
    tuple(str, str)
        (cif_text, source) where source in {"cache", "download"}.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{resname}.cif")
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return f.read(), "cache"

    url = f"https://files.rcsb.org/ligands/download/{resname}.cif"
    with urllib.request.urlopen(url, timeout=20) as resp:
        cif_text = resp.read().decode("utf-8")

    with open(cache_path, "w") as f:
        f.write(cif_text)
    return cif_text, "download"


def build_ring_templates_from_ccd(ring_atoms_by_res, cache_dir):
    """
    Build ring templates from online RCSB CCD ideal coordinates.

    Parameters
    ----------
    ring_atoms_by_res : dict
        resname -> ordered ring atom names.
    cache_dir : str
        Local cache directory for CCD CIF files.

    Returns
    -------
    tuple(dict, dict, dict)
        templates[resname] = (m,3) centered template
        source_by_res[resname] = "cache" or "download"
        errors_by_res[resname] = error string
    """
    templates = {}
    source_by_res = {}
    errors_by_res = {}

    for resname, ring_atoms in ring_atoms_by_res.items():
        try:
            cif_text, source = _load_ccd_cif_text(resname, cache_dir)
            coord_map = _extract_chem_comp_atom_coords_from_cif(cif_text)
        except (OSError, urllib.error.URLError, urllib.error.HTTPError, ValueError) as e:
            errors_by_res[resname] = f"load failed: {e}"
            continue

        missing = [a for a in ring_atoms if a not in coord_map]
        if missing:
            errors_by_res[resname] = f"missing atoms in CCD: {','.join(missing)}"
            continue

        template = np.asarray([coord_map[a] for a in ring_atoms], dtype=np.float32)
        template = template - template.mean(axis=0, keepdims=True)
        templates[resname] = template
        source_by_res[resname] = source

    return templates, source_by_res, errors_by_res


def batched_kabsch_fit(x_pred, template):
    """
    Fit a static ring template onto predicted ring coordinates for a batch.

    Parameters
    ----------
    x_pred : np.ndarray, shape (N, m, 3)
        Predicted ring coordinates.
    template : np.ndarray, shape (m, 3)
        Ring template coordinates.

    Returns
    -------
    np.ndarray, shape (N, m, 3)
        Rigidly-fitted template coordinates in each predicted ring pose.
    """
    n_batch, n_atoms, _ = x_pred.shape
    t = np.broadcast_to(template[None, :, :], (n_batch, n_atoms, 3)).astype(np.float32)

    pred_cent = x_pred.mean(axis=1, keepdims=True)
    temp_cent = t.mean(axis=1, keepdims=True)
    x0 = x_pred - pred_cent
    t0 = t - temp_cent

    h = np.einsum("nki,nkj->nij", t0, x0)
    u, _s, vt = np.linalg.svd(h)
    r = np.einsum("nij,njk->nik", vt.transpose(0, 2, 1), u.transpose(0, 2, 1))

    det_r = np.linalg.det(r)
    bad = det_r < 0.0
    if np.any(bad):
        vt_bad = vt[bad].copy()
        vt_bad[:, -1, :] *= -1.0
        r[bad] = np.einsum(
            "nij,njk->nik",
            vt_bad.transpose(0, 2, 1),
            u[bad].transpose(0, 2, 1),
        )

    x_fit = np.einsum("nij,nkj->nki", r, t0) + pred_cent
    return x_fit.astype(np.float32)


def ring_correct_residue(atom_xyz, ring_idx, template, alpha=0.25):
    """
    Correct ring geometry by fitting a residue-specific template in a rigid manner.
    """
    x_pred = atom_xyz[:, ring_idx, :]
    x_fit = batched_kabsch_fit(x_pred, template)

    a = np.float32(alpha)
    if a >= 1.0:
        atom_xyz[:, ring_idx, :] = x_fit
    elif a > 0.0:
        atom_xyz[:, ring_idx, :] = (1.0 - a) * x_pred + a * x_fit
    return atom_xyz

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
# Precompute atom-bond plans by residue (atom-atom correction, no bead radii)
# -------------------------
if apply_bond_fix:
    bond_plan_by_res = build_atom_bond_plan(
        atom_order=ATOM_ORDER,
        ff14_table=ff14sb_sidechain_bond_lengths,
        non_ring_only=bond_fix_non_ring_only,
    )
else:
    bond_plan_by_res = {}

if apply_ring_fix:
    ring_idx_by_res, ring_missing_by_res = build_ring_index_map(
        atom_order=ATOM_ORDER,
        ring_atoms_by_res=RING_ATOMS_BY_RES,
    )
    ring_template_errors = {}
    ring_template_origin = {}
    if ring_template_source == "pdb":
        if not os.path.exists(ring_template_pdb):
            raise FileNotFoundError(
                f"Ring template PDB not found: {ring_template_pdb}. "
                "Use --ring-template-pdb PATH, use --ring-template-source ccd, "
                "or disable --ring-fix."
            )
        ring_templates_by_res, ring_template_counts = build_ring_templates_from_pdb(
            pdb_path=ring_template_pdb,
            ring_atoms_by_res=RING_ATOMS_BY_RES,
        )
        ring_template_origin = {res: "pdb" for res in ring_templates_by_res}
    elif ring_template_source == "ccd":
        ring_templates_by_res, ring_template_origin, ring_template_errors = build_ring_templates_from_ccd(
            ring_atoms_by_res=RING_ATOMS_BY_RES,
            cache_dir=ring_template_cache_dir,
        )
        ring_template_counts = {res: int(res in ring_templates_by_res) for res in RING_ATOMS_BY_RES}
    else:
        raise ValueError(f"Unknown ring template source: {ring_template_source}")

    ring_fix_residues = sorted(
        set(ring_idx_by_res.keys()).intersection(ring_templates_by_res.keys())
    )
else:
    ring_idx_by_res = {}
    ring_missing_by_res = {}
    ring_templates_by_res = {}
    ring_template_counts = {}
    ring_template_errors = {}
    ring_template_origin = {}
    ring_fix_residues = []

# -------------------------
# Denormalise helper
# -------------------------
def denorm_localframe(norm_flat, bead_xyz, R, delta_range):
    # norm_flat: (N, 3*n_atoms_group)
    norm = norm_flat.reshape(N, -1, 3).astype(np.float32)
    d_local = (norm - 0.5) * delta_range

    # d_global = R @ d_local
    d_global = np.einsum("nij,nkj->nki", R.astype(np.float32, copy=False), d_local)
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
corrected_residues = 0
corrected_bond_targets = 0
ring_corrected_residues = 0

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

        atoms_flat = denorm_localframe(
            norm_slice,
            bead_xyz,
            R,
            delta_range,
        )
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

            atoms_flat = denorm_localframe(
                norm_slice,
                bead_xyz,
                R,
                delta_range,
            )
            residue_out[:, idxs] = atoms_flat

            k += 1

    if apply_bond_fix:
        plan = bond_plan_by_res.get(res, None)
        if plan is not None and plan["bond_count"] > 0:
            atom_xyz = residue_out.reshape(N, -1, 3)
            if atom_xyz.shape[1] != plan["parent_idx"].shape[0]:
                raise ValueError(
                    f"Atom-order mismatch for residue {res}: "
                    f"coords has {atom_xyz.shape[1]} atoms, plan has {plan['parent_idx'].shape[0]}"
                )
            hard_correct_atom_bonds_in_angstrom(
                atom_xyz=atom_xyz,
                parent_idx=plan["parent_idx"],
                ideal_len=plan["ideal_len"],
                apply_order=plan["apply_order"],
                deviation_threshold=bond_fix_threshold,
                soft=bond_fix_soft,
                alpha=bond_fix_alpha,
                smooth_width=bond_fix_smooth_width,
            )
            residue_out = atom_xyz.reshape(N, -1)
            corrected_residues += 1
            corrected_bond_targets += int(plan["bond_count"])

    if apply_ring_fix and res in ring_templates_by_res:
        ring_idx = ring_idx_by_res.get(res, None)
        if ring_idx is not None and ring_idx.size > 0:
            atom_xyz = residue_out.reshape(N, -1, 3)
            ring_correct_residue(
                atom_xyz=atom_xyz,
                ring_idx=ring_idx,
                template=ring_templates_by_res[res],
                alpha=ring_fix_alpha,
            )
            residue_out = atom_xyz.reshape(N, -1)
            ring_corrected_residues += 1

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
if apply_bond_fix:
    mode = "soft" if bond_fix_soft else "hard"
    print(
        f"Bond correction (atom-atom, {mode}): enabled; residues corrected =",
        corrected_residues,
        "bond-targets =",
        corrected_bond_targets,
    )
    print("Bond correction dead-zone threshold (A):", bond_fix_threshold)
    if bond_fix_soft:
        print("Soft alpha:", bond_fix_alpha, "smooth width (A):", bond_fix_smooth_width)
    print("Non-ring-only:", bond_fix_non_ring_only)
else:
    print("Hard bond correction: disabled (--no-bond-fix)")

if apply_ring_fix:
    print(
        "Ring correction: enabled; residues corrected =",
        ring_corrected_residues,
        "template source mode =",
        ring_template_source,
    )
    print("Ring correction alpha:", ring_fix_alpha)
    print("Ring residues with templates:", ring_fix_residues)
    if ring_template_source == "pdb":
        print("Ring template pdb:", ring_template_pdb)
        if ring_template_counts:
            counts_msg = ", ".join(
                f"{res}:{ring_template_counts.get(res, 0)}"
                for res in sorted(RING_ATOMS_BY_RES.keys())
            )
            print("Ring template examples used:", counts_msg)
    elif ring_template_source == "ccd":
        print("Ring template CCD cache dir:", ring_template_cache_dir)
        origin_msg = ", ".join(
            f"{res}:{ring_template_origin.get(res, 'missing')}"
            for res in sorted(RING_ATOMS_BY_RES.keys())
        )
        print("Ring template origin by residue:", origin_msg)
        if ring_template_errors:
            err_msg = "; ".join(
                f"{res}={msg}" for res, msg in sorted(ring_template_errors.items())
            )
            print("Ring template CCD warnings:", err_msg)
    if ring_missing_by_res:
        missing_msg = "; ".join(
            f"{res} missing {','.join(missing)}"
            for res, missing in sorted(ring_missing_by_res.items())
        )
        print("Ring index-map warnings:", missing_msg)
else:
    print("Ring correction: disabled (use --ring-fix to enable)")
