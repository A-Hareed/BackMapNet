#!/usr/bin/env python3
"""
Multi-chain PDB writer: parses a sequence file with chains separated by '|' and writes frames with TER records.

Usage:
    python MakePDB.py coords.npy sequence.txt

- coords.npy: NumPy array of shape (n_frames, 3 * total_atoms)
- sequence.txt: chains separated by '|', residues by commas, e.g.:
      ALA,GLY,VAL | ARG,LYS,ILE | PHE,TRP
"""

import numpy as np
import os
import sys

# --------------------- initialise settings ---------------------
if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} coords.npy seq_file")
    sys.exit(1)
coords_file = sys.argv[1]
seq_file = sys.argv[2]
out_dir = 'pdb_frames'
# ---------------------------------------------------------

# Side-chain heavy atom names per residue, From martini 2.2 Mapping
sidechain_names = {
    'ALA': ['CB'],
    'ARG': ['CB','CG','CD','NE','CZ','NH1','NH2'],
    'ASN': ['CB','CG','OD1','ND2'],
    'ASP': ['CB','CG','OD1','OD2'],
    'CYS': ['CB','SG'],
    'GLN': ['CB','CG','CD','OE1','NE2'],
    'GLU': ['CB','CG','CD','OE1','OE2'],
    'GLY': [],
    'HIS': ['CB','CG','ND1','CE1','NE2','CD2'],
    'ILE': ['CB','CG1','CG2','CD1'],
    'LEU': ['CB','CG','CD1','CD2'],
    'LYS': ['CB','CG','CD','CE','NZ'],
    'MET': ['CB','CG','SD','CE'],
    'PHE': ['CB','CG','CD1','CE1','CZ','CD2','CE2'],
    'PRO': ['CD','CB','CG'],
    'SER': ['CB','OG'],
    'THR': ['CB','OG1','CG2'],
    'TRP': ['CB','CG','CD1','NE1','CE2','CD2','CE3','CZ3','CZ2','CH2'],
    'TYR': ['CB','CG','CD1','CE1','CZ','OH','CE2','CD2'],
    'VAL': ['CB','CG1','CG2']
}

# PDB formatting
PDB_ATOM_TMPL = (
    "{record:<6s}{serial:>5d} {name:^4s}{altLoc:1s}{resName:>3s} {chainID:1s}"  # cols 1-22
    "{resSeq:>4d}{iCode:1s}   {x:8.3f}{y:8.3f}{z:8.3f}"                    # cols 23-54
    "{occupancy:6.2f}{tempFactor:6.2f}          {element:>2s}{charge:2s}"  # cols 55-80
)
PDB_TER_TMPL  = "TER   {serial:>5d}      {resName:>3s} {chainID:1s}{resSeq:>4d}\n"

# Load and parse sequence file
with open(seq_file) as fh:
    raw = fh.read().strip()
# Chains separated by '|'
chain_blocks = [blk.strip() for blk in raw.split('|') if blk.strip()]

# Build global atom metadata lists
atom_names      = []
residue_indices = []
residue_names   = []
chain_ids       = []

for c_idx, block in enumerate(chain_blocks):
    chainID = chr(ord('A') + c_idx)
    residues = [r.strip().upper() for r in block.split(',') if r.strip()]
    for resSeq, res in enumerate(residues, start=1):
        # backbone
        for atom in ('N', 'CA', 'C', 'O'):
            atom_names.append(atom)
            residue_indices.append(resSeq)
            residue_names.append(res)
            chain_ids.append(chainID)
        # side chains
        for atom in sidechain_names.get(res, []):
            atom_names.append(atom)
            residue_indices.append(resSeq)
            residue_names.append(res)
            chain_ids.append(chainID)

total_atoms = len(atom_names)

# Detect where chains end (last atom index of each chain)
chain_end_idxs = []
for i in range(total_atoms - 1):
    if chain_ids[i] != chain_ids[i+1]:
        chain_end_idxs.append(i)
# also mark final atom
chain_end_idxs.append(total_atoms - 1)

# Load coordinates
coords_all = np.load(coords_file)
n_frames, ncols = coords_all.shape
assert ncols == total_atoms * 3, f"Expected {total_atoms*3} cols, got {ncols}"

# Prepare output dir
os.makedirs(out_dir, exist_ok=True)

# Write PDB frames
for frame in range(n_frames):
    coords = coords_all[frame]
    pdb_path = os.path.join(out_dir, f'frame_{frame:04d}.pdb')
    with open(pdb_path, 'w') as f:
        atom_serial = 1
        for i in range(total_atoms):
            x, y, z = coords[3*i:3*i+3]
            f.write(PDB_ATOM_TMPL.format(
                record='ATOM', serial=atom_serial,
                name=atom_names[i], altLoc=' ',
                resName=residue_names[i], chainID=chain_ids[i],
                resSeq=residue_indices[i], iCode=' ',
                x=x, y=y, z=z,
                occupancy=1.00, tempFactor=0.00,
                element=atom_names[i][0], charge='' ) + '\n')
            atom_serial += 1
            # insert TER record at end of each chain
            if i in chain_end_idxs:
                f.write(PDB_TER_TMPL.format(
                    serial=atom_serial,
                    resName=residue_names[i],
                    chainID=chain_ids[i],
                    resSeq=residue_indices[i],
                ))
                atom_serial += 1
        f.write('END\n')
    print(f"Wrote {pdb_path}")
print("Done writing all frames.")
