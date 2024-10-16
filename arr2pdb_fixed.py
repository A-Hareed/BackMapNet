import numpy as np
from Bio.PDB import PDBIO, Structure, Model, Chain, Residue, Atom
from Bio.PDB.Polypeptide import PPBuilder
from Bio.Seq import Seq
from Bio.SeqUtils import seq1
from Bio.Data import SCOPData

residue_atom_map = {
    "A": ['N', 'CA', 'C', 'O', 'CB'],   # Alanine
    "C": ['N', 'CA', 'C', 'O', 'CB', 'SG'],   # Cysteine
    "D": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2'],   # Aspartic acid
    "E": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2'],   # Glutamic acid
    "F": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],   # Phenylalanine
    "G": ['N', 'CA', 'C', 'O'],   # Glycine
    "H": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'],   # Histidine
    "I": ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1'],   # Isoleucine
    "K": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ'],   # Lysine
    "L": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2'],   # Leucine
    "M": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE'],   # Methionine
    "N": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2'],   # Asparagine
    "P": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD'],   # Proline
    "Q": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2'],   # Glutamine
    "R": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],   # Arginine
    "S": ['N', 'CA', 'C', 'O', 'CB', 'OG'],   # Serine
    "T": ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2'],   # Threonine
    "V": ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2'],   # Valine
    "W": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],   # Tryptophan
    "Y": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH']   # Tyrosine
}

one_to_three_letter = {
    "A": "ALA",  # Alanine
    "C": "CYS",  # Cysteine
    "D": "ASP",  # Aspartic acid
    "E": "GLU",  # Glutamic acid
    "F": "PHE",  # Phenylalanine
    "G": "GLY",  # Glycine
    "H": "HIS",  # Histidine
    "I": "ILE",  # Isoleucine
    "K": "LYS",  # Lysine
    "L": "LEU",  # Leucine
    "M": "MET",  # Methionine
    "N": "ASN",  # Asparagine
    "P": "PRO",  # Proline
    "Q": "GLN",  # Glutamine
    "R": "ARG",  # Arginine
    "S": "SER",  # Serine
    "T": "THR",  # Threonine
    "V": "VAL",  # Valine
    "W": "TRP",  # Tryptophan
    "Y": "TYR",  # Tyrosine
    "X": "UNK"   # Unknown
}

def build_pdb_from_numpy_coords(coords, sequence, pdb_file_name):
    """Builds a PDB file from a numpy array of coordinates."""
    structure = Structure.Structure('Protein')
    model = Model.Model(0)
    structure.add(model)
    
    chain_id = 'A'  # Use a single-character chain ID
    chain = Chain.Chain(chain_id)
    model.add(chain)
    
    atom_id = 1
    current_atom_index = 0
    
    for residue_id, aa in enumerate(sequence, start=1):
        # Use the one-letter amino acid code directly
        resname = aa.upper()

        print(f"Processing residue: {resname} at position {residue_id}")  # Debugging line
        
        # Get atoms for the current residue from the map
        residue_atoms = residue_atom_map.get(resname, None)
        
        # Raise error if residue is not in the map
        if not residue_atoms:
            raise ValueError(f"Unknown residue: {resname}. Add it to the residue_atom_map.")

        residue = Residue.Residue((' ', residue_id, ' '), one_to_three_letter[resname], chain_id)

        # Add atoms to the residue
        for atom_name in residue_atoms:
            x, y, z = coords[current_atom_index]
            atom = Atom.Atom(atom_name, np.array([x, y, z], dtype=float), 1.0, 1.0, ' ', atom_name, atom_id)
            residue.add(atom)
            current_atom_index += 1
            atom_id += 1
        
        chain.add(residue)

    # Save to PDB file
    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_file_name)
