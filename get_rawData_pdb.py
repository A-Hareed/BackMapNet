from Bio.PDB import PDBParser, PDBList
from Bio.SeqUtils import seq1
from collections import Counter
import os

# Expanded list of PDB IDs with more human proteins
pdb_ids = ["2J4A", "1UBQ", "1J4N", "1Z83", "1LIN", "1HHO", "1A6M", "1HRC", "1TRZ", "1TUP", "1GFL", "2HI4", "1A3N", "1BI7", "1AAP", "1FHO"]

# Initialize the PDB parser
parser = PDBParser(QUIET=True)

# Directory to store PDB files
pdb_dir = './pdb_files'
if not os.path.exists(pdb_dir):
    os.makedirs(pdb_dir)

# Function to get the sequence from a PDB file
def get_sequence(pdb_id):
    try:
        structure = parser.get_structure(pdb_id, f"{pdb_dir}/pdb{pdb_id.lower()}.ent")
        sequence = ""
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_id()[0] == ' ':
                        sequence += seq1(residue.get_resname())
        return sequence
    except Exception as e:
        print(f"Error processing {pdb_id}: {e}")
        return ""

# Download PDB files
pdbl = PDBList()
for pdb_id in pdb_ids:
    try:
        pdbl.retrieve_pdb_file(pdb_id, pdir=pdb_dir, file_format='pdb')
    except Exception as e:
        print(f"Error downloading {pdb_id}: {e}")

# Combine sequences from all PDB files and get their lengths
combined_sequence = ""
sequence_lengths = {}
for pdb_id in pdb_ids:
    sequence = get_sequence(pdb_id)
    if sequence:
        combined_sequence += sequence
        sequence_lengths[pdb_id] = len(sequence)

# Count the occurrences of each amino acid
amino_acid_count = Counter(combined_sequence)

# Calculate the total number of amino acids
total_amino_acids = sum(amino_acid_count.values())

# Calculate the percentage of each amino acid
amino_acid_percentage = {aa: (count / total_amino_acids) * 100 for aa, count in amino_acid_count.items()}

# Check if all 20 amino acids are present
all_amino_acids_present = set(amino_acid_count.keys()) == set('ACDEFGHIKLMNPQRSTVWY')

# Output results
print(f"All 20 amino acids present: {all_amino_acids_present}")
print("Amino acid percentages:")
for aa, percentage in amino_acid_percentage.items():
    print(f"{aa}: {percentage:.2f}%")

# Output lengths of PDB sequences
print("Lengths of PDB sequences:")
for pdb_id, length in sequence_lengths.items():
    print(f"{pdb_id}: {length} amino acids")

# Additional: Print any missing amino acids
missing_amino_acids = set('ACDEFGHIKLMNPQRSTVWY') - set(amino_acid_count.keys())
if missing_amino_acids:
    print(f"Missing amino acids: {', '.join(missing_amino_acids)}")
