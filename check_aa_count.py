from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from collections import Counter

"""
Ran:
    1J4N
    1LIN
    2J4A

Running:
    1ubq -- finishes 5 July 2am

    1tup  -- charmgui  id 2008334691
    1c3w  -- charmgui id 2059091804
"""
# List of PDB IDs
pdb_ids = ["2j4a", "1ubq", "1j4n", "1tup", "1lin", "1c3w","2por","3eml"] # "1ake", 1TUP

# Initialize the PDB parser
parser = PDBParser(QUIET=True)

# Function to get the sequence from a PDB file
def get_sequence(pdb_id):
    structure = parser.get_structure(pdb_id, f"{pdb_id}.pdb")
    sequence = ""
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == ' ':
                    sequence += seq1(residue.get_resname())
    return sequence

# Combine sequences from all PDB files
combined_sequence = ""
for pdb_id in pdb_ids:
    combined_sequence += get_sequence(pdb_id)

# Count the occurrences of each amino acid
amino_acid_count = Counter(combined_sequence)
# Calculate the total number of amino acids
total_amino_acids = sum(amino_acid_count.values())
print(amino_acid_count, total_amino_acids)

# Calculate the percentage of each amino acid
amino_acid_percentage = {aa: (count / total_amino_acids) * 100 for aa, count in amino_acid_count.items()}

# Check if all 20 amino acids are present
all_amino_acids_present = set(amino_acid_count.keys()) == set('ACDEFGHIKLMNPQRSTVWY')

# Output results
print(f"All 20 amino acids present: {all_amino_acids_present}")
print("Amino acid percentages:")
for aa, percentage in amino_acid_percentage.items():
    print(f"{aa}: {percentage:.2f}%")

# Additional: Print any missing amino acids
missing_amino_acids = set('ACDEFGHIKLMNPQRSTVWY') - set(amino_acid_count.keys())
if missing_amino_acids:
    print(f"Missing amino acids: {', '.join(missing_amino_acids)}")


print(215+147+76+461+301)
