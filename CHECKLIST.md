## First set of Featuere inputs:
- [x] 1J4N  Yet to finish the RDF   Backbone done    {Side chain PDB done}
- [x] 1LIN  Yet to finish the RDF  Backbone done    {Side chain PDB done}
- [x] 1TUP  Yet to finish the RDF  Backbone done    {Side chain PDB done}
- [x] 1UBQ  Yet to finish the RDF  Backbone done    {Side chain PDB done}
- [x] 2J4A  Yet to finish the RDF  Backbone done    {Side chain PDB done}
- [x] 1C3W  DONE Backbone data features    {Side chain PDB done} 
- [x] 4MT2  DONE Backbone data features    {Side chain PDB done}
- [x] 4IVV  DONE Backbone data features    {Side chain PDB done}
- [x] 2POR  DONE Backbone data features    {Side chain PDB done}
- [x] 1RWH  DONE Backbone data features    {Side chain PDB done}
- [x] 1TGB  DONE Backbone data features    {\\\\\\\EXTRACTING Side chain PDB \\\\\\\\\}
- [x] 1TIM  DONE Backbone data features    {Side chain PDB done}


[4080,4896,8976,9756,13836,14616,18696]

[1020,1224,2244,2439,3459,3654,4674]

TESTING_FEAT.txt
TESTING_LAB.txt

gmx trjconv -s topol.tpr -f traj.xtc -o noPBC.xtc -pbc mol


gmx trjconv -s topol.tpr -f traj.xtc -o protein.gro -pbc mol -center -dump 0


color Display Background white


# Check if the file exists
file_name = f'sequence_{name_pdb}.txt'

# If the file doesn't exist, open and write to it
if not os.path.exists(file_name):
    with open(file_name, 'w') as f:
        result = ''
        for i, res in enumerate(sequence):
            if i < (len(sequence) - 1):
                result += res + ','
            else:
                result += res
        f.write(result)
else:
    print(f"File {file_name} already exists. Skipping.")
