## First set of Featuere inputs:
- [x] 1J4N  Yet to finish the RDF   Backbone done    {Side chain PDB done}
- [x] 1LIN  Yet to finish the RDF  Backbone done    {Side chain PDB done}
- [x] 1TUP  Yet to finish the RDF  Backbone done    {Side chain PDB done}
- [x] 1UBQ  Yet to finish the RDF  Backbone done    {Side chain PDB done}
- [x] 2J4A  Yet to finish the RDF  Backbone done    {Side chain PDB done}
- [x] 1C3W  DONE Backbone data features    {\\\\\\\EXTRACTING Side chain PDB \\\\\\\\\}
- [x] 4MT2  DONE Backbone data features    {Side chain PDB done}
- [x] 4IVV  DONE Backbone data features    {Side chain PDB done}
- [x] 2POR  DONE Backbone data features    {Side chain PDB done}
- [x] 1RWH  DONE Backbone data features    {\\\\\\\EXTRACTING Side chain PDB \\\\\\\\\}
- [x] 1TGB  DONE Backbone data features    {\\\\\\\EXTRACTING Side chain PDB \\\\\\\\\}
- [x] 1TIM  DONE Backbone data features    {Side chain PDB done}


[4080,4896,8976,9756,13836,14616,18696]

[1020,1224,2244,2439,3459,3654,4674]

TESTING_FEAT.txt
TESTING_LAB.txt


with open("TESTING_FEAT.txt", "r") as file:
  lines_feat = [line.strip() for line in file]

with open("TESTING_LAB.txt", "r") as file:
  lines_lab = [line.strip() for line in file]



for i in range(len(lines_lab)):
  print(lines_feat[i],lines_lab[i])


paired = list(zip(lines_feat,lines_lab))

random.shuffle(paired)

features_shuffled, targets_shuffled = zip(*paired)

# Convert them back to lists if needed
features_shuffled = list(features_shuffled)
targets_shuffled = list(targets_shuffled)

for i in range(len(lines_lab)):
  print(features_shuffled[i],'           ',targets_shuffled[i])
