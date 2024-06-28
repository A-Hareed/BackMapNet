import numpy as np
# import MDAnalysis as mda
# from MDAnalysis.coordinates.memory import MemoryReader
import csv


# PDB = 'CG_bb.pdb'
PDB = 'Iga_all_atom_sim'

with open(PDB,newline='') as f:
    pdb_file = csv.reader(f,delimiter='\t') 
    temp_list = []
# pdb_file[:6]
# print(len(pdb_file))
    row = []
    full_data = []
    
    for line in pdb_file:
    
        if 'MODEL' in line[0]:
            
            if len(row) > 0:
                # time_frame.append(row)
                full_data.append(row)
                row = []
            line = line[0].split()
            line = [i for i in line if i != '']
            # time_frame = [line[1]]
            row.append(line[1])
        if 'ATOM' in line[0]:
            line = line[0].split()
            line = [i for  i in line if i != '']

            row.extend(line[6:9])
        # line_no_space = line.strip()
        # print(line_no_space)
        # split_line = line_no_space.split('\t')
        # # split_line = split_line.remove('')
        # print(line)
        # break

    
    
print(full_data[:3])
    # print(row)


arr = np.array(full_data)

np.save('extracted_pdb_all_atom_backbone_iga_antigen.npy',arr)




