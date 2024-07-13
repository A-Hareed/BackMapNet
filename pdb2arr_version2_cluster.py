import numpy as np
# import MDAnalysis as mda
# from MDAnalysis.coordinates.memory import MemoryReader
import csv
import sys
import os



# PDB = 'CG_bb.pdb'
PDB = sys.argv[1]
array_file = sys.argv[2]



with open(PDB,newline='') as f:
    pdb_file = csv.reader(f,delimiter='\t') 
    temp_list = []
# pdb_file[:6]
# print(len(pdb_file))
    row = []
    full_data = []
    
    for line in pdb_file:
        if 'TER' in line[0]:

            if len(row) > 0:
                # time_frame.append(row)
                full_data.append(row)
                row = []
            line = line[0].split()
            line = [i for i in line if i != '']
            # time_frame = [line[1]]
         #   row.append(line[1])
        if 'ATOM' in line[0]:

            line = line[0].split()
            line = [i for  i in line if i != '']

            row.extend(line[5:7])
        # line_no_space = line.strip()
        # print(line_no_space)
        # split_line = line_no_space.split('\t')
        # # split_line = split_line.remove('')
        # print(line)
        # break

    
    
#print(full_data[:3])
    # print(row)


arr = np.array(full_data)


if not os.path.exists(array_file):
    np.save(array_file,arr)
    
    print(f"Coordinates saved to {array_file}")
else:
    print(f"{array_file} already exists. Skipping file.")
    
    old_arr = np.load(array_file)
    arr = np.concatenate((old_arr,arr),axis=0)

    np.save(array_file,arr)
