# import packages
import pandas as pd
import numpy as np
import csv
import re


#********************* coordinate file dataframe function *********************************
def get_df(arr):
    row_data = []
    ref_data = []
    row_count = 0
    for i in list(arr):
        row_data.append(i)
        if row_count == 2:
            ref_data.append(row_data)
            row_data = []
            row_count = 0
            continue
        row_count += 1
        

    graph_ref = pd.DataFrame(ref_data, columns=['X', 'Y', 'Z'])
    return graph_ref

#********************************************************************************************

# load predicted files
# PDB_XYZ = 'predicted_str_arr.npy'
# PDB_XYZ = 'predicted_str_arr_long_E16.npy'
# PDB_XYZ = 'sims_forces/atomic_prediction.npy'

PDB_XYZ = 'arr_reconstructed_protein.npy'


predicted_str = np.load(PDB_XYZ)
print(predicted_str[0][0])

ATOM_COL_COUNT = 7
ATOM_NUM_COL_COUNT = 4
ATOM_NUM_END_COL_COUNT = 1
ATOM_SYMB_COL_COUNT = 5
ATOM_SYMB_END_COL_COUNT = 1
THREE_LETTER_COL_COUNT = 4
# CHAIN_ID_COL_COUNT = 4
RESIDUE_NUM_COL_COUNT = 4
RESIDUE_NUM_END_COL_COUNT = 6
X_CO_COL_COUNT = 8
Y_CO_COL_COUNT = 8
Z_CO_COL_COUNT = 8
OPA_COL_COUNT = 6
TEMP_COL_COUNT = 10
SYST_COL_COUNT = 4

PDB = 'example_all_atoms.pdb'

with open(PDB, newline='') as f:
    pdb_file = csv.reader(f, delimiter='\t')
    pdb_example = [line for line in pdb_file]

print(f'length of pdb file: {len(pdb_example)}')
model_count = 0
pdb_out = ''
stop_loop = 0
for i in predicted_str:
    counter = 0
    
    model_count+=1
    df_str = get_df(i)
    for line in pdb_example:
        if 'MODEL' in line[0]:
            num_search = re.sub(r'\d',str(model_count),line[0])
            pdb_out += num_search + '\n'
        elif 'ATOM' in line[0]:
            line = line[0].split()
            predicted_coordinates = list(df_str.iloc[counter])
            
            predicted_coordinates = [str("{:.3f}".format(i)) for i in predicted_coordinates]

            # find the decimal location
            x_decimal = predicted_coordinates[0].rfind('.') 
            y_decimal = predicted_coordinates[1].rfind('.')
            z_decimal = predicted_coordinates[2].rfind('.')
            if x_decimal <2:
                line[6] = ' ' +predicted_coordinates[0]
            else:
                line[6] = predicted_coordinates[0]
            if y_decimal <2:
                line[7] = ' ' +predicted_coordinates[1]
            else:
                line[7] = predicted_coordinates[1]
            if z_decimal <2:
                line[8] = ' ' +predicted_coordinates[2]
            else:
                line[8] = predicted_coordinates[2]

            line[7] = predicted_coordinates[1]
            line[8] = predicted_coordinates[2]
            # ATOM COLUMN
            ATOM_COL = line[0] + (' '*(ATOM_COL_COUNT-len(line[0])))
            ATOM_NUM = (' '*(ATOM_NUM_COL_COUNT-len(line[1]))) + line[1] #+ (' '*(ATOM_NUM_END_COL_COUNT))
            
            if len(line[2]) >= 4:
                ATOM_SYMB = (' '*(ATOM_SYMB_COL_COUNT-len(line[2]))) + line[2] + (' '*(ATOM_SYMB_END_COL_COUNT))
            else: # hard coding the seniro where the atom symol is less than 4
                PRE_SYMB = 2
                
                ATOM_SYMB_COL_COUNT_modified = ATOM_SYMB_COL_COUNT - PRE_SYMB
                ATOM_SYMB = (' '*PRE_SYMB) + line[2] + (' '*(ATOM_SYMB_COL_COUNT_modified-len(line[2]))) + (' '*(ATOM_SYMB_END_COL_COUNT))


            THREE_LETTER = line[3] + (' '*(THREE_LETTER_COL_COUNT-len(line[3])))
            CHAIN_ID = line[4] #+ (' '*(CHAIN_ID_COL_COUNT-len(line[4])))
            RESIDUE_NUM = (' '*(RESIDUE_NUM_COL_COUNT-len(line[5]))) + line[5] + (' '*(RESIDUE_NUM_END_COL_COUNT))
            X_CO = line[6] + (' '*(X_CO_COL_COUNT-len(line[6])))
            Y_CO = line[7] + (' '*(Y_CO_COL_COUNT-len(line[7])))
            Z_CO = line[8] + (' '*(Z_CO_COL_COUNT-len(line[8])))
            OPA = line[9] + (' '*(OPA_COL_COUNT-len(line[9])))
            TEMP = line[10] + (' '*(TEMP_COL_COUNT-len(line[10])))
            SYST = line[11] + (' '*(SYST_COL_COUNT)) 
            
            pdb_out += ATOM_COL + ATOM_NUM + ATOM_SYMB + THREE_LETTER + CHAIN_ID + RESIDUE_NUM + X_CO + Y_CO + Z_CO + OPA + TEMP + SYST + '\n'

            counter +=1
        elif 'ENDMDL' in line[0]:
            pdb_out += line[0] + '\n'
        stop_loop =+ 1



# with open('predicted_500_pdbs_using_E16.pdb', 'w') as f:
# with open('atomic_LSTM.pdb', 'w') as f:
with open('atomic_all_reconstructed.pdb', 'w') as f:
    f.write(pdb_out)





# # the start location of predicted structure
# # start_len = 10778696

# # end of predicted structure 
# # end_len = 12123002


# # cat -n yourfile | sed -n '10778696,12123002p'
