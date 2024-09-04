import numpy as np
import sys
import re


with open('sequence_1J4N.txt', 'r') as f:
    sequence = f.read()
    sequence = sequence.split(',')

cluster_file = sys.argv[1]
arr_data = np.load(cluster_file)

pattern = r"cluster_(\d+)_SC.npy"
number = int(re.search(pattern, cluster_file).group(1))

no_Atoms= {
 'CYS':[1,['0,1,2'],[0,1]],
 'ALA':[1,['0,1,2'],[0,2]],
 'MET':[1,['0,1,2'],[0,1]],
 'ASP':[1,['0,1,2'],[0,3]],
 'ASN':[1,['0,1,2'],[0,3]],
 'ARG':[2,['0,1,2','3,4,5'],[1,[1,3]]],
 'GLN':[1,['0,1,2'],[0,2]],
 'GLU':[1,['0,1,2'],[0,3]],
 'GLY':[0,[0]],
 'HIS':[3,['0,1,2','3,4,5','6,7,8'],[1,[0,2,2]]],
 'ILE':[1,['0,1,2'],[0,0]],
 'LEU':[1,['0,1,2'],[0,0]],
 'LYS':[2,['0,1,2','3,4,5'],[1,[0,3]]],
 'PHE':[3,['0,1,2','3,4,5','6,7,8'],[1[0,0,0]]],
 'PRO':[1,['0,1,2'],[0,0]],
 'SER':[1,['0,1,2'],[0,2]],
 'THR':[1,['0,1,2'],[0,2]],
 'TRP':[4,['0,1,2','3,4,5','6,7,8','9,10,11'],[1,[0,2,0,0]]],
 'TYR':[3,['0,1,2','3,4,5','6,7,8'],[1,[0,0,2]]],
 'VAL':[1,['0,1,2'],[0,0]]
 }




start_indx = 0

for i in range(0,len(sequence)):
    number_at = no_Atoms[sequence[i]]
    end_indx = start_indx + ( number_at[0]*3)

    if number_at[0] == 1:

        bead_info = np.array([[number_at[2]]])
        # Initial array with shape (1, 2)

        expanded_array = np.tile(bead_info, (arr_data.shape[0], 1))


        arr_data[:,start_indx:end_indx]
        temp_arr = np.concatenate((arr_data[:,start_indx:end_indx],expanded_array),axis=1)

        if 'final_arr' in globals():
            final_arr = np.concatenate((final_arr,temp_arr),axis=1)
        else:
            final_arr = temp_arr

    else:
        residue_arr = arr_data[:,start_indx:end_indx]
        for num in range(3):
            if num == 0:
                COM = 0.0
                bead_info = np.array([[COM,number_at[2][1][num]]])
                c_point = residue_arr[:,:3]
            else:
                


            expanded_array = np.tile(bead_info, (residue_arr.shape[0], 1))




    if isinstance(number_at[1][0], int):

        padding_amount = 15 - (number_at[0]*3)
        temp_arr = np.pad(arr_data[:,start_indx:end_indx], pad_width=((0, 0), (0, padding_amount)), mode='constant', constant_values=0)
        if 'final_arr' in globals():
            final_arr = np.concatenate((final_arr,temp_arr),axis=1)
        else:
            final_arr = temp_arr
    else:
        residue_arr = arr_data[:,start_indx:end_indx]
        for num, bead in enumerate(number_at[1]):
            if ',' in bead:
                bead = [int(j) for j in bead.split(',')]
                padding_amount = 15 - (len(bead))
                singel_bead = np.pad(residue_arr[:,bead], pad_width=((0, 0), (0, padding_amount)), mode='constant', constant_values=0)
            elif '_' in bead:
                
                slicing = [int(j) for j in bead.split('_')]
                padding_amount = 15 - ((slicing[1]-slicing[0])*3)
                singel_bead = np.pad(residue_arr[:,slicing[0]:slicing[1]], pad_width=((0, 0), (0, padding_amount)), mode='constant', constant_values=0)

            if num == 0:
                temp_arr = np.copy(singel_bead)
            elif num >0:
                temp_arr = np.concatenate((temp_arr,singel_bead),axis=1)

        if 'final_arr' in globals():
            final_arr = np.concatenate((final_arr,temp_arr),axis=1)
        else:
            final_arr = temp_arr


    start_indx=end_indx




cluster_saved_file = f"cluster_PD_{number}_SC.npy"


np.save(cluster_saved_file,final_arr)
