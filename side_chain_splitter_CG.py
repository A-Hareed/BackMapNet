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
 'CYS':[1,['0,1,2'],[1.0,1]],
 'ALA':[1,['0,1,2'],[1.0,2]],
 'MET':[1,['0,1,2'],[1.0,1]],
 'ASP':[1,['0,1,2'],[1.0,3]],
 'ASN':[1,['0,1,2'],[1.0,3]],
 'ARG':[2,['0,1,2','3,4,5'],[1,[1,3]]],
 'GLN':[1,['0,1,2'],[1.0,2]],
 'GLU':[1,['0,1,2'],[1.0,3]],
 'GLY':[0,[0]],
 'HIS':[3,['0,1,2','3,4,5','6,7,8'],[1,[0,2,2]]],
 'ILE':[1,['0,1,2'],[1.0,0]],
 'LEU':[1,['0,1,2'],[1.0,0]],
 'LYS':[2,['0,1,2','3,4,5'],[1,[0,3]]],
 'PHE':[3,['0,1,2','3,4,5','6,7,8'],[1[0,0,0]]],
 'PRO':[1,['0,1,2'],[1.0,0]],
 'SER':[1,['0,1,2'],[1.0,2]],
 'THR':[1,['0,1,2'],[1.0,2]],
 'TRP':[4,['0,1,2','3,4,5','6,7,8','9,10,11'],[1,[0,2,0,0]]],
 'TYR':[3,['0,1,2','3,4,5','6,7,8'],[1,[0,0,2]]],
 'VAL':[1,['0,1,2'],[1.0,0]]
 }


import numpy as np

def calculate_rbf(array1, array2, gamma=0.1):
    """
    Calculate the Radial Basis Function (RBF) between two arrays of points.

    Parameters:
    array1 (ndarray): First array of points with shape (n, 3).
    array2 (ndarray): Second array of points with shape (n, 3).
    gamma (float): Parameter that controls the spread of the RBF (default is 0.1).

    Returns:
    ndarray: RBF values for each corresponding pair of points in the input arrays with shape (n, 1).
    """
    # Validate input shapes
    if array1.shape != array2.shape:
        raise ValueError("Input arrays must have the same shape.")

    # Calculate the squared Euclidean distances between corresponding rows
    squared_distances = np.sum((array1 - array2) ** 2, axis=1)

    # Calculate the RBF values using the Gaussian function
    rbf_values = np.exp(-gamma * squared_distances)

    # Reshape the output to (n, 1)
    rbf_values = rbf_values.reshape(-1, 1)

    return rbf_values




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
                RBF = 1.0
                bead_info = np.array([[RBF,number_at[2][1][num]]])
                expanded_array = np.tile(bead_info, (arr_data.shape[0], 1))
                c_point = residue_arr[:,:3]
                temp_arr = np.concatenate((residue_arr[:,:3],expanded_array),axis=1)
            else:
                RBF = calculate_rbf(c_point,residue_arr[:,number_at[1][num]])
                bead_info = np.array([[number_at[2][1][num]]])
                expanded_array = np.tile(bead_info, (arr_data.shape[0], 1))
                expanded_array = np.concatenate((RBF,expanded_array),axis=1)

                t1 = np.concatenate((residue_arr[:,number_at[1][num]],expanded_array),axis=1)
                temp_arr = np.concatenate((temp_arr,t1),axis=1)

        if 'final_arr' in globals():
            final_arr = np.concatenate((final_arr,temp_arr),axis=1)
        else:
            final_arr = temp_arr


    start_indx=end_indx




    start_indx=end_indx




cluster_saved_file = f"cluster_{number}_SC_CG.npy"


np.save(cluster_saved_file,final_arr)
