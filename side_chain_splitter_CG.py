import numpy as np
import sys
import re


with open('sequence_1UBQ.txt', 'r') as f:
    sequence = f.read()
    sequence = sequence.split(',')

cluster_file = sys.argv[1]
arr_data = np.load(cluster_file).astype(float)

pattern = r"cluster_(\d+)_CG_SC.npy"
number = int(re.search(pattern, cluster_file).group(1))

# Dictionary to convert three-letter amino acid codes to one-letter codes
three_to_one = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V',
    'TRP': 'W', 'TYR': 'Y'
}

BLOSUM_60 = {
"A": [4, -1, -1, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0] ,
"R": [-1, 5, 0, -1, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -2] ,
"N": [-1, 0, 6, 1, -2, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3] ,
"D": [-2, -1, 1, 6, -3, 0, 2, -1, -1, -3, -3, -1, -3, -3, -1, 0, -1, -4, -3, -3] ,
"C": [0, -3, -2, -3, 9, -3, -3, -2, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1] ,
"Q": [-1, 1, 0, 0, -3, 5, 2, -2, 1, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2] ,
"E": [-1, 0, 0, 2, -3, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2] ,
"G": [0, -2, 0, -1, -2, -2, -2, 6, -2, -3, -4, -1, -2, -3, -2, 0, -2, -2, -3, -3] ,
"H": [-2, 0, 1, -1, -3, 1, 0, -2, 7, -3, -3, -1, -1, -1, -2, -1, -2, -2, 2, -3] ,
"I": [-1, -3, -3, -3, -1, -3, -3, -3, -3, 4, 2, -3, 1, 0, -3, -2, -1, -2, -1, 3] ,
"L": [-1, -2, -3, -3, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1] ,
"K": [-1, 2, 0, -1, -3, 1, 1, -1, -1, -3, -2, 4, -1, -3, -1, 0, -1, -3, -2, -2] ,
"M": [-1, -1, -2, -3, -1, 0, -2, -2, -1, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1] ,
"F": [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1] ,
"P": [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2] ,
"S": [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2] ,
"T": [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 4, -2, -2, 0] ,
"W": [-3, -3, -4, -4, -2, -2, -3, -2, -2, -2, -2, -3, -1, 1, -4, -3, -2, 10, 2, -3] ,
"Y": [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 6, -1] ,
"V": [0, -2, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4] 
}

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
 'PHE':[3,['0,1,2','3,4,5','6,7,8'],[1,[0,0,0]]],
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
    # print(i)
    B_score = BLOSUM_60[three_to_one[sequence[i]]]
    print(sequence[i])
    if number_at[0] == 1:
        new_lst = B_score.copy()
        new_lst.extend(number_at[2])
        bead_info = np.array([new_lst])
        # Initial array with shape (1, 2)

        expanded_array = np.tile(bead_info, (arr_data.shape[0], 1))
        # print(expanded_array.shape,bead_info.shape,arr_data[:,start_indx:end_indx].shape)
       

        arr_data[:,start_indx:end_indx]
        temp_arr = np.concatenate((arr_data[:,start_indx:end_indx],expanded_array),axis=1)

        if 'final_arr' in globals():
            final_arr = np.concatenate((final_arr,temp_arr),axis=1)
        else:
            final_arr = temp_arr
 
    else:
        residue_arr = arr_data[:,start_indx:end_indx]
        for num in range(len(number_at[1])):
            new_lst = B_score.copy()
            # new_lst.extend(number_at[2])

            if num == 0:
                # new_lst = B_score.copy()
          
                RBF = [1.0]
                new_lst.extend(RBF)
                new_lst.append(number_at[2][1][num])
                bead_info = np.array([new_lst])
                expanded_array = np.tile(bead_info, (arr_data.shape[0], 1))
                c_point = residue_arr[:,:3]
               
                temp_arr = np.concatenate((residue_arr[:,:3],expanded_array),axis=1)
              
           
            else:
                # print(number_at)
                splicer = [int(j) for j in number_at[1][num].split(',')]
                RBF = calculate_rbf(c_point,residue_arr[:,splicer])
                
                new_lst.append(number_at[2][1][num])
                # print('RBF start')
                # print(RBF, RBF.shape)
                # print('RBF end')
                # bead_info = np.array([[number_at[2][1][num]]])
               
                bead_info = np.array([new_lst])

                expanded_array = np.tile(bead_info, (arr_data.shape[0], 1))
                print(expanded_array[:,:-1].shape,RBF.shape,expanded_array[:,-1].shape)
                expanded_array = np.concatenate((expanded_array[:,:-1],RBF,expanded_array[:,-1].reshape(-1,1)),axis=1)

                t1 = np.concatenate((residue_arr[:,splicer],expanded_array),axis=1)
                temp_arr = np.concatenate((temp_arr,t1),axis=1)
                print(temp_arr.shape,len(number_at[1]))
    

        if 'final_arr' in globals():
            final_arr = np.concatenate((final_arr,temp_arr),axis=1)
        else:
            final_arr = temp_arr
        
       

  
    start_indx=end_indx




    # start_indx=end_indx

print(final_arr.shape,final_arr.reshape(-1,25).shape)
print(final_arr.reshape(-1,25)[:10,:])
exit()
print(final_arr.reshape(-1,5)[:,3].max(),final_arr.reshape(-1,5)[:,3].min())
print(final_arr.reshape(-1,5)[:,4].max(),final_arr.reshape(-1,5)[:,4].min())
print(final_arr.reshape(-1,5)[:,0].max(),final_arr.reshape(-1,5)[:,0].min())



cluster_saved_file = f"cluster_{number}_SC_CG_RBF.npy"


np.save(cluster_saved_file,final_arr)
