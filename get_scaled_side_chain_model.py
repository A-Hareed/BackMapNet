import numpy as np
import sys
import re

with open(f'sequence_{sys.argv[2]}.txt', 'r') as f:
    sequence = f.read()
    sequence = sequence.split(',')

"""
Notes for the script:
sys 1 is the cluster file
sys 2 is the pdb name
"""

cluster_file = sys.argv[1]
arr_data = np.load(cluster_file).astype(float)

pattern = r"Feature_array/cluster_(\d+)_CG_SC.npy"
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

def scale_and_center_fragments(fragments, box_size, input_shape):


    # Reshape fragments based on the input shape and reshape_shape
    reshaped_fragments = fragments

    # Calculate maximum absolute coordinate value for each fragment
    max_coords = np.max(np.abs(reshaped_fragments), axis=tuple(range(1, reshaped_fragments.ndim)))

   # Calculate element-wise scaling factors
    scaling_factors = np.minimum(box_size[0] / (2 * max_coords), box_size[1] / (2 * max_coords), box_size[2] / (2 * max_coords))

    # Select minimum scaling factor across dimensions for each fragment
#    scaling_factors = np.where(box_size[1] / (2 * max_coords) < scaling_factors, box_size[1] / (2 * max_coords), scaling_factors)
    scaling_factors = np.where(box_size[2] / (2 * max_coords) < scaling_factors, box_size[2] / (2 * max_coords), scaling_factors)
    # Scale fragments individually
    scaled_fragments = reshaped_fragments * scaling_factors[:, np.newaxis, np.newaxis]

    # Calculate geometric center of each fragment
 #   fragment_centers = np.mean(scaled_fragments, axis=1)

    # Center fragments within the box
  #  centered_fragments = scaled_fragments - fragment_centers[:, np.newaxis, :] + np.array([box_size[0] / 2, box_size[1] / 2, box_size[2] / 2]).reshape(1, 1, 3)

    # Reshape back to the original input shape
    centered_fragments = scaled_fragments.reshape(input_shape)

    return centered_fragments, scaling_factors, "individual"


start_indx = 0

scaling_lst = []
ounter_s = 0

for i in range(0,len(sequence)):
    number_at = no_Atoms[sequence[i]]
    end_indx = start_indx + ( number_at[0]*3)
    # print(i)
    B_score = BLOSUM_60[three_to_one[sequence[i]]]
    if number_at[0] == 1:
        new_lst = B_score.copy()
#        print(number_at[2])
        new_lst.append(number_at[2][0])
        bead_info = np.array([new_lst])
        # Initial array with shape (1, 2)

        expanded_array = np.tile(bead_info, (arr_data.shape[0], 1))
        # print(expanded_array.shape,bead_info.shape,arr_data[:,start_indx:end_indx].shape)



        box_size = (5, 5, 5)
        input_shape = arr_data[:,start_indx:end_indx].shape
#        print('shape of input shape',input_shape)
        array_CG, scaling_factor, centering_method = scale_and_center_fragments(arr_data[:,start_indx:end_indx].reshape(-1,number_at[0],3),box_size,input_shape)
        array_CG = array_CG 
        scaling_lst.append(scaling_factor)
        counter_s+=1 
  #      print(scaling_factor.reshape(-1,1).shape,counter_s)

        temp_arr = np.concatenate((array_CG,expanded_array),axis=1)

        if 'final_arr' in globals():
            final_arr = np.concatenate((final_arr,temp_arr),axis=1)
        else:
            final_arr = temp_arr
 
    else:
        residue_arr = arr_data[:,start_indx:end_indx]
        box_size = (5, 5, 5)

#        array_CG, scaling_factor, centering_method = scale_and_center_fragments(residue_arr.reshape(-1,number_at[0],3),box_size,residue_arr.shape)
#        array_CG = array_CG 
#        scaling_lst.append(scaling_factor)

        for num in range(len(number_at[1])):
            new_lst = B_score.copy()
            # new_lst.extend(number_at[2])

            if num == 0:
                # new_lst = B_score.copy()

                RBF = [1.0]
                new_lst.extend(RBF)


                bead_info = np.array([new_lst])
                expanded_array = np.tile(bead_info, (arr_data.shape[0], 1))
                c_point = residue_arr[:,:3]
#                c_point = array_CG[:,:3]
                array_CG, scaling_factor, centering_method = scale_and_center_fragments(c_point.reshape(-1,1,3),box_size,c_point.shape)
                temp_arr = np.concatenate((array_CG,expanded_array),axis=1)
                scaling_lst.append(scaling_factor)
                counter_s+=1 
 #               print(scaling_factor.reshape(-1,1).shape,counter_s)

            else:
                # print(number_at)
                splicer = [int(j) for j in number_at[1][num].split(',')]
                RBF = calculate_rbf(c_point,residue_arr[:,splicer])


                # print('RBF start')
                # print(RBF, RBF.shape)
                # print('RBF end')
                # bead_info = np.array([[number_at[2][1][num]]])

                bead_info = np.array([new_lst])

                expanded_array = np.tile(bead_info, (arr_data.shape[0], 1))
                # print(expanded_array[:,:-1].shape,RBF.shape,expanded_array[:,-1].shape)
                expanded_array = np.concatenate((expanded_array[:,:-1],RBF,expanded_array[:,-1].reshape(-1,1)),axis=1)

                c_point = residue_arr[:,splicer]
                array_CG, scaling_factor, centering_method = scale_and_center_fragments(c_point.reshape(-1,1,3),box_size,c_point.shape)
                scaling_lst.append(scaling_factor)
                counter_s+=1 
#                print(scaling_factor.reshape(-1,1).shape,counter_s)
                t1 = np.concatenate((array_CG,expanded_array),axis=1)
                temp_arr = np.concatenate((temp_arr,t1),axis=1)
                # print(temp_arr.shape,len(number_at[1]))



        if 'final_arr' in globals():
            final_arr = np.concatenate((final_arr,temp_arr),axis=1)
        else:
            final_arr = temp_arr
  
    start_indx=end_indx

scaling_lst = [scal.reshape(-1,1) for scal in scaling_lst]


scaling_arr = np.concatenate(scaling_lst,axis=1)
np.save(f'scaling_lst_{sys.argv[2]}_clust_perBead_{number}.npy',scaling_arr)


    # start_indx=end_indx

print(final_arr.shape,final_arr.reshape(-1,24).shape,f'shape of cluster {number}')
#print(final_arr.reshape(-1,25)[:10,:])
#print(final_arr.reshape(-1,5)[:,3].max(),final_arr.reshape(-1,5)[:,3].min())
#print(final_arr.reshape(-1,5)[:,4].max(),final_arr.reshape(-1,5)[:,4].min())
#print(final_arr.reshape(-1,5)[:,0].max(),final_arr.reshape(-1,5)[:,0].min())



cluster_saved_file = f"cluster_{number}_SC_CG_RBF.npy"

#exit()
np.save(cluster_saved_file,final_arr)

