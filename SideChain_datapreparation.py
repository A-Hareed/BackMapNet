import numpy as np

arr_data = np.load('cluster_8_SC.npy')
print(arr_data.shape)

with open('sequence_1J4N.txt', 'r') as f:
    sequence = f.read()
    sequence = sequence.split(',')

no_Atoms= {
 'CYS':[2,[0]],
 'ALA':[1,[0]],
 'MET':[4,[0]],
 'ASP':[4,[0]],
 'ASN':[4,[0]],
 'ARG':[7,['0_3','3_8']],
 'GLN':[5,[0]],
 'GLU':[5,[0]],
 'GLY':[0,[0]],
 'HIS':[6,['0_2','2_4','4_6']],
 'ILE':[4,[0]],
 'LEU':[4,[0]],
 'LYS':[5,['0_3','3_5']],
 'PHE':[7,['0_3','3_5','5_7']],
 'PRO':[3,[0]],
 'SER':[2,[0]],
 'THR':[3,[0]],
 'TRP':[10,['0,1,9','2_5','7,8','5,6']],
 'TYR':[8,['0_3','3_6','6_8']],
 'VAL':[3,[0]]
 }

print(no_Atoms)


arr = np.array([[1, 2], 
                [3, 4]])
start_indx = 0

for i in range(0,len(sequence)):
    number_at = no_Atoms[sequence[i]]
    if isinstance(number_at[1][0], int):
        print(number_at[1][0],number_at[0]*3)
        padding_amount = 15 - (number_at[0]*3)
        end_indx = start_indx + ( number_at[0]*3)
        temp_arr = np.pad(arr_data[:,start_indx:end_indx], pad_width=((0, 0), (0, padding_amount)), mode='constant', constant_values=0)
        print(temp_arr.shape, arr_data[:,start_indx:end_indx].shape)
        if 'final_arr' in globals():
            final_arr = np.concatenate((final_arr,temp_arr),axis=1)
        else:
            final_arr = temp_arr

print(final_arr.shape, final_arr.reshape(-1,15).shape)
padded_arr = np.pad(arr, pad_width=((0, 0), (0, 3)), mode='constant', constant_values=0)
print(temp_arr[0])
# print(padded_arr,padded_arr.shape)