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
 'ARG':[7,['0,1,2,3,4,5,6,7,8','9,10,11,12,13,14,15,16,17,18,19,20']],
 'GLN':[5,[0]],
 'GLU':[5,[0]],
 'GLY':[0,[0]],
 'HIS':[6,['0,1,2,3,4,5','6,7,8,9,10,11','12,13,14,15,16,17']],
 'ILE':[4,[0]],
 'LEU':[4,[0]],
 'LYS':[5,['0,1,2,3,4,5,6,7,8','9,10,11,12,13,14']],
 'PHE':[7,['0,1,2,3,4,5,6,7,8','9,10,11,12,13,14','5,16,17,18,19,20']],
 'PRO':[3,[0]],
 'SER':[2,[0]],
 'THR':[3,[0]],
 'TRP':[10,['0,1,2,3,4,5,27,28,29','6,7,8,9,10,11,12,13,14','21,22,23,24,25,26','15,16,17,18,19,20']],
 'TYR':[8,['0[0,1,2],1[3,4,5],2[6,7,8]','3[9,10,11],4[12,13,14],5[15,16,17]','6[18,19,20],7[21,22,23']],
 'VAL':[3,[0]]
 }

print(no_Atoms)


arr = np.array([[1, 2], 
                [3, 4]])
start_indx = 0

for i in range(0,len(sequence)):
    number_at = no_Atoms[sequence[i]]
    end_indx = start_indx + ( number_at[0]*3)
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
                padding_amount = 15 - (len(bead)*3)
                singel_bead = np.pad(residue_arr[:,bead], pad_width=((0, 0), (0, padding_amount)), mode='constant', constant_values=0)
                    
            elif '_' in bead:
                
                slicing = [int(j) for j in bead.split('_')]
                padding_amount = 15 - ((slicing[1]-slicing[0])*3)
                singel_bead = np.pad(residue_arr[:,slicing[0]:slicing[1]], pad_width=((0, 0), (0, padding_amount)), mode='constant', constant_values=0)
                print(padding_amount, singel_bead.shape, slicing,residue_arr[:,slicing[0]:slicing[1]].shape)

            if num == 0:
                temp_arr = np.copy(singel_bead)
            elif num >0:
                temp_arr = np.concatenate((temp_arr,singel_bead),axis=1)

        if 'final_arr' in globals():
            final_arr = np.concatenate((final_arr,temp_arr),axis=1)
        else:
            final_arr = temp_arr


    start_indx=end_indx
['0_3','3_6','6_8']
print(final_arr.shape,final_arr.reshape(-1,15).shape)
print(final_arr[0,-15:],temp_arr[0,:])
padded_arr = np.pad(arr, pad_width=((0, 0), (0, 3)), mode='constant', constant_values=0)
print(temp_arr[0],start_indx)
# print(padded_arr,padded_arr.shape)