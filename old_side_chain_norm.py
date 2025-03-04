start_indx = 0

scaling_lst = []

counter_s = 0

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


        
        box_size = (10, 10, 10)
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
        box_size = (10, 10, 10)

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
np.save(f'scaling_lst_1J4N_clust_perBead_{number}.npy',scaling_arr)




    # start_indx=end_indx

print(final_arr.shape,final_arr.reshape(-1,24).shape,f'shape of cluster {number}')
#print(final_arr.reshape(-1,25)[:10,:])
#print(final_arr.reshape(-1,5)[:,3].max(),final_arr.reshape(-1,5)[:,3].min())
#print(final_arr.reshape(-1,5)[:,4].max(),final_arr.reshape(-1,5)[:,4].min())
#print(final_arr.reshape(-1,5)[:,0].max(),final_arr.reshape(-1,5)[:,0].min())



cluster_saved_file = f"cluster_{number}_SC_CG_RBF.npy"

#exit()
np.save(cluster_saved_file,final_arr)
