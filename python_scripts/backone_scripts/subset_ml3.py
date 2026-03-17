import numpy as np
from sliding_window import create_feature_set
from no_overlap import create_nonoverlapping_windows_endaligned
import sys

pdb = sys.argv[1]



feat_train = np.load(f'cluster_1_CG.npy').astype('float')
target_train = np.load(f'cluster_1.npy').astype('float')



print(feat_train.shape)
print(target_train.shape)

def normalize_fragments_per_axis(fragments):
    """
    Normalize each fragment along each axis using custom min-max normalization,
    and save custom_min and range (custom_max - custom_min) for reverse normalization.

    For each fragment and for each coordinate axis (x, y, z), the normalization is:
        normalized_value = (value - (min - 4)) / ((max + 4) - (min - 4))

    Parameters:
    fragments (ndarray): Array of fragments with shape (n_fragments, n_points, n_dimensions).

    Returns:
    normalized_fragments (ndarray): Normalized fragments with the same shape as input.
    custom_min (ndarray): Custom minimum values per axis with shape (n_fragments, 1, n_dimensions).
    custom_range (ndarray): Range (custom_max - custom_min) per axis with shape (n_fragments, 1, n_dimensions).
    """
    # Calculate the minimum and maximum along the points axis for each fragment and each coordinate
    absolute_min = np.min(fragments, axis=1, keepdims=True)  # shape: (n_fragments, 1, n_dimensions)
    absolute_max = np.max(fragments, axis=1, keepdims=True)  # shape: (n_fragments, 1, n_dimensions)
    
    # Adjust the min and max by ±4 for each axis
    custom_min = absolute_min - 4
    custom_max = absolute_max + 4
    
    # Calculate the range (custom_max - custom_min) for each axis
    custom_range = custom_max - custom_min
    
    # Normalize using the custom min-max per axis
    normalized_fragments = (fragments - custom_min) / custom_range
    
    return normalized_fragments, custom_min, custom_range
    




def process_and_save_batches(data_feat,data_target, window_size, step_size,window_size2,step_size2, batch_size, output_prefix,output_prefix2, pdb_name,chain_number, normalize_fragments_per_axis,create_feature_set):
    num_samples = data_feat.shape[0]
    num_batches = (num_samples - 1) // batch_size + 1
    print(data_feat.shape,data_target.shape)


    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, num_samples)
        batch_data = data_feat[start:end,:]
        batch_data2 = data_target[start:end,:]

        # Process the batch
        print(batch_data.shape,batch_data2.shape)


        batch_features = create_nonoverlapping_windows_endaligned(batch_data, window_size).reshape(-1,32,3)
#        batch_features = create_feature_set(batch_data, window_size, 3).reshape(-1,32,3)

        batch_LAB = create_nonoverlapping_windows_endaligned(batch_data2, window_size2).reshape(-1,128,3)
#        batch_LAB = create_feature_set(batch_data2, window_size2*12).reshape(-1,128,3)


        # normalise
        box_size = (50, 50, 50)
        #array_CG, scaling_factor, centering_method = scale_and_center_fragments(batch_features,box_size)
        array_CG, custom_min, custom_range = normalize_fragments_per_axis(batch_features)
        
        print(array_CG.shape,custom_min.shape,batch_LAB.shape,batch_features.shape)
        #array_AA = apply_scaling_and_centering(batch_LAB,scaling_factor,centering_method)
        array_AA = (batch_LAB - custom_min) / custom_range
        
        # Save the batch with proper naming
        np.save(f'{output_prefix}_B{i+1}_{pdb_name}_chain{chain_number}.npy', array_CG.reshape(-1,(32*3)))
        print(f'Saved batch {i+1} to {output_prefix}_B{i+1}_{pdb_name}_chain{chain_number}.npy')
        
        # Save the batch with proper naming
        np.save(f'{output_prefix2}_B{i+1}_{pdb_name}_chain{chain_number}.npy', array_AA.reshape(-1,(128*3)))
        print(f'Saved batch {i+1} to {output_prefix2}_B{i+1}_{pdb_name}_chain{chain_number}.npy')

        np.save(f'custom_min_B{i+1}_{pdb_name}_chain{chain_number}.npy', custom_min)
        np.save(f'custom_range_B{i+1}_{pdb_name}_chain{chain_number}.npy', custom_range)
        print(f'the shape of custom min and range is this:{custom_min.shape}, {custom_range.shape}')
        # Free memory
        del batch_data, batch_features, array_AA,array_CG,custom_min, custom_range
        print(f'Cleared memory for batch {i+1}')


 


slice_feat = [1638,2283,3921,4566]
slice_target = [6552,9132,15684,18264]


train_Lab_a1 = target_train[:, :slice_target[0]]
train_Lab_a2 = target_train[:, slice_target[0]:slice_target[1]]
train_Lab_a3 = target_train[:, slice_target[1]:slice_target[2]]
train_Lab_a4 = target_train[:, slice_target[2]:slice_target[3]]
train_Lab_a5 = target_train[:, slice_target[3]:]




train_feat_a1 = feat_train[:, :slice_feat[0]]
train_feat_a2 = feat_train[:, slice_feat[0]:slice_feat[1]]
train_feat_a3 = feat_train[:, slice_feat[1]:slice_feat[2]]
train_feat_a4 = feat_train[:, slice_feat[2]:slice_feat[3]]
train_feat_a5 = feat_train[:, slice_feat[3]:]


for i in [train_feat_a1,train_feat_a2,train_feat_a3,train_feat_a4,train_feat_a5]:
    print(i.shape)

window_size = 32
batch_size = 10000  # Adjust as needed
pdb_name = pdb


process_and_save_batches(train_feat_a1,train_Lab_a1, window_size * 3, 3,window_size * 12, 12, batch_size, 'train_feat','train_LAB', pdb_name, 1,normalize_fragments_per_axis,create_feature_set)
process_and_save_batches(train_feat_a2,train_Lab_a2, window_size * 3, 3,window_size * 12, 12, batch_size, 'train_feat','train_LAB', pdb_name, 2,normalize_fragments_per_axis,create_feature_set)
process_and_save_batches(train_feat_a3,train_Lab_a3, window_size * 3, 3,window_size * 12, 12, batch_size, 'train_feat', 'train_LAB',pdb_name, 3,normalize_fragments_per_axis,create_feature_set)
process_and_save_batches(train_feat_a4,train_Lab_a4, window_size * 3, 3,window_size * 12, 12, batch_size, 'train_feat', 'train_LAB',pdb_name, 4,normalize_fragments_per_axis,create_feature_set)
process_and_save_batches(train_feat_a5,train_Lab_a5, window_size * 3, 3,window_size * 12, 12, batch_size, 'train_feat', 'train_LAB',pdb_name, 5,normalize_fragments_per_axis,create_feature_set)
del train_feat_a1,train_Lab_a1, train_feat_a2,train_Lab_a2, train_feat_a3,train_Lab_a3
# Process and save testing data in batches



#test_array_CG, scaling_factor, centering_method = scale_and_center_fragments(feat,box_size)
#print(test_array_CG.shape)
#print(test_array_CG.max(), test_array_CG.min())
#print(scaling_factor,scaling_factor.shape)

#box_size = (50, 50, 50)
#test_array = apply_scaling_and_centering(target,scaling_factor,centering_method)
#print(test_array.shape)
#print(test_array.max(), test_array.min(),target.max())
