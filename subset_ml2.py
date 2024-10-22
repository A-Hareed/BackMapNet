import numpy as np
from sliding_window import create_feature_set


feat_train = np.load('training_feat_1J4N_subsetDim.npy')
target_train = np.load('training_targets_1J4N_subsetDim.npy')

feat_test = np.load('testing_feat_1J4N_subsetDim.npy')
target_test = np.load('testing_targets_1J4N_subsetDim.npy')



print(feat_train.shape)
print(target_train.shape)

def scale_and_center_fragments(fragments, box_size):
    """Scales and centers fragments within a specified box size.

    Args:
        fragments: A NumPy array of shape (num_fragments, 384) representing fragment coordinates.
        box_size: A tuple (x, y, z) specifying the dimensions of the box.

    Returns:
        A tuple containing:
            - scaled_fragments: A NumPy array of shape (num_fragments, 32, 3) containing the scaled fragments.
            - scaling_factor: A NumPy array of shape (num_fragments,) representing the individual scaling factors for each fragment.
            - centering_method: A string indicating the centering method used.
    """

    # Reshape fragments to (-1, 32, 3)
    reshaped_fragments = fragments.reshape(-1, 32, 3)

    # Calculate maximum absolute coordinate value for each fragment
    max_coords = np.max(np.abs(reshaped_fragments), axis=(1, 2))

    # Calculate element-wise scaling factors
    scaling_factors = np.minimum(box_size[0] / (2 * max_coords), box_size[1] / (2 * max_coords), box_size[2] / (2 * max_coords))

    # Select minimum scaling factor across dimensions for each fragment
    scaling_factors = np.where(box_size[1] / (2 * max_coords) < scaling_factors, box_size[1] / (2 * max_coords), scaling_factors)
    scaling_factors = np.where(box_size[2] / (2 * max_coords) < scaling_factors, box_size[2] / (2 * max_coords), scaling_factors)

    # Scale fragments individually
    scaled_fragments = reshaped_fragments * scaling_factors[:, np.newaxis, np.newaxis]

    # Calculate geometric center of each fragment
    fragment_centers = np.mean(scaled_fragments, axis=1)

    # Center fragments within the box
    centered_fragments = scaled_fragments - fragment_centers[:, np.newaxis, :] + np.array([box_size[0] / 2, box_size[1] / 2, box_size[2] / 2]).reshape(-1, 1, 3)

    return centered_fragments, scaling_factors, "individual"

def apply_scaling_and_centering(target_fragments, scaling_factor, centering_method):
    """Applies the specified scaling factor and centering method to target fragments.

    Args:
        target_fragments: A NumPy array of shape (num_target_fragments, 384) representing target fragment coordinates.
        scaling_factor: The scaling factor to apply.
        centering_method: A string indicating the centering method to apply.

    Returns:
        A NumPy array of shape (num_target_fragments, 32, 3) containing the scaled and centered target fragments.
    """

    # Reshape target fragments to (-1, 32, 4, 3)
    reshaped_target_fragments = target_fragments.reshape(-1, 32, 4, 3)

    # Scale target fragments
    scaled_target_fragments = reshaped_target_fragments * scaling_factor[:, np.newaxis, np.newaxis, np.newaxis]

    # Center target fragments based on centering method
    if centering_method == "individual":
        target_fragment_centers = np.mean(scaled_target_fragments, axis=(1, 2))
        centered_target_fragments = scaled_target_fragments - target_fragment_centers[:, np.newaxis, np.newaxis, :] + np.array([box_size[0] / 2, box_size[1] / 2, box_size[2] / 2]).reshape(-1, 1, 3)
    else:
        raise ValueError("Invalid centering method")

    return centered_target_fragments.reshape(-1, 384)


def process_and_save_batches(data_feat,data_target, window_size, step_size,window_size2,step_size2, batch_size, output_prefix,output_prefix2, pdb_name, chain_number,scale_and_center_fragments,apply_scaling_and_centering,create_feature_set):
    num_samples = data_feat.shape[0]
    num_batches = (num_samples - 1) // batch_size + 1

    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, num_samples)
        batch_data = data_feat[start:end,:]
        batch_data2 = data_target[start:end,:]

        # Process the batch
        print(batch_data.shape,batch_data2.shape)

        batch_features = create_feature_set(batch_data, window_size, 3)
        batch_LAB = create_feature_set(batch_data2, window_size2, 12)

        # normalise
        box_size = (50, 50, 50)
        array_CG, scaling_factor, centering_method = scale_and_center_fragments(batch_features,box_size)
        print(array_CG.shape,scaling_factor.shape,batch_LAB.shape,batch_features.shape)
        array_AA = apply_scaling_and_centering(batch_LAB,scaling_factor,centering_method)

        # Save the batch with proper naming
        np.save(f'{output_prefix}_B{i+1}_{pdb_name}_chain{chain_number}.npy', array_CG)
        print(f'Saved batch {i+1} to {output_prefix}_B{i+1}_{pdb_name}_chain{chain_number}.npy')
        
        # Save the batch with proper naming
        np.save(f'{output_prefix2}_B{i+1}_{pdb_name}_chain{chain_number}.npy', array_AA)
        print(f'Saved batch {i+1} to {output_prefix2}_B{i+1}_{pdb_name}_chain{chain_number}.npy')

        # Free memory
        del batch_data, batch_features, array_AA,array_CG,scaling_factor
        print(f'Cleared memory for batch {i+1}')


 




slice_feat = [747, 1494, 2241]
slice_target = [2988, 5976, 8964]
box_size = (50, 50, 50)



train_Lab_a1 = target_train[:, :slice_target[0]]
train_Lab_a2 = target_train[:, slice_target[0]:slice_target[1]]
train_Lab_a3 = target_train[:, slice_target[1]:slice_target[2]]
train_Lab_a4 = target_train[:, slice_target[2]:]

train_feat_a1 = feat_train[:, :slice_feat[0]]
train_feat_a2 = feat_train[:, slice_feat[0]:slice_feat[1]]
train_feat_a3 = feat_train[:, slice_feat[1]:slice_feat[2]]
train_feat_a4 = feat_train[:, slice_feat[2]:]

test_Lab_a1 = target_test[:, :slice_target[0]]
test_Lab_a2 = target_test[:, slice_target[0]:slice_target[1]]
test_Lab_a3 = target_test[:, slice_target[1]:slice_target[2]]
test_Lab_a4 = target_test[:, slice_target[2]:]

test_feat_a1 = feat_test[:, :slice_feat[0]]
test_feat_a2 = feat_test[:, slice_feat[0]:slice_feat[1]]
test_feat_a3 = feat_test[:, slice_feat[1]:slice_feat[2]]
test_feat_a4 = feat_test[:, slice_feat[2]:]
print(train_feat_a1.shape,train_Lab_a1.shape)
window_size = 32
batch_size = 10000  # Adjust as needed
pdb_name = "1J4N"
process_and_save_batches(train_feat_a1,train_Lab_a1, window_size * 3, 3,window_size * 12, 12, batch_size, 'train_feat','train_LAB', pdb_name, 1,scale_and_center_fragments,apply_scaling_and_centering,create_feature_set)
process_and_save_batches(train_feat_a2,train_Lab_a2, window_size * 3, 3,window_size * 12, 12, batch_size, 'train_feat','train_LAB', pdb_name, 2,scale_and_center_fragments,apply_scaling_and_centering,create_feature_set)
process_and_save_batches(train_feat_a3,train_Lab_a3, window_size * 3, 3,window_size * 12, 12, batch_size, 'train_feat', 'train_LAB',pdb_name, 3,scale_and_center_fragments,apply_scaling_and_centering,create_feature_set)
process_and_save_batches(train_feat_a4,train_Lab_a4, window_size * 3, 3,window_size * 12, 12, batch_size, 'train_feat','train_LAB', pdb_name, 4,scale_and_center_fragments,apply_scaling_and_centering,create_feature_set)


# Process and save testing data in batches

process_and_save_batches(test_feat_a1,test_Lab_a1, window_size * 3, 3, window_size * 12, 12, batch_size, 'test_feat','test_LAB', pdb_name, 1,scale_and_center_fragments,apply_scaling_and_centering)
process_and_save_batches(test_feat_a2,test_Lab_a2, window_size * 3, 3, window_size * 12, 12, batch_size, 'test_feat','test_LAB', pdb_name, 2,scale_and_center_fragments,apply_scaling_and_centering)
process_and_save_batches(test_feat_a3,test_Lab_a3, window_size * 3, 3, window_size * 12, 12, batch_size, 'test_feat','test_LAB', pdb_name, 3,scale_and_center_fragments,apply_scaling_and_centering)
process_and_save_batches(test_feat_a4,test_Lab_a4,  window_size * 3, 3, window_size * 12, 12, batch_size, 'test_feat','test_LAB', pdb_name, 4,scale_and_center_fragments,apply_scaling_and_centering)


#test_array_CG, scaling_factor, centering_method = scale_and_center_fragments(feat,box_size)
#print(test_array_CG.shape)
#print(test_array_CG.max(), test_array_CG.min())
#print(scaling_factor,scaling_factor.shape)

#box_size = (50, 50, 50)
#test_array = apply_scaling_and_centering(target,scaling_factor,centering_method)
#print(test_array.shape)
#print(test_array.max(), test_array.min(),target.max())
