import numpy as np
import sys


pred_norm = np.load(sys.argv[1])
actual_norm = np.load(sys.argv[2])
custom_min = np.load(sys.argv[3])
custom_range = np.load(sys.argv[4])
chain_num = sys.argv[5]

def sliding_window_reconstruct(sequence, window_size=384, stride=12):
    # Calculate the original length based on num_windows, stride, and window_size
    num_windows, window_size = sequence.shape
    original_length = (num_windows - 1) * stride + window_size

    # Initialize the reconstructed sequence and a count array
    reconstructed = np.zeros(original_length)
    counts = np.zeros(original_length)

    # Add each window to the reconstructed array
    for i in range(num_windows):
        start = i * stride
        end = start + window_size
        reconstructed[start:end] += sequence[i]
        counts[start:end] += 1

    # Avoid division by zero and average values where counts > 0
    counts[counts == 0] = 1
    reconstructed /= counts
#    print(counts)

    return reconstructed.reshape(1,-1)


def calculate_rmsd(array1, array2):
    # Ensure the arrays are the same shape
    print(array1.shape,array2.shape)
    assert array1.shape == array2.shape, "Arrays must be of the same shape."
    
    # Calculate the squared differences
    diff = array1 - array2
    squared_diff = np.square(diff)
    
    # Calculate the mean of the squared differences
    mean_squared_diff = np.mean(squared_diff)
    
    # Take the square root to get the RMSD
    rmsd = np.sqrt(mean_squared_diff)
    
    return rmsd


def reverse_normalize_fragments_per_axis(normalized_fragments, custom_min, custom_range):
    """
    Reverse the custom min-max normalization applied by normalize_fragments_per_axis.

    Parameters:
    normalized_fragments (ndarray): Normalized fragments with shape (n_fragments, n_points, n_dimensions).
    custom_min (ndarray): Custom minimum values per axis with shape (n_fragments, 1, n_dimensions).
    custom_range (ndarray): Range (custom_max - custom_min) per axis with shape (n_fragments, 1, n_dimensions).

    Returns:
    original_fragments (ndarray): The original fragments with the same shape as normalized_fragments.
    """
    original_fragments = (normalized_fragments * custom_range) + custom_min
    return original_fragments


original_fragments_pred = reverse_normalize_fragments_per_axis(pred_norm.reshape(-1,128,3), custom_min, custom_range).reshape(1,-1,384)
original_fragments_actual = reverse_normalize_fragments_per_axis(actual_norm.reshape(-1,128,3), custom_min, custom_range).reshape(1,-1,384)
diff = original_fragments_actual.reshape(-1,3) - original_fragments_pred.reshape(-1,3)



print('print the maximum difference between the actual and predicted value',diff.max(axis=0), diff.min(axis=0),diff.mean(axis=0),diff.shape)

print(original_fragments_pred.reshape(1,-1,384).shape,original_fragments_actual.reshape(1,-1,384).shape)


#yhat_array = np.empty((400, 6552))
#actual_array = np.empty((400, 6552))


for loca in range(1):
    temp = original_fragments_pred[loca,:,:].reshape(-1,384)
    temp_ori = sliding_window_reconstruct(temp)

    if loca == 0:
        yhat_array = np.empty((1, temp_ori.shape[1]))

    yhat_array[loca] = temp_ori
print(yhat_array.shape)


for loca in range(1):
    temp = original_fragments_actual[loca,:,:].reshape(-1,384)
    temp_ori = sliding_window_reconstruct(temp)

    if loca == 0:
        actual_array = np.empty((1, temp_ori.shape[1]))
    actual_array[loca] = temp_ori

diff = yhat_array.reshape(-1,3) - actual_array.reshape(-1,3)


print('print the maximum difference between the actual and predicted value',diff.max(axis=0), diff.min(axis=0),diff.mean(axis=0),diff.shape)


print(actual_array[0].reshape(-1,3).max(),actual_array[0].reshape(-1,3).min())
print(yhat_array[0].reshape(-1,3).max(),yhat_array[0].reshape(-1,3).min())

lst_rmsd = []

for frame in range(1):
    rmsd_actual = calculate_rmsd(actual_array[frame].reshape(-1,3), yhat_array[frame].reshape(-1,3))
    print('actual rmsd is this: ',rmsd_actual)
    lst_rmsd.append(rmsd_actual)


np.save(f'actual_chain{sys.argv[4]}',actual_array)
np.save(f'pred_chain{sys.argv[4]}',yhat_array)


arr_rmsd = np.array(lst_rmsd).reshape(1,-1)
print(arr_rmsd.shape)


print(arr_rmsd.shape)
exit()
#np.save('arr_rmsd.npy',arr_rmsd)

ori_rmsd = np.load('arr_rmsd.npy')

ori_rmsd = np.concatenate((ori_rmsd,arr_rmsd))
print(ori_rmsd.shape)
#np.save('arr_rmsd.npy',ori_rmsd)




