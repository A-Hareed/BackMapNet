import numpy as np


def weighted_reconstruct_sequence(predicted_windows, window_size, stride, original_length):
    # Initialize the arrays
    reconstructed = np.zeros(original_length)
    count = np.zeros(original_length)
    
    # Create a weighting vector
    weights = np.linspace(0, 1, stride)
    weights = np.concatenate([weights, weights[::-1][1:-1]])  # Symmetric weights
    weights = np.pad(weights, (0, window_size - len(weights)), 'constant', constant_values=1)  # Pad to window size
    
    num_windows = predicted_windows.shape[0]
    
    # Overlap and add with weights
    for i in range(num_windows):
        start = i * stride
        end = start + window_size
        reconstructed[start:end] += predicted_windows[i] * weights
        count[start:end] += weights
    
    # Handle places where count is zero to avoid division by zero
    count[count == 0] = 1
    
    # Average the overlapping regions
    reconstructed /= count
    
    return reconstructed

# Example usage
predicted_windows = np.random.rand(1651, 48)  # Random predicted windows
window_size = 48
stride = 12
original_length = 19764

reconstructed_sequence = weighted_reconstruct_sequence(predicted_windows, window_size, stride, original_length)

# Print the reconstructed sequence
print(reconstructed_sequence)


#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------


def reconstruct_sequence(predicted_windows, window_size, stride, original_length):
    # Initialize the arrays
    reconstructed = np.zeros(original_length)
    count = np.zeros(original_length)
    
    num_windows = predicted_windows.shape[0]
    
    # Overlap and add
    for i in range(num_windows):
        start = i * stride
        end = start + window_size
        reconstructed[start:end] += predicted_windows[i]
        count[start:end] += 1
    
    # Handle places where count is zero to avoid division by zero
    count[count == 0] = 1
    
    # Average the overlapping regions
    reconstructed /= count
    
    return reconstructed

# Example usage
predicted_windows = np.random.rand(1651, 48)  # Random predicted windows
window_size = 48
stride = 12
original_length = 19764

reconstructed_sequence = reconstruct_sequence(predicted_windows, window_size, stride, original_length)

# Print the reconstructed sequence
print(reconstructed_sequence)
