

import numpy as np

def create_feature_set(data, window_size=48, step_size=12):
    # Ensure data is a numpy array
    data = np.asarray(data)
    
    # Get the number of windows
    num_windows = (data.shape[1] - window_size) // step_size + 1
    
    # Initialize an empty list to store the windows
    
    sample_size = num_windows*data.shape[0]
    # Loop through the data and extract windows
    feat_arr = np.zeros((sample_size,window_size))
    curr = 0
    for i in range(num_windows):
        start_index = i * step_size
        end_index = start_index + window_size
        window = data[:, start_index:end_index]
       
        # print(window.shape,feat_arr[curr:curr+data.shape[0],:].shape)
        feat_arr[curr:curr+data.shape[0],:] = window
        curr+=data.shape[0]
        
    
    # Convert the list of windows to a numpy array
    
    
    return feat_arr

