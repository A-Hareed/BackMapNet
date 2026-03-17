

import numpy as np
#import matplotlib.pyplot as plt

def create_feature_set(data, window_size=48, step_size=12):
    # Ensure data is a numpy array
    data = np.asarray(data)
    
    # Get the number of windows
    num_windows = (data.shape[1] - window_size) // step_size + 1
#    print(num_windows, data.shape[1] ,window_size)
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

# Assuming 'history' is the result from model.fit()
#exit()
#def plot_history(history):
#    # Extract data from the history object
#    loss = history.history['loss']
#    val_loss = history.history['val_loss']
#    mae = history.history['mae']
#    val_mae = history.history['val_mae']
#    rmse = history.history['root_mean_squared_error']
#    val_rmse = history.history['val_root_mean_squared_error']

#    epochs = range(1, len(loss) + 1)

#    # Plot Loss
#    plt.figure(figsize=(10, 6))
#    plt.plot(epochs, loss, 'bo-', label='Training Loss')
#    plt.plot(epochs, val_loss, 'r*-', label='Validation Loss')
#    plt.title('Training and Validation Loss')
#    plt.xlabel('Epochs')
#    plt.ylabel('Loss')
#    plt.legend()
#    plt.grid(True)
#    plt.savefig('loss_plot.png', dpi=300)  # Save plot with high quality
#    plt.show()

    # Plot MAE
#    plt.figure(figsize=(10, 6))
#    plt.plot(epochs, mae, 'bo-', label='Training MAE')
#    plt.plot(epochs, val_mae, 'r*-', label='Validation MAE')
#    plt.title('Training and Validation MAE')
#    plt.xlabel('Epochs')
#    plt.ylabel('MAE')
#    plt.legend()
#    plt.grid(True)
#    plt.savefig('mae_plot.png', dpi=300)  # Save plot with high quality
#    plt.show()

    # Plot RMSE
#    plt.figure(figsize=(10, 6))
#    plt.plot(epochs, rmse, 'bo-', label='Training RMSE')
#    plt.plot(epochs, val_rmse, 'r*-', label='Validation RMSE')
#    plt.title('Training and Validation RMSE')
#    plt.xlabel('Epochs')
#    plt.ylabel('RMSE')
#    plt.legend()
#    plt.grid(True)
#    plt.savefig('rmse_plot.png', dpi=300)  # Save plot with high quality
#    plt.show()
