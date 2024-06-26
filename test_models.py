# Import libs
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import sys

#----------------------------------------------------
#--------Load Data-----------------------------------


data_aa = np.load('../change_aa.npy')


# Data Loading
data_bb = np.load('../data_bb.npy')
torsion_angle = np.load('../torsion_angles.npy')[1:,:]
com_Evector = np.load('../com_e_vector.npy')



test_feature_bb = data_bb[100:150,:].reshape(-1,3)
tor_test = torsion_angle[100:150,:].reshape(-1,1)
com_test  = com_Evector[100:150,:].reshape(-1,1)

test_feature_bb = np.concatenate((test_feature_bb,tor_test,com_test),axis=1)

def get_labels(arr):
    start = -12
    for i in range(0,arr.shape[1],12):
        start = i
        end  = i+12

        if i ==0:
            label = arr[:,start:end]
        else:
            label = np.concatenate((label,arr[:,start:end]),axis=0)
    
    return label


test_label = get_labels(data_aa[100:150,:])


# Function to bin data with specified bin width
def bin_data(values, bin_width=0.2, range_min=-3, range_max=3):
    num_bins = int((range_max - range_min) / bin_width)
    bins = np.linspace(range_min, range_max, num_bins + 1)
    bin_indices = np.digitize(values, bins) - 1  # Subtract 1 to convert to 0-indexed
    y_binned = np.clip(bin_indices, 0, num_bins - 1)  # Ensure indices are within valid range
    y_one_hot = to_categorical(y_binned, num_bins)
    return y_one_hot,num_bins

y_binned, numbin = bin_data(test_label,bin_width=0.3)
y_binned.shape,numbin

# top_L metrics 
@keras.saving.register_keras_serializable()
def top_5_accuracy(y_true, y_pred):
  return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)

model = tf.keras.models.load_model('model_binned_data.keras')


print(model.summary())

print('label data shape:  ',test_label.shape)

print('feature data shape: ',test_feature_bb.shape)


y_pred = model.predict(test_feature_bb)

np.save('predicted_model1_raw_probabilities.npy',y_pred)
