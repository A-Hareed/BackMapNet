import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical



data_aa = np.load('../change_aa.npy')


# Data Loading
data_bb = np.load('../data_bb.npy')
torsion_angle = np.load('../torsion_angles.npy')[1:,:]
com_Evector = np.load('../com_e_vector.npy')



train_feature_bb = data_bb[:100,:].reshape(-1,3)
tor_train = torsion_angle[:100,:].reshape(-1,1)
com_train  = com_Evector[:100,:].reshape(-1,1)

train_feature_bb = np.concatenate((train_feature_bb,tor_train,com_train),axis=1)


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



train_label = get_labels(data_aa[:100,:])
print('label data shape:  ',train_label.shape)


# Function to bin data with specified bin width
def bin_data(values, bin_width=0.2, range_min=-3, range_max=3):
    num_bins = int((range_max - range_min) / bin_width)
    bins = np.linspace(range_min, range_max, num_bins + 1)
    bin_indices = np.digitize(values, bins) - 1  # Subtract 1 to convert to 0-indexed
    y_binned = np.clip(bin_indices, 0, num_bins - 1)  # Ensure indices are within valid range
    y_one_hot = to_categorical(y_binned, num_bins)
    return y_one_hot,num_bins


y_binned, numbin = bin_data(train_label,bin_width=0.3)
y_binned.shape,numbin


# DNN Model:

def create_logit_model(input_shape, num_outputs):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # output = tf.keras.layers.Dense(num_outputs, activation="softmax", name="output")(x)
    # Create separate output layers for each target
    outputs = []
    for i in range(12):
        output = tf.keras.layers.Dense(num_outputs, activation="softmax", name=f"output_{i}")(x)  # No activation for logits
        outputs.append(output)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model



# Example usage:
input_shape = (5,)  # Example input shape, adjust based on your actual data
output_num = 20  # Example number of bins, adjust based on your actual data

# Create the model
model = create_logit_model(input_shape, output_num)


def top_5_accuracy(y_true, y_pred):
  return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', top_5_accuracy])


# Example training
history = model.fit(train_feature_bb,  # Input data
                    [y_binned[:, i,:] for i in range(12)],  # List of target arrays for each output
                    epochs=100,
                    batch_size=30) 

model.save('model_binned_data.keras')

