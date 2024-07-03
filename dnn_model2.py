# Import libs
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys
import matplotlib.pyplot as plt
from sliding_window import create_feature_set
from sliding_window import plot_history

import sys

sytem_naming = sys.argv[1]

print(sytem_naming)
print(f'model_binned_data{sytem_naming}.keras')

#----------------------------------------------------
#----------------------------------------------------

# label 
data_target = np.load('extracted_pdb_all_atom_backbone_iga_antigen.npy').astype(float)[:,1:]
data_target = data_target/240

train_LAB = data_target[:600,:]
test_LAB = data_target[600:800,:]

print('training size and shape',train_LAB.shape)
print('testing size and shape',test_LAB.shape)
train_LAB_48 = create_feature_set(train_LAB)
test_LAB_48 = create_feature_set(test_LAB)



#feature
data_bb = np.load('data_bb.npy')
train_feat = data_bb[:600,:]
test_feat = data_bb[600:800,:]

train_feat_12 = create_feature_set(train_feat,12,3)
test_feat_12 = create_feature_set(test_feat,12,3)

del data_bb, data_target


def create_model_2(input_shape, num_outputs):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(2048, activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

     
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

     
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    #-----------------------------------------------------------------------------------

    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    output = tf.keras.layers.Dense(num_outputs, activation='linear')(x)
  
    
    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model

# Example usage:
input_shape = (12,)  # Example input shape, adjust based on your actual data
output_num = 48  # Example number of bins, adjust based on your actual data

# Create the model
model = create_model_2(input_shape, output_num)

print(model.summary())

# COMPILE
model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001), loss='mse',metrics=['mae',keras.metrics.RootMeanSquaredError()])

# training
history = model.fit(
    train_feat_12,train_LAB_48,
    epochs =300,
    shuffle = True,
    validation_data=(test_feat_12, test_LAB_48),
    verbose=1

)


model.save(f'model_dnn_2{sytem_naming}.keras')



# Assuming 'history' is your training history from model.fit()
plot_history(history)


