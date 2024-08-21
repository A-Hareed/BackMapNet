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



def create_model_LeakyRELU(input_shape, num_outputs,activation_layer=tf.keras.layers.LeakyReLU):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(2048,kernel_initializer='he_normal')(inputs)
    x = activation_layer(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(1024,kernel_initializer='he_normal')(x)
    x = activation_layer(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Dense(512,kernel_initializer='he_normal' )(x)
    x = activation_layer(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Dense(256,kernel_initializer='he_normal')(x)
    x = activation_layer(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Dense(256,kernel_initializer='he_normal')(x)
    x = activation_layer(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Dense(128,kernel_initializer='he_normal')(x)
    x = activation_layer(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Dense(128,kernel_initializer='he_normal')(x)
    x = activation_layer(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Dense(64,kernel_initializer='he_normal')(x)
    x = activation_layer(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    output = tf.keras.layers.Dense(num_outputs, activation='linear')(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model


input_shape = (12,)  # Example input shape, adjust based on your actual data
output_num = 48  # Example number of bins, adjust based on your actual data

# Create the model
model = create_model_LeakyRELU(input_shape, output_num)
model.save(f'model_dnn_2{sytem_naming}.keras')

print(model.summary())
exit()
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



# # Assuming 'history' is your training history from model.fit()
# plot_history(history)

#     # Define the model creation function
#     def create_model_2(input_shape, num_outputs):
#         inputs = tf.keras.Input(shape=input_shape)
#         regularizer = tf.keras.regularizers.l2(0.01)
#         x = tf.keras.layers.Dense(2048, kernel_regularizer=regularizer)(inputs)
#         x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
#         x = tf.keras.layers.BatchNormalization()(x)

#         x = residual_block(x, 2048)
#         x = residual_block(x, 2048)  # Ensure dimensions match within residual blocks
#         x = residual_block(x, 1024)  # Ensure dimensions match within residual blocks
#         x = tf.keras.layers.Dropout(0.2)(x)
        
#         x = residual_block(x, 512)  # Ensure dimensions match within residual blocks
#         x = tf.keras.layers.Dropout(0.2)(x)
        
#         x = residual_block(x, 256)  # Ensure dimensions match within residual blocks
#         x = tf.keras.layers.Dropout(0.2)(x)

#         x = residual_block(x, 256)  # Ensure dimensions match within residual blocks
#         x = tf.keras.layers.Dropout(0.2)(x)

#         x = residual_block(x, 128)  # Ensure dimensions match within residual blocks
#         x = tf.keras.layers.Dropout(0.2)(x)

#         x = residual_block(x, 128)  # Ensure dimensions match within residual blocks
#         x = tf.keras.layers.Dropout(0.2)(x)

#         x = residual_block(x, 64)  # Ensure dimensions match within residual blocks
#         x = tf.keras.layers.Dropout(0.2)(x)

#         output = tf.keras.layers.Dense(num_outputs, activation='linear')(x)
        
#         model = tf.keras.Model(inputs=inputs, outputs=output)
#         return model