import os
import tensorflow as tf
#import matplotlib.pyplot as plt
import numpy as np
import json
import sys
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from model_activation_test import build_combined_model_masking_LSTM


window_size = int(sys.argv[1])


# Set your desired learning rate range:
max_lr = 1e-3   # Maximum learning rate
min_lr = 1e-7   # Minimum learning rate

# Calculate alpha for CosineDecayRestarts:
alpha = min_lr / max_lr  # This equals 1e-7 / 1e-3 = 1e-4

# Choose the number of iterations for one full cycle.
# For example, if you decide a full cycle lasts 4000 iterations:
first_decay_steps = 341178

# Create the cyclic learning rate schedule using CosineDecayRestarts.
lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=max_lr,   # Start at maximum LR
    first_decay_steps=first_decay_steps,
    t_mul=1.0,       # Keep the cycle length constant
    m_mul=1.0,       # Do not change the amplitude across cycles
    alpha=alpha      # Minimum LR as a fraction of max LR (1e-4 here)
)

# Use MirroredStrategy for distributed training across multiple GPUs
strategy = tf.distribute.MirroredStrategy()

print(f'Number of devices: {strategy.num_replicas_in_sync}')

# Function to load and combine batches
def load_and_combine_batches(x_file_list, y_file_list):
    for x_file, y_file in zip(x_file_list, y_file_list):
        x = np.load(x_file).astype(np.float32)
        y = np.load(y_file).astype(np.float32)
        for i in range(len(x)):
            yield x[i], y[i]

# Define batch size (you can adjust this depending on your memory limits)
batch_size = 32

# Create the training and validation datasets
train_feat = np.load('final_train_feat_perBead.npy')        #.reshape(-1,24,1) #.reshape(-1,24)
train_lab = np.load('final_train_lab_perBead.npy')
mask_train = np.load('final_train_input_masking_perBead.npy')


val_feat = np.load(f'final_test_feat_perBead.npy') #.reshape(-1,24)
val_labels = np.load(f'final_test_lab_perBead.npy')

mask_val = np.load('final_test_input_masking_perBead.npy')




class WarmUpLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, warmup_epochs, initial_lr, verbose=0):
        super(WarmUpLearningRateScheduler, self).__init__()
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr  # The target learning rate at the end of warmup
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            # Linearly increase the learning rate: start at 0 and go to initial_lr
            lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
            if self.verbose > 0:
                print(f"Epoch {epoch+1}: WarmUpLearningRateScheduler setting learning rate to {lr:.6f}")




# Open a strategy scope.
with strategy.scope():
    # Define the model
    optimizer = Adam(learning_rate=lr_schedule)
    
    input_shape = (24,1) 
    output_num = 15     # Adjust this based on your output shape
    alpha_value = 0.2    # LeakyReLU alpha value

#    model = build_combined_model_masking_LSTM(input_shape, output_num, mask_value=True)
    model = tf.keras.models.load_model('continued_model7_minmax_epoch_27.keras')

    # Compile the model with optimizer, loss, and metrics
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', tf.keras.metrics.RootMeanSquaredError()])

    # Print model summary
    print(model.summary())

#    warmup_callback = WarmUpLearningRateScheduler(warmup_epochs=5, initial_lr=0.001, verbose=1)
    batch_size = 224
    steps_per_epoch = np.ceil(train_feat.shape[0] / batch_size)
    # Define callbacks
#    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6, verbose=1)


    # Save the model including the optimizer
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='continued_3_model7_minmax_epoch_{epoch:02d}.keras',
        save_best_only=False,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        verbose=1
    )

    # ModelCheckpoint callback to save the best model
    checkpoint_callback2 = tf.keras.callbacks.ModelCheckpoint(
        filepath='best_continued_3_model7_minmax_1D_conv_v1.keras', 
        save_best_only=True, 
        monitor='val_loss', 
        mode='min',
        save_freq='epoch'
    )

    # Train the model using the Dataset API
    history = model.fit(
        [train_feat,mask_train],train_lab,  # Using the dataset from the generator
        epochs=100, batch_size=batch_size,      # Adjust number of epochs
        shuffle=True,
        validation_data=([val_feat,mask_val], val_labels),  # Replace with your actual validation data
        callbacks=[checkpoint_callback,checkpoint_callback2],
        verbose=1
    )



    # Save the training history to a JSON file
#    history_dict = history.history
 #   with open('training_history_w32_LR_1D_Conv.json', 'w') as history_file:
#        json.dump(history_dict, history_file)

