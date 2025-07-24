import tensorflow as tf
import numpy as np
import json
import sys
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from final_model_activation_test import build_1d_conv_autoencoder2
import random
from tensorflow.data.experimental import AutoShardPolicy

window_size = int(sys.argv[1])


# Use MirroredStrategy for distributed training across multiple GPUs
strategy = tf.distribute.MirroredStrategy()

print(f'Number of devices: {strategy.num_replicas_in_sync}')

# Function to load and combine batches efficiently
def load_and_combine_batches(x_file_list, y_file_list):
    for x_file, y_file in zip(x_file_list, y_file_list):
        x = np.load(x_file).astype(np.float32)  # Load as float32 to reduce memory usage
        y = np.load(y_file).astype(np.float32)
        for i in range(len(x)):
            yield x[i], y[i]

# Wrapper for TensorFlow's dataset API (to load files in parallel)
def tf_load_function(x_file_list, y_file_list):
    return tf.py_function(func=load_and_combine_batches, 
                          inp=[x_file_list, y_file_list], 
                          Tout=(tf.float32, tf.float32))

# Function to create dataset for training
def create_dataset(x_list, batch_size, buffer_size=5000000):
    def generator():
        for x_file_list, y_file_list in x_list:
            yield from load_and_combine_batches(x_file_list, y_file_list)

    dataset = tf.data.Dataset.from_generator(generator,
                                             output_signature=(
                                                 tf.TensorSpec(shape=((3 * window_size),), dtype=tf.float32),
                                                 tf.TensorSpec(shape=(12 * window_size,), dtype=tf.float32)
                                             ))

    # 🔹 Shuffling, batching, and prefetching
    dataset = dataset.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # 🔹 Set dataset options for performance
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = AutoShardPolicy.OFF  # Use OFF for single GPU, DATA for multi-GPU
    dataset = dataset.with_options(options)

    return dataset

# Function to create dataset for testing (no shuffling)
def create_test_dataset(x_list, batch_size):
    def generator():
        for x_file_list, y_file_list in x_list:
            yield from load_and_combine_batches(x_file_list, y_file_list)

    dataset = tf.data.Dataset.from_generator(generator,
                                             output_signature=(
                                                 tf.TensorSpec(shape=((3 * window_size),), dtype=tf.float32),
                                                 tf.TensorSpec(shape=(12 * window_size,), dtype=tf.float32)
                                             ))

    # 🔹 Only batching and prefetching (no shuffle)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # 🔹 Set dataset options for performance
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = AutoShardPolicy.OFF
    dataset = dataset.with_options(options)

    return dataset

# Example x_list (replace with your actual file lists)
with open("TRAINING_FEAT.txt", "r") as file:
    lines_feat = [line.strip() for line in file]


with open("TRAINING_LAB.txt", "r") as file:
    lines_lab = [line.strip() for line in file]

random.seed(42)
paired = list(zip(lines_feat,lines_lab))
random.shuffle(paired)
features_shuffled, targets_shuffled = zip(*paired)

x_list = [(features_shuffled, targets_shuffled)]


# Define batch size (you can adjust this depending on your memory limits)
batch_size = 512

# Create the training and validation datasets
train_dataset = create_dataset(x_list, batch_size)


# Example x_list (replace with your actual file lists)
with open("TESTING_FEAT.txt", "r") as file:
    lines_feat = [line.strip() for line in file]


with open("TESTING_LAB.txt", "r") as file:
    lines_lab = [line.strip() for line in file]

paired_test = list(zip(lines_feat,lines_lab))
random.shuffle(paired_test)
features_shuffled_test, targets_shuffled_test = zip(*paired)

x_list_test = [(features_shuffled_test, targets_shuffled_test)]

test_dataset = create_test_dataset(x_list_test, batch_size)

# Set your desired learning rate range:
max_lr = 0.002 
min_lr = 0.0003 


# Calculate alpha for CosineDecayRestarts:
alpha = min_lr / max_lr  # This equals 1e-7 / 1e-3 = 1e-4



# Choose the number of iterations for one full cycle.
# For example, if you decide a full cycle lasts 4000 iterations:
first_decay_steps = 1297822 
# Create the cyclic learning rate schedule using CosineDecayRestarts.
lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=max_lr,   # Start at maximum LR
    first_decay_steps=first_decay_steps,
    t_mul=1.0,       # Keep the cycle length constant
    m_mul=1.0,       # Do not change the amplitude across cycles
    alpha=alpha      # Minimum LR as a fraction of max LR (1e-4 here)
)



class ActivationSparsityLogger(tf.keras.callbacks.Callback):
    def __init__(self, model, sample_data, layers_to_watch=None):
        super().__init__()
        self.sample_x, _ = sample_data
        # by default, watch every layer that ends in 'relu'
        self.layers = [
            layer for layer in model.layers 
            if hasattr(layer, 'activation') and layer.activation == tf.keras.activations.relu
        ] if layers_to_watch is None else layers_to_watch

    def on_epoch_end(self, epoch, logs=None):
        # get a forward pass on sample data
        outputs = tf.keras.Model(self.model.input,
                                 [layer.output for layer in self.layers]).predict(self.sample_x, verbose=0)
        print(f"\nEpoch {epoch+1} activation sparsity:")
        for layer, out in zip(self.layers, outputs):
            zero_rate = np.mean(out == 0)
            print(f"  {layer.name:20s}: {zero_rate*100:5.1f}% zeros")



# Decide how many samples you want for sparsity checks
NUM_SAMPLES = 128

# Create empty lists to accumulate
x_accum = []
y_accum = []

# Iterate over your tf.data.Dataset until you’ve got NUM_SAMPLES
for x_batch, y_batch in test_dataset:
    x_accum.append(x_batch.numpy())
    y_accum.append(y_batch.numpy())
    total = sum(arr.shape[0] for arr in x_accum)
    if total >= NUM_SAMPLES:
        break

# Concatenate and trim to exactly NUM_SAMPLES
val_x = np.concatenate(x_accum, axis=0)[:NUM_SAMPLES]
val_y = np.concatenate(y_accum, axis=0)[:NUM_SAMPLES]





# Open a strategy scope.
with strategy.scope():
    # Define the model
    optimizer = Adam(learning_rate=0.001)   


    input_shape = 3*window_size  
    output_num = 12*window_size     # Adjust this based on your output shape
    alpha_value = 0.2    # LeakyReLU alpha value

    batch_size = 256

#    model = build_1d_conv_autoencoder2(input_shape, output_num)
    model = tf.keras.models.load_model('model9_check_epoch_41.keras')
    # Compile the model with optimizer, loss, and metrics
#    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', tf.keras.metrics.RootMeanSquaredError()])

    # Print model summary
    print(model.summary())

    # Define callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6, verbose=1)

    # Save the model including the optimizer
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='continued_model9_check_epoch_{epoch:02d}.keras',
        save_best_only=False,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        verbose=1
    )

    # ModelCheckpoint callback to save the best model
    checkpoint_callback2 = tf.keras.callbacks.ModelCheckpoint(
        filepath='best_continued_model9_check_MinMax_Conv3D.keras', 
        save_best_only=True, 
        monitor='val_loss', 
        mode='min',
        save_freq='epoch'
    )

    sparsity_cb = ActivationSparsityLogger(model, sample_data=(val_x, val_y))
    # Train the model using the Dataset API
    history = model.fit(
        train_dataset,  # Using the dataset from the generator
        epochs=100, batch_size=batch_size,       # Adjust number of epochs
        #shuffle=True,
        validation_data=(test_dataset),  # Replace with your actual validation data
        callbacks=[checkpoint_callback,checkpoint_callback2,sparsity_cb],
        verbose=1
    )

    # Save the training history to a JSON file
    history_dict = history.history
    with open('training_history_w32_LR_1D_Conv.json', 'w') as history_file:
        json.dump(history_dict, history_file)

