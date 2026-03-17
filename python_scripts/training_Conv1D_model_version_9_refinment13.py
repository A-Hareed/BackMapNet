import tensorflow as tf
import numpy as np
import json
import sys
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from final_model_activation_test import build_1d_conv_autoencoder_multi_input
from torsion_loss4 import torsion_mse_loss_fast
import random
from tensorflow.data.experimental import AutoShardPolicy
from tensorflow.keras.losses import MeanSquaredError

window_size = int(sys.argv[1])


# Use MirroredStrategy for distributed training across multiple GPUs
strategy = tf.distribute.MirroredStrategy()

print(f'Number of devices: {strategy.num_replicas_in_sync}')

# Function to load and combine batches efficiently
def load_and_combine_batches(x_file_list, y_file_list, r_file_list):
    for x_file, y_file, r_fp in zip(x_file_list, y_file_list, r_file_list):
        x = np.load(x_file).astype(np.float32)  # Load as float32 to reduce memory usage
        y = np.load(y_file).astype(np.float32)
        r_all = np.load(r_fp).astype(np.float32)   # [N, 1, 3]
        for i in range(len(x)):
            yield (x[i],r_all[i]), y[i]

# Wrapper for TensorFlow's dataset API (to load files in parallel)
def tf_load_function(x_file_list, y_file_list):
    return tf.py_function(func=load_and_combine_batches, 
                          inp=[x_file_list, y_file_list], 
                          Tout=(tf.float32, tf.float32))

# Function to create dataset for training
def create_dataset(x_list, batch_size, buffer_size=5000000):
    def generator():
        for x_file_list, y_file_list, r_file_list in x_list:
            yield from load_and_combine_batches(x_file_list, y_file_list,r_file_list)

    dataset = tf.data.Dataset.from_generator(generator,
                                             output_signature=(
                                                 (tf.TensorSpec(shape=((3 * window_size),), dtype=tf.float32),
                                                 tf.TensorSpec(shape=(1,3), dtype=tf.float32), ),
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
        for x_file_list, y_file_list, r_file_list in x_list:
            yield from load_and_combine_batches(x_file_list, y_file_list, r_file_list)

    dataset = tf.data.Dataset.from_generator(generator,
                                             output_signature=(
                                                 (tf.TensorSpec(shape=((3 * window_size),), dtype=tf.float32),
                                                 tf.TensorSpec(shape=(1,3), dtype=tf.float32), ),
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

with open("RANGE_TRAIN.txt", "r") as file:
    lines_range = [line.strip() for line in file]

# Make sure all are same length
assert len(lines_feat) == len(lines_lab) == len(lines_range)

random.seed(42)
paired = list(zip(lines_feat,lines_lab,lines_range))
random.shuffle(paired)
features_shuffled, targets_shuffled, range_shuffled = zip(*paired)

x_list = [(features_shuffled, targets_shuffled, range_shuffled)]


# Define batch size (you can adjust this depending on your memory limits)
batch_size = 512

# Create the training and validation datasets
train_dataset = create_dataset(x_list, batch_size)


# Example x_list (replace with your actual file lists)
with open("TESTING_FEAT.txt", "r") as file:
    lines_feat = [line.strip() for line in file]


with open("TESTING_LAB.txt", "r") as file:
    lines_lab = [line.strip() for line in file]


with open("RANGE_TEST.txt", "r") as file:
    lines_range = [line.strip() for line in file]

# Make sure all are same length
assert len(lines_feat) == len(lines_lab) == len(lines_range)

paired_test = list(zip(lines_feat,lines_lab,lines_range))
random.shuffle(paired_test)
features_shuffled_test, targets_shuffled_test, range_shuffled_test = zip(*paired)

x_list_test = [(features_shuffled_test, targets_shuffled_test, range_shuffled_test)]

test_dataset = create_test_dataset(x_list_test, batch_size)

# Set your desired learning rate range:
max_lr = 0.002 
min_lr = 0.0003 


# Calculate alpha for CosineDecayRestarts:
alpha = min_lr / max_lr  # This equals 1e-7 / 1e-3 = 1e-4



# Decide how many samples you want for sparsity checks
NUM_SAMPLES = 128



# --- Hyperparameters & Annealed Weight ---
a_initial = 1.0  # weight for torsion term
######### mse_loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)


def combined_torsion_loss(y_true, y_pred):
    """
    y_true: Tensor of shape [B,384]      — true normalized coords
    y_pred: Tensor of shape [B,387]      — [pred_coords (384), range (3)]
    """
    # 1) Split coords vs. range
    coords_pred = y_pred[:, :384]            # [B,384]
    ranges_raw  = y_pred[:, 384:]            # [B,3]

    # 2) Restore shape [B,1,3]
    ranges = tf.expand_dims(ranges_raw, axis=1)  # [B,1,3]

    # 3) Compute your torsion‐MSE
    #    ⬇️ this is where torsion_mse_loss_from_flat is invoked
    loss = torsion_mse_loss_fast(
        norm_coords      = coords_pred,
        true_norm_coords = y_true,
        ranges           = ranges
    )

    return loss


def normalized_coord_mse(y_true, y_pred):
    """
    y_true: [B,384]       — true normalized coords
    y_pred: [B,387]       — [pred_norm_coords (384), ranges (3)]
    returns: scalar MSE of coords only (in normalized units)
    """
    coords_pred = y_pred[:, :384]       # drop the last 3 cols of y_pred
    coords_true = y_true[:, :384]       # true coords already shape [B,384]
  
    return tf.reduce_mean(tf.square(coords_pred - coords_true))




# Load once at the top
neglog = np.load('RamachandranEval_prors.npy').T

neglog_tf  = tf.constant(neglog, dtype=tf.float32)            # [bins, bins]
PRIOR_GRID = neglog_tf
bins = neglog.shape[0]
bin_width = 360.0 / float(bins)
BIN_WIDTH_TF = tf.constant(bin_width, dtype=tf.float32)
 
PI = tf.constant(np.pi, dtype=tf.float32)
ONE_F = tf.constant(1.0, dtype=tf.float32)
ZERO_F = tf.constant(0.0, dtype=tf.float32)

# change percentile as you like (99.5 is strict; 99.0 less strict)
percentile = 14
T_value = float(np.percentile(neglog.ravel(), percentile))
T_TF = tf.constant(T_value, dtype=tf.float32)

lambda_rama = tf.constant(1e-4, dtype=tf.float32)  # <-- tune this

def _tf_sample_bilinear(grid, phi_deg, psi_deg, bins, bin_width_tf):
    """
    grid: [bins, bins] float32 (neglog)
    phi_deg, psi_deg: [B, N] degrees in [0,360)
    returns: [B, N] sampled neglog values (float32)
    """
    # enforce float32
    grid = tf.cast(grid, tf.float32)
    phi_deg = tf.cast(phi_deg, tf.float32)
    psi_deg = tf.cast(psi_deg, tf.float32)

    # continuous index coordinates
    x = phi_deg / bin_width_tf
    y = psi_deg / bin_width_tf

    x0 = tf.floor(x)
    y0 = tf.floor(y)
    x_frac = x - x0
    y_frac = y - y0

    # integer indices wrapped modulo bins
    x0_i = tf.cast(tf.math.floormod(tf.cast(x0, tf.int32), bins), tf.int32)
    y0_i = tf.cast(tf.math.floormod(tf.cast(y0, tf.int32), bins), tf.int32)
    x1_i = tf.cast(tf.math.floormod(x0_i + 1, bins), tf.int32)
    y1_i = tf.cast(tf.math.floormod(y0_i + 1, bins), tf.int32)

    def gather_vals(ix, iy):
        ix_flat = tf.reshape(ix, [-1])
        iy_flat = tf.reshape(iy, [-1])
        idx = tf.stack([ix_flat, iy_flat], axis=1)   # [B*N, 2]
        vals = tf.gather_nd(grid, idx)               # [B*N]
        return tf.reshape(vals, tf.shape(ix))        # [B, N]

    Q11 = gather_vals(x0_i, y0_i)
    Q21 = gather_vals(x1_i, y0_i)
    Q12 = gather_vals(x0_i, y1_i)
    Q22 = gather_vals(x1_i, y1_i)

    wx = tf.cast(x_frac, tf.float32)
    wy = tf.cast(y_frac, tf.float32)

    top    = Q11 * (ONE_F - wx) + Q21 * wx
    bottom = Q12 * (ONE_F - wx) + Q22 * wx
    val = top * (ONE_F - wy) + bottom * wy

    return val  # [B, N], float32

# final penalty inside TF function
@tf.function
def rama_penalty(y_true, y_pred, use_squared_hinge=True, return_fraction=False):
    """
    Returns per-example hinge penalty [B].
    If return_fraction=True, returns (penalty, fraction_forbidden) -- but Keras loss expects [B], so keep False for training.
    """
    # Unpack predicted coords -> get torsions using your function
    coords_pred = y_pred[:, :384]
    ranges_raw  = y_pred[:, 384:]
    ranges = tf.expand_dims(ranges_raw, axis=1)

    # Your existing function that computes phi/psi in radians
    phi_psi_rad = torsion_mse_loss_fast(norm_coords=coords_pred, ranges=ranges)
    phi_rad = phi_psi_rad[:, :31]   # [B, N]
    psi_rad = phi_psi_rad[:, 31:]   # [B, N]

    # valid mask
    valid = tf.math.is_finite(phi_rad) & tf.math.is_finite(psi_rad)  # [B, N]

    # map radians -> degrees in [0,360)
    phi_deg = tf.math.floormod((phi_rad * 180.0 / PI) + 180.0, 360.0)
    psi_deg = tf.math.floormod((psi_rad * 180.0 / PI) + 180.0, 360.0)

    # sample the prior (differentiable)
    sampled_neglog = _tf_sample_bilinear(PRIOR_GRID, phi_deg, psi_deg, bins, BIN_WIDTH_TF)  # [B, N]

    # mask invalids: set them to 0 so they don't contribute in mean
    sampled_neglog = tf.where(valid, sampled_neglog, tf.zeros_like(sampled_neglog))

    # compute hinge values: max(0, E - T)
    hinge_vals = tf.nn.relu(sampled_neglog - T_TF)  # [B, N]

    if use_squared_hinge:
        hinge_vals = tf.square(hinge_vals)  # optional: square the hinge for stronger effect

    # per-example average across torsions (ignore invalids via counts)
    counts = tf.reduce_sum(tf.cast(valid, tf.float32), axis=1)  # [B]
    sums = tf.reduce_sum(hinge_vals, axis=1)                    # [B]
    rama_per_example = sums / (counts + 1e-6)                   # [B]

    # optional diagnostics: fraction of torsions above threshold (for monitoring)
    is_forbidden = tf.cast(sampled_neglog > T_TF, tf.float32)
    fraction_forbidden = tf.reduce_sum(is_forbidden * tf.cast(valid, tf.float32), axis=1) / (counts + 1e-6)  # [B]

    if return_fraction:
        return rama_per_example, fraction_forbidden

    return rama_per_example






#log_sigma = tf.Variable(0.0, trainable=True)
#log_sigma_coor = tf.Variable(1.0, trainable=True)

def combined_coord_and_torsion_loss(y_true, y_pred):
    """
    y_true: [B,384]      — true normalized coords
    y_pred: [B,387]      — [pred_norm_coords (384), ranges (3)]

    α: weight for coord MSE term
    β: weight for torsion MSE term

    returns: scalar loss = α * normalized_coord_mse + β * torsion_loss
    """
    # 1) compute current β
#    beta = 0.5
    #scale_mse = 10000
    scale_mse = 0.0001000
    #scale_mse = 10000
     
    # 2) coordinate MSE (normalized space)
    mse = normalized_coord_mse(y_true, y_pred)

    # 3) torsion-MSE (uses only normalized coords + ranges)
    rama_loss = rama_penalty(y_true,y_pred,use_squared_hinge=True)

#    sigma_coor = tf.exp(log_sigma_coor)
#    sigma = tf.exp(log_sigma)
    # 4) weighted sum
    return  mse + scale_mse*rama_loss
#    return  (0.5/(sigma_coor**2) *mse) + (0.5/(sigma**2) * rama_loss) + log_sigma + log_sigma_coor

# —————————————————————————————————————————————
# 2) Custom metric for coordinate RMSE
#—————————————————————————————————————————————
class CoordRMSE(tf.keras.metrics.RootMeanSquaredError):
    def update_state(self, y_true, y_pred, sample_weight=None):
        coords_pred = y_pred[:, :384]
        return super().update_state(y_true[:, :384], coords_pred, sample_weight)




def phi_metric(y_true, y_pred):
    
    phi_loss = combined_torsion_loss(y_true, y_pred)
    return tf.reduce_mean(phi_loss[:, :31])

def psi_metric(y_true, y_pred):
    
    psi_loss = combined_torsion_loss(y_true, y_pred)
    return tf.reduce_mean(psi_loss[:, 31:])



class FractionForbidden(tf.keras.metrics.Metric):
    def __init__(self,
                 prior_grid,        # tf.Tensor [bins, bins] float32 (neglog)
                 bin_width,         # scalar float32 (degrees)
                 threshold,         # scalar float32
                 bins,
                 name="fraction_forbidden",
                 dtype=tf.float32,
                 **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.prior_grid = tf.cast(prior_grid, tf.float32)
        self.bin_width = tf.cast(bin_width, tf.float32)
        self.threshold = tf.cast(threshold, tf.float32)
        self.bins = int(bins)

        # accumulators for average over batches
        self.total = self.add_weight(name="total", initializer="zeros", dtype=dtype)
        self.count = self.add_weight(name="count", initializer="zeros", dtype=dtype)

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)

    def _sample_bilinear(self, phi_deg, psi_deg):
        """Same bilinear sampler as used in rama_penalty. Inputs: [B, N] degrees."""
        # ensure float32
        grid = self.prior_grid
        phi_deg = tf.cast(phi_deg, tf.float32)
        psi_deg = tf.cast(psi_deg, tf.float32)
        bin_width = self.bin_width
        bins = self.bins

        x = phi_deg / bin_width
        y = psi_deg / bin_width

        x0 = tf.floor(x)
        y0 = tf.floor(y)
        x_frac = x - x0
        y_frac = y - y0

        x0_i = tf.cast(tf.math.floormod(tf.cast(x0, tf.int32), bins), tf.int32)
        y0_i = tf.cast(tf.math.floormod(tf.cast(y0, tf.int32), bins), tf.int32)
        x1_i = tf.cast(tf.math.floormod(x0_i + 1, bins), tf.int32)
        y1_i = tf.cast(tf.math.floormod(y0_i + 1, bins), tf.int32)

        def gather_vals(ix, iy):
            ix_flat = tf.reshape(ix, [-1])
            iy_flat = tf.reshape(iy, [-1])
            idx = tf.stack([ix_flat, iy_flat], axis=1)  # [B*N,2]
            vals = tf.gather_nd(grid, idx)              # [B*N]
            return tf.reshape(vals, tf.shape(ix))       # [B,N]

        Q11 = gather_vals(x0_i, y0_i)
        Q21 = gather_vals(x1_i, y0_i)
        Q12 = gather_vals(x0_i, y1_i)
        Q22 = gather_vals(x1_i, y1_i)

        wx = tf.cast(x_frac, tf.float32)
        wy = tf.cast(y_frac, tf.float32)

        top = Q11 * (1.0 - wx) + Q21 * wx
        bottom = Q12 * (1.0 - wx) + Q22 * wx
        val = top * (1.0 - wy) + bottom * wy
        return val  # [B, N]

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        y_true, y_pred: as passed by model.fit when metric is registered.
        Use same logic as rama_penalty to extract phi/psi (radians) from predictions.
        """
        # Unpack predicted coords / ranges as in your pipeline:
        coords_pred = y_pred[:, :384]
        ranges_raw  = y_pred[:, 384:]
        ranges = tf.expand_dims(ranges_raw, axis=1)

        # Call your torsion-extraction function (must be TF / graph-friendly)
        phi_psi_rad = torsion_mse_loss_fast(norm_coords=coords_pred, ranges=ranges)
        phi_rad = phi_psi_rad[:, :31]
        psi_rad = phi_psi_rad[:, 31:]

        # validity mask
        valid = tf.math.is_finite(phi_rad) & tf.math.is_finite(psi_rad)  # [B,N]

        # degrees in [0,360)
        PI = tf.constant(np.pi, dtype=tf.float32)
        phi_deg = tf.math.floormod((phi_rad * 180.0 / PI) + 180.0, 360.0)
        psi_deg = tf.math.floormod((psi_rad * 180.0 / PI) + 180.0, 360.0)

        # sampled neglog at each torsion
        sampled_neglog = self._sample_bilinear(phi_deg, psi_deg)  # [B,N]

        # mask invalids
        sampled_neglog = tf.where(valid, sampled_neglog, tf.zeros_like(sampled_neglog))

        # compute per-example fraction forbidden (sampled_neglog > threshold)
        is_forbidden = tf.cast(sampled_neglog > self.threshold, tf.float32)
        forbidden_counts = tf.reduce_sum(is_forbidden * tf.cast(valid, tf.float32), axis=1)  # [B]
        counts = tf.reduce_sum(tf.cast(valid, tf.float32), axis=1)  # [B]
        fraction_per_example = forbidden_counts / (counts + 1e-6)  # [B]

        # batch mean fraction (apply sample_weight if provided)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            # expand sample_weight if scalar
            if tf.rank(sample_weight) == 0:
                sample_weight = tf.fill(tf.shape(fraction_per_example), sample_weight)
            weighted_sum = tf.reduce_sum(fraction_per_example * sample_weight)
            denom = tf.reduce_sum(sample_weight)
            batch_mean = weighted_sum / (denom + 1e-6)
        else:
            batch_mean = tf.reduce_mean(fraction_per_example)

        # accumulate
        self.total.assign_add(batch_mean)
        self.count.assign_add(1.0)

    def result(self):
        return self.total / (self.count + 1e-12)


# Open a strategy scope.
with strategy.scope():
    # Define the model
    optimizer = Adam(learning_rate=1e-4)   
    frac_metric = FractionForbidden(prior_grid=PRIOR_GRID,
                                bin_width=BIN_WIDTH_TF,
                                threshold=T_TF,
                                bins=bins)


    input_shape = 3*window_size  
    output_num = 12*window_size     # Adjust this based on your output shape
    alpha_value = 0.2    # LeakyReLU alpha value

    batch_size = 256

#    model = build_1d_conv_autoencoder2(input_shape, output_num)
#    model = tf.keras.models.load_model('refined17_model9_check_epoch_11.keras', custom_objects={'combined_coord_and_torsion_loss':combined_coord_and_torsion_loss,
#                                  'CoordRMSE':CoordRMSE, 'rama_penalty':rama_penalty ,'normalized_coord_mse':normalized_coord_mse,'frac_metric':frac_metric},compile=False)
#    model = tf.keras.models.load_model('best_model9_check_MinMax_Conv3D.keras')


    model = tf.keras.models.load_model('refined24_model9_check_epoch_11.keras', custom_objects={'combined_coord_and_torsion_loss':combined_coord_and_torsion_loss,
                                  'CoordRMSE':CoordRMSE, 'rama_penalty':rama_penalty ,'normalized_coord_mse':normalized_coord_mse,'frac_metric':frac_metric},compile=False)



    weights = model.get_weights()

    model = build_1d_conv_autoencoder_multi_input(96)

    model.set_weights(weights)

    for layer in model.layers:
        layer.trainable = False


    trainable_from = ["conv1d_transpose",  "batch_normalization_9", "leaky_re_lu_11", "batch_normalization_10", 
    "leaky_re_lu_12", "conv1d_10", 'batch_normalization_11','leaky_re_lu_13', 'conv1d_11', 
    'batch_normalization_12', 'leaky_re_lu_14', 'conv1d_12', 'spatial_dropout1d_2', 'add_2', 
    'conv1d_transpose_1','batch_normalization_13', 'leaky_re_lu_15', 'batch_normalization_14',
    'leaky_re_lu_16', 'conv1d_13', 'batch_normalization_15', 'leaky_re_lu_17', 'conv1d_14', 
    'batch_normalization_16', 'leaky_re_lu_18', 'conv1d_15', 'spatial_dropout1d_3', 'add_3', 
    'conv1d_16', "flatten_1","dense_2","leaky_re_lu_19","scaling_factors_input","predicted_coords", 
    "reshape_2", "coords_and_range"]


    make_trainable = False

    for layer in model.layers:
        if layer.name in trainable_from:
            make_trainable = True
        if make_trainable:
            layer.trainable = True




    # Compile the model with optimizer, loss, and metrics
    model.compile(optimizer=optimizer, loss=combined_coord_and_torsion_loss, metrics=[CoordRMSE(name='coord_rmse'),rama_penalty,normalized_coord_mse,frac_metric])

    # Print model summary
    print(model.summary())

    # Define callbacks
#    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6, verbose=1)

    # Save the model including the optimizer
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='refined25_model9_check_epoch_{epoch:02d}.keras',
        save_best_only=False,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        verbose=1
    )

    # ModelCheckpoint callback to save the best model
    checkpoint_callback2 = tf.keras.callbacks.ModelCheckpoint(
        filepath='best_refined25_model9_check_MinMax_Conv3D.keras', 
        save_best_only=True, 
        monitor='val_loss', 
        mode='min',
        save_freq='epoch'
    )

#    sparsity_cb = ActivationSparsityLogger(model, sample_data=(val_x, val_y))
    # Train the model using the Dataset API

history = model.fit(
        train_dataset,  # Using the dataset from the generator
        epochs=100, batch_size=batch_size,       # Adjust number of epochs
        #shuffle=True,
        validation_data=(test_dataset),  # Replace with your actual validation data
        callbacks=[checkpoint_callback,checkpoint_callback2],
        verbose=1
    )



    # Save the training history to a JSON file
history_dict = history.history
with open('training_history_w32_LR_1D_Conv.json', 'w') as history_file:
    json.dump(history_dict, history_file)

