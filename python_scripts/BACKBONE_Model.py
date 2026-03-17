import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Conv1DTranspose, Dense, Flatten, Reshape, BatchNormalization, LeakyReLU, Add, SpatialDropout1D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import numpy as np


# Your existing residual_block_1d2 remains the same
def residual_block_1d2(x, filters, dropout_rate=0.0):
    shortcut = x
    y = BatchNormalization()(x)
    y = LeakyReLU(alpha=0.1)(y)
    y = Conv1D(filters, kernel_size=3, padding='same')(y)
    y = BatchNormalization()(y)
    y = LeakyReLU(alpha=0.1)(y)
    y = Conv1D(filters, kernel_size=3, padding='same')(y)
    y = BatchNormalization()(y)
    y = LeakyReLU(alpha=0.1)(y)
    y = Conv1D(filters, kernel_size=3, padding='same')(y)
    if dropout_rate > 0:
        y = SpatialDropout1D(rate=dropout_rate)(y)
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, kernel_size=1, padding='same')(shortcut)
    return Add()([shortcut, y])

def build_1d_conv_autoencoder2(
    input_length, output_features,
    latent_dim=128, dropout_rate=0.1
):
    """
    Builds a 1D conv autoencoder with Conv1DTranspose instead of upsampling,
    explicit conv layers between residual blocks, bottleneck, BatchNorm,
    SpatialDropout, and LeakyReLU.
    """
    inputs = Input(shape=(input_length,), name='coords_input')

    # Reshape and initial feature expansion
    x = Reshape((32, 3))(inputs)
    x = Conv1D(64, kernel_size=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Encoder stage 1
    x = residual_block_1d2(x, 64, dropout_rate)
    x = MaxPooling1D(pool_size=2)(x)  # 32 -> 16
    x = Conv1D(64, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Encoder stage 2
    x = residual_block_1d2(x, 128, dropout_rate)
    x = MaxPooling1D(pool_size=2)(x)  # 16 -> 8
    x = Conv1D(128, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Bottleneck
    shape_before = x.shape[1:]  # (8, channels)
    x = Flatten()(x)
    x = Dense(latent_dim)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dense(shape_before[0] * shape_before[1])(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Reshape(shape_before)(x)

    # Decoder stage 1 (transpose conv)
    x = Conv1DTranspose(
        filters=128, kernel_size=4, strides=2, padding='same'
    )(x)  # 8 -> 16
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = residual_block_1d2(x, 128, dropout_rate)

    # Decoder stage 2 (transpose conv)
    x = Conv1DTranspose(
        filters=64, kernel_size=4, strides=2, padding='same'
    )(x)  # 16 -> 32
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = residual_block_1d2(x, 64, dropout_rate)

    # Reconstruction layer
    x = Conv1D(3, kernel_size=1, padding='same')(x)  # linear

    # Dense head
    x = Flatten()(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.1)(x)
    outputs = Dense(output_features, activation='linear', name='predicted_coords')(x)

    return Model(inputs, outputs)
