from tensorflow.keras.layers import (
    Input, Reshape, Conv1D, Conv1DTranspose, MaxPooling1D,
    Add, Flatten, Dense, BatchNormalization,
    SpatialDropout1D, LeakyReLU
)
from tensorflow.keras.models import Model
# Define a residual block using 1D convolutions
def residual_block_1d(x, filters):
    """
    Residual block using 1D convolutions focusing on the sequence axis.
    """
    # Store input for skip connection
    shortcut = x

    # First convolution
    y = Conv1D(filters, kernel_size=3, padding='same', activation='relu')(x)
    # Second convolution
    y = Conv1D(filters, kernel_size=3, padding='same', activation='relu')(y)
    # Third convolution
    y = Conv1D(filters, kernel_size=3, padding='same', activation='relu')(y)

    # Adjust the skip path if channel dimensions differ using a 1x1 convolution
    # Conv1D(filters, kernel_size=1) is the 1D equivalent of Conv3D(filters, kernel_size=(1,1,1))
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, kernel_size=1, padding='same')(shortcut)

    # Add the shortcut to the main path
    return Add()([shortcut, y])

def residual_block_1d2(x, filters, dropout_rate=0.0):
    """
    Pre-activation residual block using 1D convolutions focusing on the sequence axis.
    Includes BatchNorm, LeakyReLU, optional SpatialDropout1D.
    """
    shortcut = x

    # First conv
    y = BatchNormalization()(x)
    y = LeakyReLU(alpha=0.1)(y)
    y = Conv1D(filters, kernel_size=3, padding='same')(y)

    # Second conv
    y = BatchNormalization()(y)
    y = LeakyReLU(alpha=0.1)(y)
    y = Conv1D(filters, kernel_size=3, padding='same')(y)

    # Third conv
    y = BatchNormalization()(y)
    y = LeakyReLU(alpha=0.1)(y)
    y = Conv1D(filters, kernel_size=3, padding='same')(y)

    # Dropout
    if dropout_rate > 0:
        y = SpatialDropout1D(rate=dropout_rate)(y)

    # Skip projection if needed
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
    inputs = Input(shape=(input_length,))

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
    outputs = Dense(output_features, activation='linear')(x)

    return Model(inputs, outputs)


def build_1d_conv_autoencoder(input_length, output_features):
    """
    Constructs a 1D convolutional autoencoder using the
    (batch, sequence_length, channels=3) layout for input coordinates.
    - input_length: total flattened features (e.g., 32 beads × 3 coords = 96)
    - output_features: dimension of the final prediction (e.g., 128 atoms × 3 coords = 384)
    """
    # Input is the flattened vector (batch, 96)
    inputs = Input(shape=(input_length,))

    # Reshape to (sequence_length=32, channels=3) for Conv1D
    # input_length (96) = 32 * 3
    x = Reshape((32, 3))(inputs)

    # Optional 1x1 conv (Conv1D kernel_size=1) to mix coordinate channels into a higher-dimensional space
    # This is the 1D equivalent of your initial Conv3D(64, kernel_size=(1,1,1), ...)
    x = Conv1D(64, kernel_size=1, padding='same', activation='relu')(x)

    # Encoder: extract hierarchical sequence features
    # Each residual_block_1d uses kernel_size=3, padding='same'
    x = residual_block_1d(x, 64)
    # MaxPooling1D pools along the sequence dimension (axis 1)
    x = MaxPooling1D(pool_size=2)(x)  # Sequence length becomes 32 / 2 = 16

    x = residual_block_1d(x, 128)
    x = MaxPooling1D(pool_size=2)(x) # Sequence length becomes 16 / 2 = 8

    # Decoder: mirror the encoder to reconstruct the original sequence length
    # UpSampling1D upsamples along the sequence dimension (axis 1)
    x = UpSampling1D(size=2)(x)      # Sequence length becomes 8 * 2 = 16
    x = residual_block_1d(x, 128)

    x = UpSampling1D(size=2)(x)      # Sequence length becomes 16 * 2 = 32
    x = residual_block_1d(x, 64)

    # Restore to 3 channels for final coordinate output
    # This is the 1D equivalent of your final Conv3D(3, kernel_size=(1,1,1), ...)
    x = Conv1D(3, kernel_size=1, padding='same', activation='relu')(x) # Shape is now (batch, 32, 3)

    # Flatten the (batch, 32, 3) tensor to (batch, 96)
    x = Flatten()(x) # 32 * 3 = 96

    # Dense head to map to output_features
    x = Dense(512, activation='relu')(x)
    outputs = Dense(output_features, activation='linear')(x)

    # Create and return the model
    return Model(inputs, outputs)
