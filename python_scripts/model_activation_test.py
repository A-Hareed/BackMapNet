import os
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input, Conv3D, Add, MaxPooling3D, UpSampling3D, Reshape, Dense, Flatten, Multiply
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D, Flatten, Dense, BatchNormalization, Add, Input, Lambda, Concatenate, Dropout
from tensorflow.keras.models import Model


def scaled_sigmoid(x, a=0, b=1):
    """
    Scaled sigmoid activation function.

    Args:
    x (tensor): Input tensor.
    a (float): Minimum value of the output range.
    b (float): Maximum value of the output range.

    Returns:
    tensor: Scaled sigmoid output.
    """
    sigmoid = tf.math.sigmoid(x)  # Standard sigmoid
    return a + (b - a) * sigmoid


def residual_block_1D(x, filters):
    y = Conv1D(filters, kernel_size=3, padding='same', activation='relu')(x)
    y = BatchNormalization()(y)
    y = Conv1D(filters, kernel_size=3, padding='same')(y)
    y = BatchNormalization()(y)

    if x.shape[-1] != filters:
        x = Conv1D(filters, kernel_size=1, padding='same')(x)
        x = BatchNormalization()(x)

    return Add()([x, y])

def transformer_block(inputs, num_heads=4, key_dim=32, ff_dim=128, dropout_rate=0.1):
    """
    A single Transformer block for sequence data.
    """
    # Multi-Head Attention
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(inputs, inputs)
    attention_output = Dropout(dropout_rate)(attention_output)
    attention_output = LayerNormalization(epsilon=1e-6)(inputs + attention_output)

    # Feedforward Network
    ff_output = Dense(ff_dim, activation="relu")(attention_output)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    ff_output = Dropout(dropout_rate)(ff_output)
    output = LayerNormalization(epsilon=1e-6)(attention_output + ff_output)
    return output



class FeatureScalingLayer(Layer):
    def __init__(self, scale_factor, feature_indices, **kwargs):
        super(FeatureScalingLayer, self).__init__(**kwargs)
        self.scale_factor = scale_factor
        self.feature_indices = feature_indices

    def call(self, inputs):
        # Create a mask with 1's for all features and scale_factor for important features
        scaling_mask = tf.ones_like(inputs)
        for index in self.feature_indices:
            scaling_mask = scaling_mask * tf.where(
                tf.range(inputs.shape[-1]) == index,
                self.scale_factor,
                1.0
            )
        # Apply scaling mask
        return inputs * scaling_mask


def build_combined_model_masking_LSTM(input_shape, output_size, mask_value=None):
    # Input Layer
    input_layer = Input(shape=input_shape)

    # Split into BLOSUM and bead-specific pathways
    blosum_input = Lambda(lambda input_layer: input_layer[:, :20, :])(input_layer)  # First 20 features
    
    bead_input = Lambda(lambda input_layer: input_layer[:, 20:, :])(input_layer)   # Last 4 features

    # BLOSUM Pathway
    x_blosum = tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(20, 1), activation='relu')(blosum_input)
    x_blosum = tf.keras.layers.LSTM(128, return_sequences=True,activation='relu')(x_blosum)

    # Bead-Specific Pathway
    x_bead = tf.keras.layers.Reshape((4,))(bead_input)
    x_bead = Dense(34, activation='relu')(x_bead)
    x_bead = BatchNormalization()(x_bead)
    x_bead = Dense(64, activation='relu')(x_bead)
    x_bead = BatchNormalization()(x_bead)
    x_bead = Dense(128, activation='relu')(x_bead)
    x_bead = BatchNormalization()(x_bead)
    x_bead = tf.keras.layers.Reshape((-1,128))(x_bead)


    # Combine the two pathways
    combined = Concatenate(axis=1)([x_blosum, x_bead])

    # Encoder Pathway
    x = Conv1D(32, kernel_size=3, padding='same', activation='relu')(combined)
    x = BatchNormalization()(x)
    x = residual_block_1D(x, 32)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = residual_block_1D(x, 64)
    x = MaxPooling1D(pool_size=2)(x)
    # Additional block: Increase complexity further
    x = Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = residual_block_1D(x, 128)
    x = MaxPooling1D(pool_size=2)(x)

    # Decoder Pathway
    x = UpSampling1D(size=2)(x)
    x = residual_block_1D(x, 128)   
    x = UpSampling1D(size=2)(x)
    x = residual_block_1D(x, 64)

    x = UpSampling1D(size=2)(x)
    x = residual_block_1D(x, 32)

    # Flatten the output and add dense layers
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)  # Increased neurons from 128 to 256
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x) 
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
#    output = Dense(output_size, activation=lambda x: scaled_sigmoid(x, a=0, b=0.88))(x)
    output = Dense(output_size, activation='linear')(x)


    # Optional masking
    if mask_value is not None:
        mask_input = Input(shape=(output_size,))
        output = Multiply()([output, mask_input])
        model = Model(inputs=[input_layer, mask_input], outputs=output)
    else:
        model = Model(inputs=input_layer, outputs=output)

    return model




# Apply masking to layer
def build_combined_model_with_transformer(input_shape, output_size, mask_value=None):
    # Input Layer
    input_layer = Input(shape=input_shape)

    # Split into BLOSUM and bead-specific pathways
    blosum_input = Lambda(lambda input_layer: input_layer[:, :, 4:])(input_layer)  # First 20 features
    bead_input = Lambda(lambda input_layer: input_layer[:, :, :4])(input_layer)   # Last 4 features

    # BLOSUM Pathway
    x_blosum = Dense(64, activation='relu')(blosum_input)
    x_blosum = BatchNormalization()(x_blosum)
    x_blosum = Dense(32, activation='relu')(x_blosum)
    x_blosum = BatchNormalization()(x_blosum)

    # Bead-Specific Pathway
    x_bead = Dense(16, activation='relu')(bead_input)
    x_bead = BatchNormalization()(x_bead)

    # Combine the two pathways
    combined = Concatenate()([x_blosum, x_bead])

    # Encoder Pathway
    x = Conv1D(32, kernel_size=3, padding='same', activation='relu')(combined)
    x = BatchNormalization()(x)
    x = residual_block_1D(x, 32)

    # Add a Transformer Layer after initial feature extraction
    x = transformer_block(x, num_heads=4, key_dim=32, ff_dim=128)

    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = residual_block_1D(x, 64)

    # Add another Transformer Layer
    x = transformer_block(x, num_heads=4, key_dim=64, ff_dim=256)

    x = MaxPooling1D(pool_size=2)(x)

    # Decoder Pathway
    x = UpSampling1D(size=2)(x)
    x = residual_block_1D(x, 64)

    x = UpSampling1D(size=2)(x)
    x = residual_block_1D(x, 32)

    # Flatten the output and add dense layers
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
#    output = Dense(output_size, activation=lambda x: scaled_sigmoid(x, a=0, b=0.88))(x)
    output = Dense(output_size, activation='sigmoid')(x)
    # Optional masking
    if mask_value is not None:
        mask_input = Input(shape=(output_size,))
        output = Multiply()([output, mask_input])
        model = Model(inputs=[input_layer, mask_input], outputs=output)
    else:
        model = Model(inputs=input_layer, outputs=output)

    return model

# Apply masking to layer


def build_combined_model_masking(input_shape, output_size, mask_value=None):
    # Input Layer
    input_layer = Input(shape=input_shape)

    # Split into BLOSUM and bead-specific pathways
    blosum_input = Lambda(lambda input_layer: input_layer[:, :, :20])(input_layer)  # First 20 features
    bead_input = Lambda(lambda input_layer: input_layer[:, :, 20:])(input_layer)   # Last 4 features

    # BLOSUM Pathway
    x_blosum = Dense(64, activation='relu')(blosum_input)
    x_blosum = BatchNormalization()(x_blosum)
    x_blosum = Dense(32, activation='relu')(x_blosum)
    x_blosum = BatchNormalization()(x_blosum)

    # Bead-Specific Pathway
    x_bead = Dense(16, activation='relu')(bead_input)
    x_bead = BatchNormalization()(x_bead)

    # Combine the two pathways
    combined = Concatenate()([x_blosum, x_bead])

    # Encoder Pathway
    x = Conv1D(32, kernel_size=3, padding='same', activation='relu')(combined)
    x = BatchNormalization()(x)
    x = residual_block_1D(x, 32)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = residual_block_1D(x, 64)
    x = MaxPooling1D(pool_size=2)(x)

    # Decoder Pathway
    x = UpSampling1D(size=2)(x)
    x = residual_block_1D(x, 64)

    x = UpSampling1D(size=2)(x)
    x = residual_block_1D(x, 32)

    # Flatten the output and add dense layers
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
#    output = Dense(output_size, activation=lambda x: scaled_sigmoid(x, a=0, b=0.88))(x)
    output = Dense(output_size, activation='linear')(x)


    # Optional masking
    if mask_value is not None:
        mask_input = Input(shape=(output_size,))
        output = Multiply()([output, mask_input])
        model = Model(inputs=[input_layer, mask_input], outputs=output)
    else:
        model = Model(inputs=input_layer, outputs=output)

    return model


# Build model function with feature scaling
def build_1d_conv_autoencoder2(input_shape, output_size, scale_factor=2, feature_indices=[0, 1]):
    input_layer = Input(shape=input_shape)

    # Apply feature scaling
    x = FeatureScalingLayer(scale_factor=scale_factor, feature_indices=feature_indices)(input_layer)
    # Reshape for Conv1D compatibility
    x = Reshape((input_shape[0], 1))(x)  # Expands dimensions to (batch_size, 24, 1)    
    # Encoder
    x = Conv1D(32, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = residual_block_1D(x, 32)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = residual_block_1D(x, 64)
    x = MaxPooling1D(pool_size=2)(x)

    # Decoder
    x = UpSampling1D(size=2)(x)
    x = residual_block_1D(x, 64)

    x = UpSampling1D(size=2)(x)
    x = residual_block_1D(x, 32)

    # Flatten the output and add dense layers
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(output_size, activation='sigmoid')(x)

    model = Model(input_layer, x)
    return model

# Combined Model
def build_combined_model(input_shape, output_size):
    # Input Layer
    input_layer = Input(shape=input_shape)


    # Split into BLOSUM and bead-specific pathways
    blosum_input = Lambda(lambda input_layer: input_layer[:, :, :20])(input_layer)  # First 20 features
    bead_input = Lambda(lambda input_layer: input_layer[:, :, 20:])(input_layer)   # Last 4 features

    # BLOSUM Pathway
    x_blosum = Dense(64, activation='relu')(blosum_input)
    x_blosum = BatchNormalization()(x_blosum)
    x_blosum = Dense(32, activation='relu')(x_blosum)
    x_blosum = BatchNormalization()(x_blosum)

    # Bead-Specific Pathway
    x_bead = Dense(16, activation='relu')(bead_input)
    x_bead = BatchNormalization()(x_bead)

    # Combine the two pathways
    combined = Concatenate()([x_blosum, x_bead])

    # Encoder Pathway
    x = Conv1D(32, kernel_size=3, padding='same', activation='relu')(combined)
    x = BatchNormalization()(x)
    x = residual_block_1D(x, 32)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = residual_block_1D(x, 64)
    x = MaxPooling1D(pool_size=2)(x)

    # Decoder Pathway
    x = UpSampling1D(size=2)(x)
    x = residual_block_1D(x, 64)

    x = UpSampling1D(size=2)(x)
    x = residual_block_1D(x, 32)

    # Flatten the output and add dense layers
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(output_size, activation=lambda x: scaled_sigmoid(x, a=0, b=0.88))(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=x)
    return model


def build_1d_conv_autoencoder_sigmoid2(input_shape, output_size):
    input_layer = Input(shape=input_shape)

    # Encoder
    x = Conv1D(32, kernel_size=3, padding='same', activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = residual_block_1D(x, 32)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = residual_block_1D(x, 64)
    x = MaxPooling1D(pool_size=2)(x)

    # Decoder
    x = UpSampling1D(size=2)(x)
    x = residual_block_1D(x, 64)

    x = UpSampling1D(size=2)(x)
    x = residual_block_1D(x, 32)

    # Flatten the output and add a dense layer
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(output_size, activation=lambda x: scaled_sigmoid(x, a=0, b=0.88))(x)

    model = Model(input_layer, x)
    return model


def build_1d_conv_autoencoder_sigmoid(input_shape, output_size):
    input_layer = Input(shape=input_shape)

    # Encoder
    x = Conv1D(32, kernel_size=3, padding='same', activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = residual_block_1D(x, 32)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = residual_block_1D(x, 64)
    x = MaxPooling1D(pool_size=2)(x)

    # Decoder
    x = UpSampling1D(size=2)(x)
    x = residual_block_1D(x, 64)

    x = UpSampling1D(size=2)(x)
    x = residual_block_1D(x, 32)

    # Flatten the output and add a dense layer
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(output_size, activation='sigmoid')(x)

    model = Model(input_layer, x)
    return model

def build_1d_conv_autoencoder(input_shape, output_size):
    input_layer = Input(shape=input_shape)

    # Encoder
    x = Conv1D(32, kernel_size=3, padding='same', activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = residual_block_1D(x, 32)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = residual_block_1D(x, 64)
    x = MaxPooling1D(pool_size=2)(x)

    # Decoder
    x = UpSampling1D(size=2)(x)
    x = residual_block_1D(x, 64)

    x = UpSampling1D(size=2)(x)
    x = residual_block_1D(x, 32)

    # Flatten the output and add a dense layer
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(output_size, activation='linear')(x)

    model = Model(input_layer, x)
    return model




# Define the updated residual block
def residual_block(x, filters):
    # Convolution layers in the block
    y = Conv3D(filters, kernel_size=(3, 3, 3), padding='same', activation='relu')(x)
    y = Conv3D(filters, kernel_size=(3, 3, 3), padding='same', activation='relu')(y)
    y = Conv3D(filters, kernel_size=(3, 3, 3), padding='same', activation='relu')(y)

    # Adjust input x dimensions if needed (i.e., if the number of filters is different)
    if x.shape[-1] != filters:
        x = Conv3D(filters, kernel_size=(1, 1, 1), padding='same')(x)

    return Add()([x, y])

def build_3d_conv_autoencoder2(input_length, output_features):
    # Input layer (batch, 32, 3) represents 32 atoms with (x, y, z) coordinates
    inputs = Input(shape=(96,))

    # Reshape to 5D shape: (32, 1, 3, 1) -> (depth, height, width, channels)
    x = Reshape((32, 1, 3, 1))(inputs)

    # Encoder
    x = Conv3D(64, kernel_size=(3, 3, 3), padding='same', activation='relu')(x)
    x = residual_block(x, 64)
    x = MaxPooling3D(pool_size=(2, 1, 1))(x)  # Downsample spatial dims

    x = residual_block(x, 128)
    x = MaxPooling3D(pool_size=(2, 1, 1))(x)  # Further downsample spatial dims
    
    # Decoder
    x = UpSampling3D(size=(2, 1, 1))(x)  # Upsample to restore spatial dims
    x = residual_block(x, 128)
    
    x = UpSampling3D(size=(2, 1, 1))(x)  # Upsample back to original size
    x = residual_block(x, 64)

    # x = UpSampling3D(size=(2, 1, 1))(x)  # Upsample back to original size
    # x = residual_block(x, 32)

    # x = UpSampling3D(size=(2, 1, 1))(x)  # Upsample back to original size
    # x = residual_block(x, 16)
    # Output layer
    x = Conv3D(3, kernel_size=(3, 3, 3), padding='same', activation='relu')(x)  # Restore original feature dimension (x, y, z)

    # Flattening the 3D convolution output
    x = Flatten()(x)

    # Dense layer after flattening
    x = Dense(512, activation='relu')(x)

    # Final output layer
    outputs = Dense(384, activation='linear')(x)  # Final output of size 384

    # Model creation
    model = Model(inputs, outputs)
    return model

# Build the encoder-decoder model
def build_3d_conv_autoencoder(input_length, output_features):
    # Input layer (batch, 32, 3) represents 32 atoms with (x, y, z) coordinates
    inputs = Input(shape=(96, ))

    # Reshape to 5D shape: (32, 1, 3, 1) -> (depth, height, width, channels)
    x = Reshape((32, 1, 3, 1))(inputs)

    # Encoder
    x = Conv3D(64, kernel_size=(3, 3, 3), padding='same', activation='relu')(x)
    x = residual_block(x, 64)
    x = MaxPooling3D(pool_size=(2, 1, 1))(x)  # Downsample spatial dims

    x = residual_block(x, 128)
    x = MaxPooling3D(pool_size=(2, 1, 1))(x)  # Further downsample spatial dims
    
    # Decoder
    x = UpSampling3D(size=(2, 1, 1))(x)  # Upsample to restore spatial dims
    x = residual_block(x, 128)
    
    x = UpSampling3D(size=(2, 1, 1))(x)  # Upsample back to original size
    x = residual_block(x, 64)

    x = UpSampling3D(size=(2, 1, 1))(x)  # Upsample back to original size
    x = residual_block(x, 32)

    x = UpSampling3D(size=(2, 1, 1))(x)  # Upsample back to original size
    x = residual_block(x, 16)
    # Output layer
    x = Conv3D(3, kernel_size=(3, 3, 3), padding='same', activation='relu')(x)  # Restore original feature dimension (x, y, z)

    # Flattening the 3D convolution output
    x = Flatten()(x)

    # Dense layer after flattening
    x = Dense(512, activation='relu')(x)

    # Final output layer
    outputs = Dense(384, activation='linear')(x)  # Final output of size 384

    # Model creation
    model = Model(inputs, outputs)
    return model




def build_seq2seq_unet_1d_V2(input_length, num_features):
    # Input layer
    inputs = layers.Input(shape=(input_length, num_features))
    reshaped_inputs = layers.Reshape((96, 1))(inputs)
    
    # Contracting path (encoder)
    conv1 = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu',
                          kernel_regularizer=regularizers.l2(0.001))(reshaped_inputs)
    conv1 = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu',
                          kernel_regularizer=regularizers.l2(0.001))(conv1)
    conv1 = layers.Dropout(0.3)(conv1)  # Dropout to reduce overfitting
    pool1 = layers.MaxPooling1D(pool_size=2)(conv1)

    conv2 = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu',
                          kernel_regularizer=regularizers.l2(0.001))(pool1)
    conv2 = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu',
                          kernel_regularizer=regularizers.l2(0.001))(conv2)
    conv2 = layers.Dropout(0.3)(conv2)  # Dropout
    pool2 = layers.MaxPooling1D(pool_size=2)(conv2)

    conv3 = layers.Conv1D(256, kernel_size=3, padding='same', activation='relu',
                          kernel_regularizer=regularizers.l2(0.001))(pool2)
    conv3 = layers.Conv1D(256, kernel_size=3, padding='same', activation='relu',
                          kernel_regularizer=regularizers.l2(0.001))(conv3)
    conv3 = layers.Dropout(0.3)(conv3)  # Dropout
    pool3 = layers.MaxPooling1D(pool_size=2)(conv3)

    conv4 = layers.Conv1D(512, kernel_size=3, padding='same', activation='relu',
                          kernel_regularizer=regularizers.l2(0.001))(pool3)
    conv4 = layers.Conv1D(512, kernel_size=3, padding='same', activation='relu',
                          kernel_regularizer=regularizers.l2(0.001))(conv4)
    conv4 = layers.Dropout(0.3)(conv4)  # Dropout
    pool4 = layers.MaxPooling1D(pool_size=2)(conv4)

    # Bottleneck (deepest part)
    conv5 = layers.Conv1D(1024, kernel_size=3, padding='same', activation='relu',
                          kernel_regularizer=regularizers.l2(0.001))(pool4)
    conv5 = layers.Conv1D(1024, kernel_size=3, padding='same', activation='relu',
                          kernel_regularizer=regularizers.l2(0.001))(conv5)
    conv5 = layers.Dropout(0.3)(conv5)  # Dropout

    # Expansive path (decoder)
    up6 = layers.Conv1DTranspose(512, kernel_size=2, strides=2, padding='same')(conv5)
    up6 = layers.Concatenate()([up6, conv4])
    conv6 = layers.Conv1D(512, kernel_size=3, padding='same', activation='relu',
                          kernel_regularizer=regularizers.l2(0.001))(up6)
    conv6 = layers.Conv1D(512, kernel_size=3, padding='same', activation='relu',
                          kernel_regularizer=regularizers.l2(0.001))(conv6)
    conv6 = layers.Dropout(0.3)(conv6)  # Dropout

    up7 = layers.Conv1DTranspose(256, kernel_size=2, strides=2, padding='same')(conv6)
    up7 = layers.Concatenate()([up7, conv3])
    conv7 = layers.Conv1D(256, kernel_size=3, padding='same', activation='relu',
                          kernel_regularizer=regularizers.l2(0.001))(up7)
    conv7 = layers.Conv1D(256, kernel_size=3, padding='same', activation='relu',
                          kernel_regularizer=regularizers.l2(0.001))(conv7)
    conv7 = layers.Dropout(0.3)(conv7)  # Dropout

    up8 = layers.Conv1DTranspose(128, kernel_size=2, strides=2, padding='same')(conv7)
    up8 = layers.Concatenate()([up8, conv2])
    conv8 = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu',
                          kernel_regularizer=regularizers.l2(0.001))(up8)
    conv8 = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu',
                          kernel_regularizer=regularizers.l2(0.001))(conv8)
    conv8 = layers.Dropout(0.3)(conv8)  # Dropout

    up9 = layers.Conv1DTranspose(64, kernel_size=2, strides=2, padding='same')(conv8)
    up9 = layers.Concatenate()([up9, conv1])
    conv9 = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu',
                          kernel_regularizer=regularizers.l2(0.001))(up9)
    conv9 = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu',
                          kernel_regularizer=regularizers.l2(0.001))(conv9)
    conv9 = layers.Dropout(0.3)(conv9)  # Dropout

    # Output layer
    outputs = layers.Conv1D(384, kernel_size=1, activation='linear')(conv9)
    outputs = layers.GlobalAveragePooling1D()(outputs)
    outputs = layers.Flatten()(outputs)  # Flatten the output

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model


def build_seq2seq_unet_1d(input_length, num_features):
    # Input layer
    inputs = layers.Input(shape=(input_length, num_features))
    reshaped_inputs = layers.Reshape((96, 1))(inputs)
    # Contracting path (encoder)
    conv1 = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(reshaped_inputs)
    conv1 = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(conv1)
    pool1 = layers.MaxPooling1D(pool_size=2)(conv1)

    conv2 = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(pool1)
    conv2 = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(conv2)
    pool2 = layers.MaxPooling1D(pool_size=2)(conv2)

    conv3 = layers.Conv1D(256, kernel_size=3, padding='same', activation='relu')(pool2)
    conv3 = layers.Conv1D(256, kernel_size=3, padding='same', activation='relu')(conv3)
    pool3 = layers.MaxPooling1D(pool_size=2)(conv3)

    conv4 = layers.Conv1D(512, kernel_size=3, padding='same', activation='relu')(pool3)
    conv4 = layers.Conv1D(512, kernel_size=3, padding='same', activation='relu')(conv4)
    pool4 = layers.MaxPooling1D(pool_size=2)(conv4)

    # Bottleneck (deepest part)
    conv5 = layers.Conv1D(1024, kernel_size=3, padding='same', activation='relu')(pool4)
    conv5 = layers.Conv1D(1024, kernel_size=3, padding='same', activation='relu')(conv5)

    # Expansive path (decoder)
    up6 = layers.Conv1DTranspose(512, kernel_size=2, strides=2, padding='same')(conv5)
    up6 = layers.Concatenate()([up6, conv4])
    conv6 = layers.Conv1D(512, kernel_size=3, padding='same', activation='relu')(up6)
    conv6 = layers.Conv1D(512, kernel_size=3, padding='same', activation='relu')(conv6)

    up7 = layers.Conv1DTranspose(256, kernel_size=2, strides=2, padding='same')(conv6)
    up7 = layers.Concatenate()([up7, conv3])
    conv7 = layers.Conv1D(256, kernel_size=3, padding='same', activation='relu')(up7)
    conv7 = layers.Conv1D(256, kernel_size=3, padding='same', activation='relu')(conv7)

    up8 = layers.Conv1DTranspose(128, kernel_size=2, strides=2, padding='same')(conv7)
    up8 = layers.Concatenate()([up8, conv2])
    conv8 = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(up8)
    conv8 = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(conv8)

    up9 = layers.Conv1DTranspose(64, kernel_size=2, strides=2, padding='same')(conv8)
    up9 = layers.Concatenate()([up9, conv1])
    conv9 = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(up9)
    conv9 = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(conv9)

    # Output layer (96 time steps, 384 features per time step)

    outputs = layers.Conv1D(384, kernel_size=1, activation='linear')(conv9)
    outputs = layers.GlobalAveragePooling1D()(outputs)
    # Flatten the output to (None, 96 * 384), collapsing the time steps (96)
    outputs = layers.Flatten()(outputs)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model



def create_model_LeakyRELU(input_shape, num_outputs, alpha=0.2, activation_layer=tf.keras.layers.LeakyReLU):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(2048, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-5))(inputs)
    x = activation_layer(alpha=alpha)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(1024, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = activation_layer(alpha=alpha)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Dense(512, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = activation_layer(alpha=alpha)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Dense(256, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = activation_layer(alpha=alpha)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Dense(256, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = activation_layer(alpha=alpha)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Dense(128, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = activation_layer(alpha=alpha)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Dense(128, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = activation_layer(alpha=alpha)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Dense(64, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = activation_layer(alpha=alpha)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    output = tf.keras.layers.Dense(num_outputs, activation='linear')(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model


def create_model_Swish(input_shape, num_outputs):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(2048, kernel_initializer='he_normal')(inputs)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(1024, kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Dense(512, kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Dense(256, kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Dense(256, kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Dense(128, kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Dense(128, kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Dense(64, kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    output = tf.keras.layers.Dense(num_outputs, activation='linear')(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model

