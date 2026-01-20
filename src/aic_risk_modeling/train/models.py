"""Defines tensorflow keras models used in training"""

from tensorflow import keras
from tensorflow.keras import layers

def get_unet(input_shape):
    inputs = keras.Input(shape=input_shape)
    # --- Encoder ---
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    # --- Bottleneck ---
    b = conv_block(p4, 1024)

    # --- Decoder ---
    d1 = decoder_block(b, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    # --- Output layer ---
    outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    return keras.Model(inputs, outputs, name="U-Net")


def conv_block(inputs, num_filters):
    x = layers.SeparableConv2D(num_filters, 3, padding="same", activation="relu")(inputs)
    x = layers.SeparableConv2D(num_filters, 3, padding="same", activation="relu")(x)
    return x


def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    p = layers.MaxPooling2D((2, 2))(x)
    return x, p


def decoder_block(inputs, skip, num_filters):
    x = layers.Conv2DTranspose(num_filters, 2, strides=2, padding="same")(inputs)
    x = layers.concatenate([x, skip])
    x = conv_block(x, num_filters)
    return x


def get_unet_lite(input_shape):
    inputs = layers.Input(shape=input_shape)

    # --- Encoder (shallow + fewer filters) ---
    s1, p1 = encoder_block(inputs, 32)
    s2, p2 = encoder_block(p1, 64)
    s3, p3 = encoder_block(p2, 128)

    # --- Bottleneck ---
    b = conv_block(p3, 256)

    # --- Decoder ---
    d1 = decoder_block(b, s3, 128)
    d2 = decoder_block(d1, s2, 64)
    d3 = decoder_block(d2, s1, 32)

    # --- Output ---
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(d3)

    return keras.Model(inputs, outputs, name="U-Net-Lite")

def get_mlp(input_shape):
    inputs = keras.Input(shape=input_shape)

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Dense(128, activation='relu')(inputs)
    # x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)

    # Add a per-pixel classification layer
    outputs = layers.Dense(1, activation="sigmoid")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model

def get_multi_scale_mlp_head(input_shape, hidden=128):
    inputs = keras.Input(shape=input_shape)

    # --- scale 1 (original resolution) ---
    s1 = layers.Dense(hidden, activation="gelu")(inputs)

    # --- scale 2 (128x128) ---
    s2 = layers.AveragePooling2D(pool_size=2)(inputs)
    s2 = layers.Dense(hidden, activation="gelu")(s2)
    s2 = layers.UpSampling2D(size=2, interpolation="bilinear")(s2)

    # --- scale 3 (64x64) ---
    s3 = layers.AveragePooling2D(pool_size=4)(inputs)
    s3 = layers.Dense(hidden, activation="gelu")(s3)
    s3 = layers.UpSampling2D(size=4, interpolation="bilinear")(s3)

    # Fuse
    fused = layers.Concatenate()([s1, s2, s3])
    fused = layers.LayerNormalization()(fused)
    fused = layers.Dense(hidden, activation="gelu")(fused)

    outputs = layers.Dense(1, activation='sigmoid')(fused)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model
