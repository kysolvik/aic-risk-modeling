"""Defines tensorflow keras models used in training"""

from tensorflow import keras

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
    outputs = keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    return keras.Model(inputs, outputs, name="U-Net")


def conv_block(inputs, num_filters):
    x = keras.layers.SeparableConv2D(num_filters, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.SeparableConv2D(num_filters, 3, padding="same", activation="relu")(x)
    return x


def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    p = keras.layers.MaxPooling2D((2, 2))(x)
    return x, p


def decoder_block(inputs, skip, num_filters):
    x = keras.layers.Conv2DTranspose(num_filters, 2, strides=2, padding="same")(inputs)
    x = keras.layers.concatenate([x, skip])
    x = conv_block(x, num_filters)
    return x


def get_unet_lite(input_shape):
    inputs = keras.layers.Input(shape=input_shape)

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
    outputs = keras.layers.Conv2D(1, 1, activation="sigmoid")(d3)

    return keras.Model(inputs, outputs, name="U-Net-Lite")

def get_mlp(input_shape):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = keras.layers.Dense(128, activation='relu')(inputs)
    # x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(64, activation='relu')(x)

    # Add a per-pixel classification layer
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model

def get_multi_scale_mlp_head(input_shape, hidden=128):
    inputs = keras.Input(shape=input_shape)

    # --- scale 1 (original resolution) ---
    s1 = keras.layers.Dense(hidden, activation="gelu")(inputs)

    # --- scale 2 (128x128) ---
    s2 = keras.layers.AveragePooling2D(pool_size=2)(inputs)
    s2 = keras.layers.Dense(hidden, activation="gelu")(s2)
    s2 = keras.layers.UpSampling2D(size=2, interpolation="bilinear")(s2)

    # --- scale 3 (64x64) ---
    s3 = keras.layers.AveragePooling2D(pool_size=4)(inputs)
    s3 = keras.layers.Dense(hidden, activation="gelu")(s3)
    s3 = keras.layers.UpSampling2D(size=4, interpolation="bilinear")(s3)

    # Fuse
    fused = keras.layers.Concatenate()([s1, s2, s3])
    fused = keras.layers.LayerNormalization()(fused)
    fused = keras.layers.Dense(hidden, activation="gelu")(fused)

    outputs = keras.layers.Dense(1, activation='sigmoid')(fused)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model

def get_simple_convlstm(input_shape):

    inputs = keras.Input(shape=input_shape)
    t1 = keras.layers.TimeDistributed(
        keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')
    )(inputs)
    c1 = keras.layers.ConvLSTM2D(filters=64, kernel_size=(3, 3),
          input_shape=input_shape, padding='same')(t1)
    b1 = keras.layers.BatchNormalization()(c1)
    outputs = keras.layers.Conv2D(filters=1, kernel_size=(3, 3),
                                  activation="sigmoid", padding="same")(b1)

    model = keras.Model(inputs, outputs)
    return model

def get_convlstm(image_shape, include_metadata=False, metadata_shape=None):
    image_inputs = keras.Input(shape=image_shape, name='image')

    # Image
    c1 = keras.layers.ConvLSTM2D(
        filters=32,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(image_inputs)
    b1 = keras.layers.BatchNormalization()(c1)
    c2 = keras.layers.ConvLSTM2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(b1)
    b2 = keras.layers.BatchNormalization()(c2)
    c3 = keras.layers.ConvLSTM2D(
        filters=32,
        kernel_size=(1, 1),
        padding="same",
        return_sequences=False,
        activation="relu",
    )(b2)
    b3 = keras.layers.BatchNormalization()(c3)

    if include_metadata:
        metadata_inputs = keras.Input(shape=metadata_shape, name='metadata')
        # metadata
        m1 = keras.layers.Dense(16, activation='relu')(metadata_inputs)
        m2 = keras.layers.Dense(8, activation='relu')(m1)
        # Broadcast
        h = b3.shape[1]
        w = b3.shape[2]
        m3 = keras.ops.expand_dims(keras.ops.expand_dims(m2, 1), 1)
        m4 = keras.ops.tile(m3, [1, h, w, 1])

        concat = keras.layers.Concatenate()([b3, m4])
        outputs = keras.layers.Conv2D(filters=1, kernel_size=(3, 3),
                                    activation="sigmoid", padding="same")(concat)
        model = keras.Model([image_inputs, metadata_inputs], outputs)

    else:
        outputs = keras.layers.Conv2D(filters=1, kernel_size=(3, 3),
                                    activation="sigmoid", padding="same")(b3)
        model = keras.Model(image_inputs, outputs)

    return model
