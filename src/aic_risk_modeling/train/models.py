"""Defines keras models used in training"""

import keras
from keras import layers, models

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
    outputs = layers.Conv2D(1, 1, padding="same", activation="relu")(d4)

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


def get_unet_lite(input_shape, input_name):
    inputs = keras.Input(shape=input_shape,  name=input_name)

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
    outputs = layers.Conv2D(1, 1, padding="same", activation="relu")(d3)

    return keras.Model(inputs, outputs, name="U-Net-Lite")

def get_mlp(input_shape, input_name):
    inputs = keras.Input(shape=input_shape, name=input_name)

    # Entry block
    x = layers.Dense(1024, activation='relu')(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)

    # Add a per-pixel classification layer
    outputs = layers.Dense(1, activation="sigmoid")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model

def get_mlp_for_fusion(input_shape, input_name):
    inputs = keras.Input(shape=input_shape, name=input_name)

    # Entry block
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(16, activation='relu')(x)

    # Add a per-pixel classification layer

    # Reshape and broadcast  to (128, 128, 16)
    x = layers.Reshape((1, 1, 16))(x)
    outputs = keras.ops.tile(x, [1, 128, 128, 1])

    # Define the model
    model = keras.Model(inputs, outputs)
    return model

def get_multi_scale_mlp_head(input_shape, input_name, hidden=128):
    inputs = keras.Input(shape=input_shape, name=input_name)

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

def get_simple_convlstm(input_shape, input_name):

    inputs = keras.Input(shape=input_shape, name=input_name)
    t1 = layers.TimeDistributed(
        layers.Conv2D(32, (3, 3), padding='same', activation='relu')
    )(inputs)
    c1 = layers.ConvLSTM2D(filters=64, kernel_size=(3, 3),
          input_shape=input_shape, padding='same')(t1)
    b1 = layers.BatchNormalization()(c1)
    outputs = layers.Conv2D(filters=1, kernel_size=(3, 3),
                                  activation="sigmoid", padding="same")(b1)

    model = keras.Model(inputs, outputs)
    return model

def get_convlstm(input_shape, input_name, for_fusion=True):
    image_inputs = keras.Input(shape=input_shape, name=input_name)

    # Image
    c1 = layers.ConvLSTM2D(
        filters=128,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True,
    )(image_inputs)
    b1 = layers.BatchNormalization()(c1)
    c2 = layers.ConvLSTM2D(
        filters=128,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
    )(b1)
    b2 = layers.BatchNormalization()(c2)
    c3 = layers.ConvLSTM2D(
        filters=128,
        kernel_size=(1, 1),
        padding="same",
        return_sequences=False
    )(b2)
    b3 = layers.BatchNormalization()(c3)


    if not for_fusion:
        outputs = layers.Conv2D(filters=1, kernel_size=(3, 3),
                                    activation="sigmoid", padding="same")(b3)
        model = keras.Model(image_inputs, outputs)
    else:
        model = keras.Model(image_inputs, b3)

    return model


def get_convlstm_bottleneck(input_shape, input_name, for_fusion=True):
    image_inputs = keras.Input(shape=input_shape, name=input_name)

    # --- ENCODER: Shrink spatially
    # Downsample 128x128 -> 64x64
    x = layers.TimeDistributed(layers.Conv2D(32, (3, 3), strides=2, padding="same", activation="relu"))(image_inputs)
    # Downsample 64x64 -> 32x32
    x = layers.TimeDistributed(layers.Conv2D(64, (3, 3), strides=2, padding="same", activation="relu"))(x)

    # --- TEMPORAL CORE: ConvLSTM at lower res
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=False, # We collapse time here
        activation="tanh"       # Enables CuDNN optimization
    )(x)
    x = layers.BatchNormalization()(x)

    # --- DECODER: Recover 128x128 resolution ---
    # 32x32 -> 64x64
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, padding="same", activation="relu")(x)
    # 64x64 -> 128x128
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, padding="same", activation="relu")(x)

    if not for_fusion:
        outputs = layers.Conv2D(filters=1, kernel_size=(3, 3),
                                activation="sigmoid", padding="same")(x)
        model = keras.Model(image_inputs, outputs)
    else:
        # Returns (Batch, 128, 128, 32) ready for fusion
        model = keras.Model(image_inputs, x)

    return model

def get_lstm(input_shape, input_name):
        # Input shape should be (timesteps, features)
    input = keras.Input(shape=input_shape, name=input_name)

    l1 = layers.LSTM(32, return_sequences=True)(input)
    d1 = layers.Dropout(0.2)(l1)

    l2 = layers.LSTM(32, return_sequences=False)(d1)
    d2 = layers.Dropout(0.2)(l2)

    # Reshape and broadcast  to (128, 128, 16)
    x = layers.Reshape((1, 1, 32))(d2)
    output = keras.ops.tile(x, [1, 128, 128, 1])

    model = keras.Model(input, output)

    return model

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # 1. Multi-Head Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # 2. Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def get_transformer(input_shape, input_name):
    inputs = keras.Input(shape=input_shape, name=input_name)

    # Step 1: Project the 5 features up to 32 (to match your current capacity)
    x = layers.Dense(32)(inputs)

    # Step 2: Add Positional Encoding
    # Since Transformers don't know the "order" of time, we inject it.
    positions = keras.ops.arange(start=0, stop=input_shape[0], step=1)
    pos_encoding = layers.Embedding(input_dim=input_shape[0], output_dim=32)(positions)
    x = x + pos_encoding

    # Step 3: Transformer Block(s)
    # head_size: dimension of queries/keys/values
    # num_heads: number of attention "eyes"
    # ff_dim: internal hidden layer size of the feed-forward network
    x = transformer_encoder(x, head_size=16, num_heads=4, ff_dim=64, dropout=0.1)

    # Step 4: Reduction
    # Instead of an LSTM with return_sequences=False, we use Global Average Pooling
    # to turn the (60, 32) sequence into a single (32,) vector for fusion.
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)

    # Step 5: Spatial Expansion to 128x128
    # First, turn (32,) into (1, 1, 32)
    x = layers.Reshape((1, 1, 32))(x)

    # Second, project to the number of channels you want for fusion
    x = layers.Conv2D(16, (1, 1), activation="relu")(x)

    # Third, UpSample to match image resolution (1, 1) -> (128, 128)
    # This effectively "tiles" the data across the spatial grid
    outputs = layers.UpSampling2D(size=(128, 128), interpolation="nearest")(x)

    return keras.Model(inputs, outputs)

def get_identity(input_shape, input_name):
    # Input shape should be (timesteps, features)
    input = keras.Input(shape=input_shape, name=input_name)

    output = layers.Identity()(input)

    model = keras.Model(input, output)

    return model

def decoder_fusion(branch_models):
    """
    branch_models: List of Keras models (e.g., [lstm_branch1, lstm_branch2, cnn_branch])
    """

    model_outputs = [m.output for m in branch_models]
    model_inputs = {m.input.name: m.input for m in branch_models}
    print('Model inputs:', model_inputs)
    print('Model outputs:', model_outputs)

    fused = layers.Concatenate(axis=-1)(model_outputs)

    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(fused)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)

    mask_output = layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(x)

    # Define the final Multi-Input model
    full_model = models.Model(inputs=model_inputs, outputs=mask_output)

    return full_model
