"""TSMixer architecture of the 12-hour IMF Bz forecast model (Section 3.1, Figure 1).

Alternating time-mixing and feature-mixing MLP blocks (Chen et al., 2023) with
layer normalisation and residual connections. The released model uses
hidden_dim=798, n_blocks=4 and dropout_rate=0.5 (dropout is active during training
only; inference is deterministic). About 0.33 M trainable parameters.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


def tsmixer(input_length, input_channels, output_length,
            hidden_dim, n_blocks, dropout_rate=0.5):
    inputs = layers.Input(shape=(input_length, input_channels))
    x = inputs

    for _ in range(n_blocks):
        # Time-mixing MLP (along the 24 time steps, shared across channels)
        y = tf.transpose(x, perm=[0, 2, 1])
        y = layers.LayerNormalization(axis=-1)(y)
        y = layers.Dense(hidden_dim, activation="relu")(y)
        y = layers.Dropout(dropout_rate)(y)
        y = layers.Dense(input_length, activation="relu")(y)
        y = layers.Dropout(dropout_rate)(y)
        y = tf.transpose(y, perm=[0, 2, 1])
        x = layers.Add()([x, y])

        # Feature-mixing MLP (across the 6 solar-wind channels, per time step)
        z = layers.LayerNormalization(axis=-1)(x)
        z = layers.Dense(hidden_dim, activation="relu")(z)
        z = layers.Dropout(dropout_rate)(z)
        z = layers.Dense(input_channels, activation="relu")(z)
        z = layers.Dropout(dropout_rate)(z)
        x = layers.Add()([x, z])

    # Prediction head
    x = layers.Flatten()(x)
    x = layers.Dense(hidden_dim, activation="relu")(x)
    outputs = layers.Dense(output_length)(x)          # linear
    return Model(inputs=inputs, outputs=outputs)
