#!/usr/bin/env python3
"""Variational Autoencoder"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder.

    Returns:
        encoder: Encoder model
        decoder: Decoder model
        auto: Full autoencoder model
    """

    # Encoder
    encoder_input = keras.Input(shape=(input_dims,))

    x = encoder_input
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation="relu")(x)

    # Mean and log variance
    z_mean = keras.layers.Dense(latent_dims, activation=None)(x)
    z_log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    # Sampling layer
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = keras.backend.random_normal(
            shape=keras.backend.shape(z_mean)
        )
        return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon

    z = keras.layers.Lambda(sampling)([z_mean, z_log_var])

    encoder = keras.Model(
        encoder_input,
        [z, z_mean, z_log_var],
        name="encoder"
    )

    # Decoder
    decoder_input = keras.Input(shape=(latent_dims,))

    x = decoder_input
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation="relu")(x)

    decoder_output = keras.layers.Dense(
        input_dims,
        activation="sigmoid"
    )(x)

    decoder = keras.Model(
        decoder_input,
        decoder_output,
        name="decoder"
    )

    # Full autoencoder
    auto_output = decoder(z)
    auto = keras.Model(
        encoder_input,
        auto_output,
        name="autoencoder"
    )

    auto.compile(
        optimizer="adam",
        loss="binary_crossentropy"
    )

    return encoder, decoder, auto
