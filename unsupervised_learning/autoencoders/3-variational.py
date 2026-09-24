#!/usr/bin/env python3
""" Variational Autoencoder """

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder.

    Args:
        input_dims: Dimensions of the model input.
        hidden_layers: Number of nodes in each hidden layer.
        latent_dims: Dimensions of the latent space.

    Returns:
        encoder, decoder, auto: The encoder, decoder, and autoencoder models.
    """
    X_input = keras.Input(shape=(input_dims,))

    Y = X_input
    for units in hidden_layers:
        Y = keras.layers.Dense(units, activation='relu')(Y)

    z_mean = keras.layers.Dense(latent_dims, activation=None)(Y)
    z_log_sigma = keras.layers.Dense(latent_dims, activation=None)(Y)

    def sampling(args):
        """Samples a point from the latent distribution."""
        z_m, z_log_s = args
        batch = keras.backend.shape(z_m)[0]
        dim = keras.backend.int_shape(z_m)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_m + keras.backend.exp(z_log_s / 2) * epsilon

    z = keras.layers.Lambda(
        sampling,
        output_shape=(latent_dims,)
    )([z_mean, z_log_sigma])

    encoder = keras.Model(
        X_input,
        [z, z_mean, z_log_sigma]
    )

    X_decode = keras.Input(shape=(latent_dims,))

    Y = keras.layers.Dense(
        hidden_layers[-1],
        activation='relu'
    )(X_decode)

    for units in reversed(hidden_layers[:-1]):
        Y = keras.layers.Dense(
            units,
            activation='relu'
        )(Y)

    output = keras.layers.Dense(
        input_dims,
        activation='sigmoid'
    )(Y)

    decoder = keras.Model(X_decode, output)

    decoder_output = decoder(z)
    auto = keras.Model(X_input, decoder_output)

    def vae_loss(x, x_decoder_mean):
        """Calculates the VAE loss."""
        reconstruction_loss = keras.backend.binary_crossentropy(
            x, x_decoder_mean
        )
        reconstruction_loss = keras.backend.sum(
            reconstruction_loss,
            axis=1
        )

        kl_loss = 1 + z_log_sigma - keras.backend.square(z_mean)
        kl_loss -= keras.backend.exp(z_log_sigma)
        kl_loss = keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        return reconstruction_loss + kl_loss

    auto.compile(
        loss=vae_loss,
        optimizer='adam'
    )

    return encoder, decoder, auto
