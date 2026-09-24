#!/usr/bin/env python3
"""Variational Autoencoder"""

import tensorflow.keras as keras

def autoencoder(input_dims, hidden_layers, latent_dims):
"""
Creates a variational autoencoder.

```
Args:
    input_dims: dimensions of the model input
    hidden_layers: list containing the number of nodes for each
        hidden layer in the encoder
    latent_dims: dimensions of the latent space representation

Returns:
    encoder, decoder, auto
"""
X = keras.Input(shape=(input_dims,))

Y = X
for units in hidden_layers:
    Y = keras.layers.Dense(
        units,
        activation='relu'
    )(Y)

z_mean = keras.layers.Dense(
    latent_dims,
    activation=None
)(Y)

z_log_sigma = keras.layers.Dense(
    latent_dims,
    activation=None
)(Y)

def sampling(args):
    """Samples a point from the latent distribution."""
    z_mean, z_log_sigma = args
    epsilon = keras.backend.random_normal(
        shape=keras.backend.shape(z_mean)
    )
    return z_mean + keras.backend.exp(
        z_log_sigma / 2
    ) * epsilon

z = keras.layers.Lambda(
    sampling,
    output_shape=(latent_dims,)
)([z_mean, z_log_sigma])

encoder = keras.Model(
    X,
    [z, z_mean, z_log_sigma]
)

X_decoder = keras.Input(
    shape=(latent_dims,)
)

Y = X_decoder
for units in reversed(hidden_layers):
    Y = keras.layers.Dense(
        units,
        activation='relu'
    )(Y)

Y = keras.layers.Dense(
    input_dims,
    activation='sigmoid'
)(Y)

decoder = keras.Model(
    X_decoder,
    Y
)

auto_output = decoder(z)

auto = keras.Model(
    X,
    auto_output
)

def vae_loss(x, x_decoder_mean):
    """Calculates the variational autoencoder loss."""
    reconstruction_loss = keras.backend.binary_crossentropy(
        x,
        x_decoder_mean
    )
    reconstruction_loss = keras.backend.sum(
        reconstruction_loss,
        axis=1
    )

    kl_loss = -0.5 * keras.backend.sum(
        1
        + z_log_sigma
        - keras.backend.square(z_mean)
        - keras.backend.exp(z_log_sigma),
        axis=1
    )

    return reconstruction_loss + kl_loss

auto.compile(
    loss=vae_loss,
    optimizer='adam'
)

return encoder, decoder, auto
```
