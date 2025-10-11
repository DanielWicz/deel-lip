import numpy as np


import keras

from deel.lip.layers.convolutional import SpectralConv2D


def test_spectral_conv2d_can_roundtrip_through_keras_saving(tmp_path):
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(8, 8, 3)),
            SpectralConv2D(filters=4, kernel_size=(3, 3), activation="relu"),
            keras.layers.Flatten(),
            keras.layers.Dense(2),
        ]
    )

    sample = np.random.default_rng(1234).standard_normal((2, 8, 8, 3)).astype(
        np.float32
    )
    original = model.predict(sample)

    target_path = tmp_path / "spectral_conv.keras"
    model.save(target_path)

    restored = keras.models.load_model(target_path, safe_mode=False)
    reloaded = restored.predict(sample)

    np.testing.assert_allclose(original, reloaded, atol=1e-5)
