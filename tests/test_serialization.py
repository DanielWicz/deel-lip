import numpy as np


import keras

from deel.lip.layers.convolutional import SpectralConv2D


def _build_spectral_conv2d_model():
    return keras.Sequential(
        [
            keras.layers.Input(shape=(8, 8, 3)),
            SpectralConv2D(filters=4, kernel_size=(3, 3), activation="relu"),
            keras.layers.Flatten(),
            keras.layers.Dense(2),
        ]
    )


def test_spectral_conv2d_can_roundtrip_through_keras_saving(tmp_path):
    model = _build_spectral_conv2d_model()

    sample = np.random.default_rng(1234).standard_normal((2, 8, 8, 3)).astype(
        np.float32
    )
    original = model.predict(sample)

    target_path = tmp_path / "spectral_conv.keras"
    model.save(target_path)

    restored = keras.models.load_model(target_path, safe_mode=False)
    reloaded = restored.predict(sample)

    np.testing.assert_allclose(original, reloaded, atol=1e-5)


def test_spectral_conv2d_restores_internal_state(tmp_path):
    model = _build_spectral_conv2d_model()

    sample = np.zeros((1, 8, 8, 3), dtype=np.float32)
    _ = model(sample)  # ensure the model is built

    layer = model.layers[1]

    u_value = np.arange(layer.filters, dtype=np.float32)[None, :]
    sigma_value = np.array([[2.5]], dtype=np.float32)
    wbar_value = np.arange(np.prod(layer.wbar.shape), dtype=np.float32).reshape(
        layer.wbar.shape
    )

    layer.u.assign(u_value)
    layer.sig.assign(sigma_value)
    layer.wbar.assign(wbar_value)

    target_path = tmp_path / "spectral_conv_state.keras"
    model.save(target_path)

    restored = keras.models.load_model(target_path, safe_mode=False)
    restored_layer = restored.layers[1]

    np.testing.assert_allclose(restored_layer.u.numpy(), u_value)
    np.testing.assert_allclose(restored_layer.sig.numpy(), sigma_value)
    np.testing.assert_allclose(restored_layer.wbar.numpy(), wbar_value)
