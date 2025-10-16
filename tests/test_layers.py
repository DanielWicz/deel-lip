import torch

from deel.lip.layers.convolutional import SpectralConv2D
from deel.lip.layers.dense import SpectralDense


def test_spectral_dense_forward_and_vanilla():
    layer = SpectralDense(4)
    x = torch.randn(8, 4)
    y = layer(x)
    assert y.shape == (8, 4)
    vanilla = layer.vanilla_export()
    y_vanilla = vanilla(x)
    assert torch.allclose(y, y_vanilla, atol=1e-4)


def test_spectral_conv2d_forward():
    layer = SpectralConv2D(
        in_channels=3,
        out_channels=3,
        kernel_size=3,
        padding="same",
    )
    x = torch.randn(2, 3, 16, 16)
    y = layer(x)
    assert y.shape == (2, 3, 16, 16)
