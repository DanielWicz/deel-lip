import torch

from deel.lip.layers.unconstrained import PadConv2D


def test_pad_conv2d_supports_same_padding():
    layer = PadConv2D(
        filters=2,
        kernel_size=(3, 3),
        padding="same",
        data_format="channels_first",
    )
    x = torch.randn(1, 2, 16, 16)
    y = layer(x)
    assert y.shape == x.shape


def test_pad_conv2d_circular_padding():
    layer = PadConv2D(
        filters=2,
        kernel_size=(3, 3),
        padding="circular",
        data_format="channels_first",
    )
    x = torch.randn(1, 2, 8, 8)
    y = layer(x)
    assert y.shape == (1, 2, 8, 8)
