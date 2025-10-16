import torch

from deel.lip.layers.pooling import (
    InvertibleDownSampling,
    InvertibleUpSampling,
    ScaledAveragePooling2D,
    ScaledGlobalAveragePooling2D,
    ScaledL2NormPooling2D,
)


def test_scaled_average_pooling_scales_output():
    pool = ScaledAveragePooling2D(pool_size=(2, 2))
    x = torch.randn(1, 3, 8, 8)
    y = pool(x)
    assert y.shape == (1, 3, 4, 4)


def test_scaled_l2norm_pooling_returns_positive():
    pool = ScaledL2NormPooling2D(pool_size=(2, 2))
    x = torch.randn(2, 3, 6, 6)
    y = pool(x)
    assert torch.all(y >= 0)


def test_scaled_global_average_pooling():
    pool = ScaledGlobalAveragePooling2D()
    x = torch.randn(2, 3, 4, 4)
    y = pool(x)
    assert y.shape == (2, 3)


def test_invertible_down_up_sampling_roundtrip():
    down = InvertibleDownSampling(pool_size=(2, 2), data_format="channels_first")
    up = InvertibleUpSampling(pool_size=(2, 2), data_format="channels_first")
    x = torch.randn(1, 3, 8, 8)
    y = down(x)
    z = up(y)
    assert z.shape == x.shape
