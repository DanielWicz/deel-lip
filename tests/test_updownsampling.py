import torch

from deel.lip.layers.pooling import InvertibleDownSampling, InvertibleUpSampling


def test_down_up_sampling_inverse():
    down = InvertibleDownSampling(pool_size=(2, 2), data_format="channels_last")
    up = InvertibleUpSampling(pool_size=(2, 2), data_format="channels_last")
    x = torch.randn(1, 8, 8, 4)
    y = down(x)
    z = up(y)
    assert z.shape == x.shape
