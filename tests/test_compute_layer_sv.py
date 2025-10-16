import torch

from deel.lip.compute_layer_sv import compute_layer_sv
from deel.lip.layers.dense import SpectralDense


def test_compute_layer_sv_spectral_dense():
    layer = SpectralDense(4)
    x = torch.randn(2, 4)
    layer(x)
    sv_min, sv_max = compute_layer_sv(layer, input_shape=(1, 4))
    assert sv_min is not None and sv_max is not None
    assert abs(sv_max - 1.0) < 1e-2
