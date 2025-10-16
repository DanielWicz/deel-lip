import torch

from deel.lip.layers.dense import SpectralDense
from deel.lip.model import Sequential


def test_sequential_forward_preserves_gradient_norm():
    layer1 = SpectralDense(3)
    layer2 = SpectralDense(3)
    model = Sequential(layer1, layer2, k_coef_lip=1.0)
    x = torch.randn(2, 3, requires_grad=True)
    y = model(x)
    loss = y.norm()
    loss.backward()
    assert torch.isfinite(x.grad).all()
