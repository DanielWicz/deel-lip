import torch

from deel.lip.layers.dense import SpectralDense
from deel.lip.model import Model, Sequential, vanillaModel


class SimpleModel(Model):
    def __init__(self):
        super().__init__()
        self.net = Sequential(SpectralDense(3), k_coef_lip=1.0)

    def forward(self, x):
        return self.net(x)


def test_sequential_propagates_lipschitz_factor():
    seq = Sequential(SpectralDense(3), SpectralDense(3), k_coef_lip=1.0)
    for layer in seq:
        if hasattr(layer, "k_coef_lip"):
            assert layer.k_coef_lip <= 1.0


def test_vanilla_model_exports_condensable_layers():
    model = Sequential(SpectralDense(3))
    x = torch.randn(2, 3)
    y = model(x)
    vanilla = vanillaModel(model)
    assert isinstance(vanilla, Sequential)
    y_vanilla = vanilla(x)
    assert torch.allclose(y, y_vanilla, atol=1e-4)
