import torch

from deel.lip.callbacks import CondenseCallback
from deel.lip.layers.dense import SpectralDense
from deel.lip.model import Sequential


def test_condense_callback_updates_weights():
    model = Sequential(SpectralDense(3))
    x = torch.randn(4, 3)
    y = model(x)
    callback = CondenseCallback(on_batch=True, on_epoch=False)
    callback.set_model(model)
    callback.on_train_batch_end(0)
    with torch.no_grad():
        w = model[0].kernel
        wbar = model[0].wbar
        assert torch.allclose(w, wbar, atol=1e-5)
