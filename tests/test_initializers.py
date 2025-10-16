import torch

from deel.lip.initializers import SpectralInitializer


def test_spectral_initializer_produces_orthogonal_rows():
    init = SpectralInitializer()
    weight = init((5, 5), dtype=torch.float32, device=torch.device("cpu"))
    gram = weight @ weight.t()
    eye = torch.eye(gram.shape[0])
    assert torch.allclose(gram, eye, atol=1e-3)
