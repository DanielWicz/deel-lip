import torch

from deel.lip.normalizers import spectral_normalization


def test_spectral_normalization_scales_matrix():
    kernel = torch.randn(5, 3)
    w_bar, u, sigma = spectral_normalization(kernel, None)
    assert w_bar.shape == kernel.shape
    assert u.shape[-1] == kernel.shape[-1]
    assert abs(sigma.item() - 1.0) < 1e-1
