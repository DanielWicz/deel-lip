import torch

from deel.lip.normalizers import bjorck_normalization, reshaped_kernel_orthogonalization


def test_bjorck_normalization_returns_orthogonal_matrix():
    w = torch.randn(6, 6)
    w = w / torch.linalg.norm(w)
    w_bar = bjorck_normalization(w, eps=1e-4, beta=0.5, maxiter=10)
    gram = w_bar @ w_bar.T
    eye = torch.eye(gram.shape[0])
    assert torch.allclose(gram, eye, atol=1e-2)


def test_reshaped_kernel_orthogonalization_shapes():
    kernel = torch.randn(4, 3)
    w_bar, u, sigma = reshaped_kernel_orthogonalization(kernel, None, 1.0)
    assert w_bar.shape == kernel.shape
    assert u.shape[-1] == kernel.shape[-1]
    assert sigma.shape == (1, 1)
