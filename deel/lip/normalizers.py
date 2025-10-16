# "Copyright" Daniel Wiczew, NEBULA IRT Antoine de Saint Exupéry et Université Paul Sabatier
# Toulouse III - All rights reserved. DEEL is a research program operated by
# IVADO, IRT Saint Exupéry, CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Spectral and Björck normalisation utilities implemented with PyTorch tensors.
"""
from __future__ import annotations

from typing import Callable, Iterable, Tuple

import torch
from torch import Tensor

DEFAULT_BETA_BJORCK = 0.5
DEFAULT_EPS_SPECTRAL = 1e-3
DEFAULT_EPS_BJORCK = 1e-3
DEFAULT_MAXITER_BJORCK = 15
DEFAULT_MAXITER_SPECTRAL = 10
SWAP_MEMORY = True
STOP_GRAD_SPECTRAL = True


def set_swap_memory(value: bool):
    """Kept for API compatibility. PyTorch does not expose swap-memory control."""
    global SWAP_MEMORY
    SWAP_MEMORY = value


def set_stop_grad_spectral(value: bool):
    """Toggle whether power iteration should detach its result from the graph."""
    global STOP_GRAD_SPECTRAL
    STOP_GRAD_SPECTRAL = value


def _check_RKO_params(eps_spectral, eps_bjorck, beta_bjorck):
    """Assert that RKO hyper-parameters are supported values."""
    if eps_spectral <= 0:
        raise ValueError("eps_spectral has to be > 0")
    if (eps_bjorck is not None) and (eps_bjorck <= 0.0):
        raise ValueError("eps_bjorck must be > 0")
    if (beta_bjorck is not None) and not (0.0 < beta_bjorck <= 0.5):
        raise ValueError("beta_bjorck must be in ]0, 0.5]")


def reshaped_kernel_orthogonalization(
    kernel: Tensor,
    u: Tensor | None,
    adjustment_coef: float,
    eps_spectral: float = DEFAULT_EPS_SPECTRAL,
    eps_bjorck: float | None = DEFAULT_EPS_BJORCK,
    beta: float | None = DEFAULT_BETA_BJORCK,
    maxiter_spectral: int = DEFAULT_MAXITER_SPECTRAL,
    maxiter_bjorck: int = DEFAULT_MAXITER_BJORCK,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Apply spectral normalisation followed by Björck normalisation on the input kernel.
    """
    original_shape = kernel.shape
    w_reshaped = kernel.reshape(-1, original_shape[-1])
    w_bar, u, sigma = spectral_normalization(
        w_reshaped, u, eps=eps_spectral, maxiter=maxiter_spectral
    )
    if (eps_bjorck is not None) and (beta is not None):
        w_bar = bjorck_normalization(
            w_bar, eps=eps_bjorck, beta=beta, maxiter=maxiter_bjorck
        )
    w_bar = (w_bar * adjustment_coef).reshape(original_shape)
    return w_bar, u, sigma


def bjorck_normalization(
    w: Tensor,
    eps: float = DEFAULT_EPS_BJORCK,
    beta: float = DEFAULT_BETA_BJORCK,
    maxiter: int = DEFAULT_MAXITER_BJORCK,
) -> Tensor:
    """
    Apply the Björck iterative normalisation.
    """
    if eps is None or beta is None:
        return w

    u, _, vh = torch.linalg.svd(w, full_matrices=False)
    return u @ vh


def _normalise_vector(u: Tensor, axis: int | Iterable[int] | None) -> Tensor:
    if axis is None:
        norm = torch.linalg.norm(u)
        return u / (norm + 1e-12)
    if isinstance(axis, int):
        axis = (axis,)
    norm = torch.linalg.vector_norm(u, dim=tuple(axis), keepdim=True)
    return u / (norm + 1e-12)


def _power_iteration(
    linear_operator: Callable[[Tensor], Tensor],
    adjoint_operator: Callable[[Tensor], Tensor],
    u: Tensor,
    eps: float = DEFAULT_EPS_SPECTRAL,
    maxiter: int = DEFAULT_MAXITER_SPECTRAL,
    axis: int | Iterable[int] | None = None,
) -> Tensor:
    """
    Power iteration algorithm to estimate the largest singular vector.
    """
    u = _normalise_vector(u, axis)
    old_u = u + 2 * eps

    for _ in range(maxiter):
        if torch.linalg.norm(u - old_u) < eps:
            break
        old_u = u
        v = linear_operator(u)
        u = adjoint_operator(v)
        u = _normalise_vector(u, axis)

    if STOP_GRAD_SPECTRAL:
        u = u.detach()
    return u


def spectral_normalization(
    kernel: Tensor,
    u: Tensor | None,
    eps: float = DEFAULT_EPS_SPECTRAL,
    maxiter: int = DEFAULT_MAXITER_SPECTRAL,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Normalise a matrix so that its largest singular value equals 1.
    """
    if u is None:
        device, dtype = kernel.device, kernel.dtype
        u = torch.rand((1, kernel.shape[-1]), device=device, dtype=dtype)

    def linear_op(vec: Tensor) -> Tensor:
        return vec @ kernel.transpose(-1, -2)

    def adjoint_op(vec: Tensor) -> Tensor:
        return vec @ kernel

    u = _power_iteration(linear_op, adjoint_op, u, eps=eps, maxiter=maxiter)
    sigma = torch.linalg.norm(linear_op(u), dim=-1, keepdim=True)
    normalized_kernel = kernel / (sigma + eps)
    sigma_normalized = sigma / (sigma + eps)
    return normalized_kernel, u, sigma_normalized


def get_conv_operators(*args, **kwargs):  # pragma: no cover - placeholder
    raise NotImplementedError(
        "Convolutional spectral normalisation is not implemented yet for PyTorch."
    )


def spectral_normalization_conv(*args, **kwargs):  # pragma: no cover - placeholder
    raise NotImplementedError(
        "Convolutional spectral normalisation is not implemented yet for PyTorch."
    )


spectral_normalization_conv.unavailable_class = True
get_conv_operators.unavailable_class = True


__all__ = [
    "DEFAULT_BETA_BJORCK",
    "DEFAULT_EPS_SPECTRAL",
    "DEFAULT_EPS_BJORCK",
    "DEFAULT_MAXITER_BJORCK",
    "DEFAULT_MAXITER_SPECTRAL",
    "set_swap_memory",
    "set_stop_grad_spectral",
    "_check_RKO_params",
    "reshaped_kernel_orthogonalization",
    "bjorck_normalization",
    "spectral_normalization",
    "spectral_normalization_conv",
    "get_conv_operators",
]
