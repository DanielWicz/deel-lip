# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier
# Toulouse III - All rights reserved. DEEL is a research program operated by
# IVADO, IRT Saint Exupéry, CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Spectral initializer implemented with PyTorch.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Tuple

import torch
from torch import Tensor

from .normalizers import (
    DEFAULT_BETA_BJORCK,
    DEFAULT_EPS_BJORCK,
    DEFAULT_EPS_SPECTRAL,
    DEFAULT_MAXITER_BJORCK,
    DEFAULT_MAXITER_SPECTRAL,
    reshaped_kernel_orthogonalization,
)


_BASE_INITIALIZERS: Dict[str, Callable[[Tensor], Tensor]] = {
    "orthogonal": torch.nn.init.orthogonal_,
    "glorot_uniform": torch.nn.init.xavier_uniform_,
    "glorot_normal": torch.nn.init.xavier_normal_,
    "xavier_uniform": torch.nn.init.xavier_uniform_,
    "xavier_normal": torch.nn.init.xavier_normal_,
    "he_uniform": lambda tensor: torch.nn.init.kaiming_uniform_(tensor, a=math.sqrt(5)),
    "he_normal": lambda tensor: torch.nn.init.kaiming_normal_(tensor, a=math.sqrt(5)),
}


def _resolve_initializer(base_initializer: Callable | str) -> Callable[[Tensor], Tensor]:
    if callable(base_initializer):
        def wrapper(tensor: Tensor) -> Tensor:
            result = base_initializer(tensor)
            return result if isinstance(result, Tensor) else tensor

        return wrapper

    if isinstance(base_initializer, str):
        key = base_initializer.lower()
        if key not in _BASE_INITIALIZERS:
            raise ValueError(f"Unsupported base initializer '{base_initializer}'.")
        return _BASE_INITIALIZERS[key]

    raise TypeError("base_initializer must be a callable or a supported string.")


class SpectralInitializer:
    """
    Initialize a kernel to be 1-Lipschitz orthogonal using Björck normalisation.
    """

    def __init__(
        self,
        eps_spectral: float = DEFAULT_EPS_SPECTRAL,
        eps_bjorck: float | None = DEFAULT_EPS_BJORCK,
        beta_bjorck: float | None = DEFAULT_BETA_BJORCK,
        k_coef_lip: float = 1.0,
        base_initializer: Callable | str = "orthogonal",
        maxiter_spectral: int = DEFAULT_MAXITER_SPECTRAL,
        maxiter_bjorck: int = DEFAULT_MAXITER_BJORCK,
    ) -> None:
        self.eps_spectral = eps_spectral
        self.eps_bjorck = eps_bjorck
        self.beta_bjorck = beta_bjorck
        self.k_coef_lip = k_coef_lip
        self.maxiter_spectral = maxiter_spectral
        self.maxiter_bjorck = maxiter_bjorck
        self.base_initializer = _resolve_initializer(base_initializer)

    def __call__(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> Tensor:
        dtype = dtype or torch.float32
        device = device or torch.device("cpu")
        tensor = torch.empty(shape, dtype=dtype, device=device)
        tensor = self.base_initializer(tensor)
        tensor, _, _ = reshaped_kernel_orthogonalization(
            tensor,
            None,
            self.k_coef_lip,
            self.eps_spectral,
            self.eps_bjorck,
            self.beta_bjorck,
            self.maxiter_spectral,
            self.maxiter_bjorck,
        )
        return tensor

    def get_config(self):
        return {
            "eps_spectral": self.eps_spectral,
            "eps_bjorck": self.eps_bjorck,
            "beta_bjorck": self.beta_bjorck,
            "k_coef_lip": self.k_coef_lip,
            "maxiter_spectral": self.maxiter_spectral,
            "maxiter_bjorck": self.maxiter_bjorck,
        }


__all__ = ["SpectralInitializer"]
