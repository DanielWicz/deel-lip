# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III -
# All rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Constraint helpers implemented for PyTorch modules.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from .normalizers import (
    DEFAULT_BETA_BJORCK,
    DEFAULT_EPS_BJORCK,
    DEFAULT_EPS_SPECTRAL,
    reshaped_kernel_orthogonalization,
)


@dataclass
class WeightClipConstraint:
    """Clip weights to lie in [-c, c]."""

    c: float = 2.0

    def __call__(self, w: Tensor) -> Tensor:
        return torch.clamp(w, -self.c, self.c)

    def apply_(self, w: Tensor) -> Tensor:
        with torch.no_grad():
            w.clamp_(-self.c, self.c)
        return w


@dataclass
class AutoWeightClipConstraint:
    """Clip weights automatically based on the tensor size."""

    scale: float = 1.0

    def __call__(self, w: Tensor) -> Tensor:
        size = w.numel()
        c = 1.0 / (torch.sqrt(torch.tensor(size, dtype=w.dtype, device=w.device)) * self.scale)
        return torch.clamp(w, -c, c)

    def apply_(self, w: Tensor) -> Tensor:
        with torch.no_grad():
            w.copy_(self(w))
        return w


@dataclass
class FrobeniusConstraint:
    """Project weights onto the unit Frobenius norm sphere."""

    eps: float = 1e-7

    def __call__(self, w: Tensor) -> Tensor:
        norm = torch.linalg.norm(w)
        return w / (norm + self.eps)

    def apply_(self, w: Tensor) -> Tensor:
        with torch.no_grad():
            w.copy_(self(w))
        return w


class SpectralConstraint:
    """
    Enforce all singular values of the weight tensor to equal a target Lipschitz
    coefficient using spectral and Björck normalisation.
    """

    def __init__(
        self,
        k_coef_lip: float = 1.0,
        eps_spectral: float = DEFAULT_EPS_SPECTRAL,
        eps_bjorck: Optional[float] = DEFAULT_EPS_BJORCK,
        beta_bjorck: Optional[float] = DEFAULT_BETA_BJORCK,
        u: Optional[Tensor] = None,
    ) -> None:
        self.k_coef_lip = k_coef_lip
        self.eps_spectral = eps_spectral
        self.eps_bjorck = eps_bjorck
        self.beta_bjorck = beta_bjorck
        self.u = u

    def __call__(self, w: Tensor) -> Tensor:
        wbar, self.u, _ = reshaped_kernel_orthogonalization(
            w,
            self.u,
            self.k_coef_lip,
            self.eps_spectral,
            self.eps_bjorck,
            self.beta_bjorck,
        )
        return wbar

    def apply_(self, w: Tensor) -> Tensor:
        with torch.no_grad():
            w.copy_(self(w))
        return w

    def get_config(self):
        return {
            "k_coef_lip": self.k_coef_lip,
            "eps_spectral": self.eps_spectral,
            "eps_bjorck": self.eps_bjorck,
            "beta_bjorck": self.beta_bjorck,
        }


__all__ = [
    "WeightClipConstraint",
    "AutoWeightClipConstraint",
    "FrobeniusConstraint",
    "SpectralConstraint",
]
