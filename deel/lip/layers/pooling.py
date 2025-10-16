# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier
# Toulouse III - All rights reserved. DEEL is a research program operated by
# IVADO, IRT Saint Exupéry, CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
PyTorch implementation of Lipschitz-constrained pooling layers.
"""
from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .base_layer import LipschitzLayer


def _to_2tuple(value: int | Sequence[int]) -> Tuple[int, int]:
    if isinstance(value, Sequence):
        value = tuple(value)
        if len(value) != 2:
            raise ValueError("pool_size must have length 2.")
        return int(value[0]), int(value[1])
    return int(value), int(value)


class _SafeSqrt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor: Tensor, eps: float):
        ctx.save_for_backward(input_tensor)
        ctx.eps = eps
        return torch.sqrt(input_tensor)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (saved_input,) = ctx.saved_tensors
        sqrt_input = torch.sqrt(saved_input + 1e-12)
        denom = 2.0 * (sqrt_input + ctx.eps)
        return grad_output / denom, None


class ScaledAveragePooling2D(nn.Module, LipschitzLayer):
    def __init__(
        self,
        pool_size: int | Sequence[int] = (2, 2),
        stride: int | Sequence[int] | None = None,
        padding: str = "valid",
        k_coef_lip: float = 1.0,
    ) -> None:
        super().__init__()
        self.pool_size = _to_2tuple(pool_size)
        stride = stride if stride is not None else self.pool_size
        if _to_2tuple(stride) != self.pool_size:
            raise ValueError("stride must be equal to pool_size.")
        if padding != "valid":
            raise ValueError("Only 'valid' padding is supported.")
        self.stride = self.pool_size
        self.set_klip_factor(k_coef_lip)
        self.coef_initialized = False

    def _ensure_lip(self, x: Tensor) -> None:
        if not self.coef_initialized:
            self._init_lip_coef(x.shape)
            self.coef_initialized = True

    def _compute_lip_coef(self, input_shape=None):
        return math.sqrt(self.pool_size[0] * self.pool_size[1])

    def forward(self, x: Tensor) -> Tensor:
        self._ensure_lip(x)
        pooled = F.avg_pool2d(x, kernel_size=self.pool_size, stride=self.stride)
        return pooled * self._get_coef()

    def get_config(self) -> Dict[str, object]:
        return {"pool_size": self.pool_size, "k_coef_lip": self.k_coef_lip}


class ScaledL2NormPooling2D(nn.Module, LipschitzLayer):
    def __init__(
        self,
        pool_size: int | Sequence[int] = (2, 2),
        stride: int | Sequence[int] | None = None,
        padding: str = "valid",
        k_coef_lip: float = 1.0,
        eps_grad_sqrt: float = 1e-6,
    ) -> None:
        super().__init__()
        self.pool_size = _to_2tuple(pool_size)
        stride = stride if stride is not None else self.pool_size
        if _to_2tuple(stride) != self.pool_size:
            raise ValueError("stride must be equal to pool_size.")
        if padding != "valid":
            raise ValueError("Only 'valid' padding is supported.")
        if eps_grad_sqrt < 0.0:
            raise ValueError("eps_grad_sqrt must be positive.")
        self.stride = self.pool_size
        self.eps_grad_sqrt = eps_grad_sqrt
        self.set_klip_factor(k_coef_lip)
        self.coef_initialized = False

    def _ensure_lip(self, x: Tensor) -> None:
        if not self.coef_initialized:
            self._init_lip_coef(x.shape)
            self.coef_initialized = True

    def _compute_lip_coef(self, input_shape=None):
        return math.sqrt(self.pool_size[0] * self.pool_size[1])

    def forward(self, x: Tensor) -> Tensor:
        self._ensure_lip(x)
        squared = F.avg_pool2d(x.pow(2), kernel_size=self.pool_size, stride=self.stride)
        norm = _SafeSqrt.apply(squared, self.eps_grad_sqrt)
        return norm * self._get_coef()

    def get_config(self) -> Dict[str, object]:
        return {
            "pool_size": self.pool_size,
            "k_coef_lip": self.k_coef_lip,
            "eps_grad_sqrt": self.eps_grad_sqrt,
        }


class ScaledGlobalAveragePooling2D(nn.Module, LipschitzLayer):
    def __init__(self, k_coef_lip: float = 1.0) -> None:
        super().__init__()
        self.set_klip_factor(k_coef_lip)

    def _compute_lip_coef(self, input_shape=None):
        height, width = input_shape[-2], input_shape[-1]
        return math.sqrt(height * width)

    def forward(self, x: Tensor) -> Tensor:
        self._init_lip_coef(x.shape)
        pooled = F.adaptive_avg_pool2d(x, output_size=(1, 1)).flatten(1)
        return pooled * self._get_coef()

    def get_config(self) -> Dict[str, object]:
        return {"k_coef_lip": self.k_coef_lip}


class ScaledGlobalL2NormPooling2D(nn.Module, LipschitzLayer):
    def __init__(self, k_coef_lip: float = 1.0, eps_grad_sqrt: float = 1e-6) -> None:
        super().__init__()
        if eps_grad_sqrt < 0.0:
            raise ValueError("eps_grad_sqrt must be positive.")
        self.set_klip_factor(k_coef_lip)
        self.eps_grad_sqrt = eps_grad_sqrt

    def _compute_lip_coef(self, input_shape=None):
        return 1.0

    def forward(self, x: Tensor) -> Tensor:
        self._init_lip_coef(x.shape)
        summed = x.pow(2).sum(dim=(-2, -1))
        norm = _SafeSqrt.apply(summed, self.eps_grad_sqrt)
        return norm * self._get_coef()

    def get_config(self) -> Dict[str, object]:
        return {"k_coef_lip": self.k_coef_lip, "eps_grad_sqrt": self.eps_grad_sqrt}


class InvertibleDownSampling(nn.Module):
    def __init__(self, pool_size: int | Sequence[int], data_format: str = "channels_last") -> None:
        super().__init__()
        self.pool_size = _to_2tuple(pool_size)
        if data_format not in {"channels_last", "channels_first"}:
            raise ValueError("data_format must be 'channels_last' or 'channels_first'.")
        self.data_format = data_format

    def forward(self, x: Tensor) -> Tensor:
        if self.data_format == "channels_last":
            x = x.permute(0, 3, 1, 2)
        n, c, h, w = x.shape
        ph, pw = self.pool_size
        if h % ph != 0 or w % pw != 0:
            raise ValueError("Input spatial dimensions must be divisible by pool_size.")
        x = x.reshape(n, c, h // ph, ph, w // pw, pw)
        x = x.permute(0, 1, 3, 5, 2, 4).reshape(n, c * ph * pw, h // ph, w // pw)
        if self.data_format == "channels_last":
            x = x.permute(0, 2, 3, 1)
        return x

    def get_config(self) -> Dict[str, object]:
        return {"pool_size": self.pool_size, "data_format": self.data_format}


class InvertibleUpSampling(nn.Module):
    def __init__(self, pool_size: int | Sequence[int], data_format: str = "channels_last") -> None:
        super().__init__()
        self.pool_size = _to_2tuple(pool_size)
        if data_format not in {"channels_last", "channels_first"}:
            raise ValueError("data_format must be 'channels_last' or 'channels_first'.")
        self.data_format = data_format

    def forward(self, x: Tensor) -> Tensor:
        if self.data_format == "channels_last":
            x = x.permute(0, 3, 1, 2)
        n, c, h, w = x.shape
        ph, pw = self.pool_size
        if c % (ph * pw) != 0:
            raise ValueError("Number of channels must be divisible by the product of pool_size.")
        c_out = c // (ph * pw)
        x = x.reshape(n, c_out, ph, pw, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).reshape(n, c_out, h * ph, w * pw)
        if self.data_format == "channels_last":
            x = x.permute(0, 2, 3, 1)
        return x

    def get_config(self) -> Dict[str, object]:
        return {"pool_size": self.pool_size, "data_format": self.data_format}


__all__ = [
    "ScaledAveragePooling2D",
    "ScaledL2NormPooling2D",
    "ScaledGlobalAveragePooling2D",
    "ScaledGlobalL2NormPooling2D",
    "InvertibleDownSampling",
    "InvertibleUpSampling",
]
