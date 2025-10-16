# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier
# Toulouse III - All rights reserved. DEEL is a research program operated by
# IVADO, IRT Saint Exupéry, CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
PyTorch implementations of Lipschitz-respecting activation modules.
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor, nn

from .base_layer import LipschitzLayer


def _resolve_channel_axis(axis: int, ndim: int) -> int:
    if axis < 0:
        axis = ndim + axis
    if axis < 0 or axis >= ndim:
        raise ValueError("Invalid channel axis.")
    return axis


class _BaseActivation(nn.Module, LipschitzLayer):
    def __init__(self, k_coef_lip: float = 1.0) -> None:
        super().__init__()
        self.set_klip_factor(k_coef_lip)
        self._built = False

    def _ensure_built(self, input_shape: torch.Size) -> None:
        if not self._built:
            self._init_lip_coef(input_shape)
            self._built = True


class MaxMin(_BaseActivation):
    def __init__(self, data_format: str = "channels_last", k_coef_lip: float = 1.0):
        super().__init__(k_coef_lip=k_coef_lip)
        if data_format not in {"channels_last", "channels_first"}:
            raise ValueError("data_format must be 'channels_last' or 'channels_first'.")
        self.data_format = data_format
        self.channel_axis = -1 if data_format == "channels_last" else 1

    def forward(self, x: Tensor) -> Tensor:
        self._ensure_built(x.shape)
        pos = torch.relu(x)
        neg = torch.relu(-x)
        return torch.cat([pos, neg], dim=self.channel_axis) * self._get_coef()

    def get_config(self):
        return {"data_format": self.data_format, "k_coef_lip": self.k_coef_lip}


class GroupSort(_BaseActivation):
    def __init__(
        self,
        n: Optional[int] = None,
        data_format: str = "channels_last",
        k_coef_lip: float = 1.0,
    ):
        super().__init__(k_coef_lip=k_coef_lip)
        if data_format not in {"channels_last", "channels_first"}:
            raise ValueError("data_format must be 'channels_last' or 'channels_first'.")
        if data_format == "channels_first":
            raise RuntimeError("channels_first not implemented for GroupSort activation")
        self.data_format = data_format
        self.channel_axis = -1
        self.n = n
        self._groups = None

    def _build(self, input_shape: torch.Size) -> None:
        channel_axis = _resolve_channel_axis(self.channel_axis, len(input_shape))
        channels = input_shape[channel_axis]
        group_size = channels if (self.n is None or self.n > channels) else self.n
        if channels % group_size != 0:
            raise RuntimeError("Group size must divide the number of channels.")
        self.n = group_size
        self._groups = channels // self.n
        self.channel_axis = channel_axis
        self._built = True
        self._init_lip_coef(input_shape)

    def forward(self, x: Tensor) -> Tensor:
        if not self._built:
            self._build(x.shape)
        x_perm = torch.movedim(x, self.channel_axis, -1)
        new_shape = x_perm.shape[:-1] + (self._groups, self.n)
        grouped = x_perm.view(new_shape)
        if self.n == 2:
            a, b = grouped.unbind(-1)
            mins = torch.minimum(a, b)
            maxs = torch.maximum(a, b)
            sorted_group = torch.stack((mins, maxs), dim=-1)
        else:
            sorted_group, _ = torch.sort(grouped, dim=-1)
        sorted_flat = sorted_group.reshape(x_perm.shape)
        output = torch.movedim(sorted_flat, -1, self.channel_axis)
        return output * self._get_coef()

    def get_config(self):
        return {"n": self.n, "k_coef_lip": self.k_coef_lip, "data_format": self.data_format}


class GroupSort2(GroupSort):
    def __init__(self, **kwargs):
        kwargs["n"] = 2
        super().__init__(**kwargs)


class FullSort(GroupSort):
    def __init__(self, **kwargs):
        kwargs["n"] = None
        super().__init__(**kwargs)


class Householder(_BaseActivation):
    def __init__(
        self,
        data_format: str = "channels_last",
        k_coef_lip: float = 1.0,
        theta_initializer: Optional[Callable[[Tuple[int]], Tensor]] = None,
    ):
        super().__init__(k_coef_lip=k_coef_lip)
        if data_format != "channels_last":
            raise RuntimeError("Only 'channels_last' data format is supported")
        self.data_format = data_format
        self.channel_axis = -1
        self.theta_initializer = theta_initializer
        self.theta: Optional[nn.Parameter] = None

    def _build(self, input_shape: torch.Size, dtype: torch.dtype, device: torch.device):
        channel_axis = _resolve_channel_axis(self.channel_axis, len(input_shape))
        channels = input_shape[channel_axis]
        if channels % 2 != 0:
            raise RuntimeError("Number of channels must be divisible by 2 for Householder.")
        half_channels = channels // 2
        if self.theta_initializer is None:
            initial_theta = torch.full((half_channels,), math.pi / 2, dtype=dtype, device=device)
        else:
            initial_theta = self.theta_initializer((half_channels,))
            initial_theta = torch.as_tensor(initial_theta, dtype=dtype, device=device)
        self.theta = nn.Parameter(initial_theta)
        self.channel_axis = channel_axis
        self._built = True
        self._init_lip_coef(input_shape)

    def forward(self, x: Tensor) -> Tensor:
        if not self._built:
            self._build(x.shape, dtype=x.dtype, device=x.device)
        theta = self.theta
        x_perm = torch.movedim(x, self.channel_axis, -1)
        z1, z2 = torch.chunk(x_perm, 2, dim=-1)
        selector = (z1 * torch.sin(0.5 * theta)) - (z2 * torch.cos(0.5 * theta))
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        reflected_z1 = z1 * cos_theta + z2 * sin_theta
        reflected_z2 = z1 * sin_theta - z2 * cos_theta
        a = torch.where(selector <= 0, z1, reflected_z1)
        b = torch.where(selector <= 0, z2, reflected_z2)
        output = torch.cat([a, b], dim=-1)
        output = torch.movedim(output, -1, self.channel_axis)
        return output

    def get_config(self):
        return {
            "k_coef_lip": self.k_coef_lip,
            "data_format": self.data_format,
            "theta_initializer": self.theta_initializer,
        }


class _PReLUlip(nn.Module, LipschitzLayer):
    def __init__(self, num_parameters: int = 1, k_coef_lip: float = 1.0):
        super().__init__()
        self.set_klip_factor(k_coef_lip)
        self.prelu = nn.PReLU(num_parameters=num_parameters)

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            self.prelu.weight.clamp_(-self.k_coef_lip, self.k_coef_lip)
        return self.prelu(x)


def PReLUlip(k_coef_lip: float = 1.0, num_parameters: int = 1) -> _PReLUlip:
    """
    PReLU activation retaining Lipschitz constant by clamping the negative slope.
    """
    return _PReLUlip(num_parameters=num_parameters, k_coef_lip=k_coef_lip)


__all__ = [
    "MaxMin",
    "GroupSort",
    "GroupSort2",
    "FullSort",
    "PReLUlip",
    "Householder",
]
