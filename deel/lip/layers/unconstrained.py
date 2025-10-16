# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III -
# All rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Unconstrained building blocks implemented with PyTorch.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..utils import _padding_circular
from .base_layer import Condensable
from .convolutional import _apply_same_padding
from .dense import _resolve_activation, _resolve_bias_initializer


def _to_2tuple(value: int | Sequence[int]) -> Tuple[int, int]:
    if isinstance(value, Sequence):
        value = tuple(value)
        if len(value) != 2:
            raise ValueError("Expected tuple of length 2.")
        return int(value[0]), int(value[1])
    return int(value), int(value)


class PadConv2D(nn.Module, Condensable):
    """
    Convolution layer with configurable padding modes beyond PyTorch defaults.
    This layer does not enforce Lipschitz constraints and primarily serves as a
    building block for constrained modules.
    """

    SUPPORTED_PADDING = {
        "same",
        "valid",
        "constant",
        "symmetric",
        "reflect",
        "circular",
        "replicate",
    }

    def __init__(
        self,
        filters: int,
        kernel_size: int | Sequence[int],
        strides: int | Sequence[int] = (1, 1),
        padding: str = "same",
        data_format: Optional[str] = "channels_last",
        dilation_rate: int | Sequence[int] = (1, 1),
        activation: Optional[Callable | str] = None,
        use_bias: bool = True,
        kernel_initializer: Optional[Callable] = None,
        bias_initializer: Callable | str = "zeros",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if padding.lower() not in self.SUPPORTED_PADDING:
            raise ValueError(f"Unsupported padding: {padding}")
        if data_format not in {"channels_last", "channels_first"}:
            raise ValueError("data_format must be 'channels_last' or 'channels_first'.")

        self.out_channels = int(filters)
        self.kernel_size = _to_2tuple(kernel_size)
        self.stride = _to_2tuple(strides)
        self.dilation = _to_2tuple(dilation_rate)
        self.padding = padding.lower()
        self.data_format = data_format
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = _resolve_bias_initializer(bias_initializer)
        self.activation_fn, self.activation_name = _resolve_activation(activation)
        self.padding_value = kwargs.pop("padding_value", 0.0)

        self.conv: Optional[nn.Conv2d] = None
        self.in_channels: Optional[int] = None
        self.built = False
        self._kwargs = kwargs

    def _maybe_build(self, x: Tensor) -> None:
        if self.built:
            return
        if self.data_format == "channels_last":
            in_channels = x.shape[-1]
        else:
            in_channels = x.shape[1]
        self.in_channels = in_channels

        device, dtype = x.device, x.dtype
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            bias=self.use_bias,
        ).to(device=device, dtype=dtype)

        if self.kernel_initializer is not None:
            weight = self.kernel_initializer(
                (*self.kernel_size, in_channels, self.out_channels),
                dtype=dtype,
                device=device,
            )
            weight = weight.permute(3, 2, 0, 1).contiguous()
            with torch.no_grad():
                self.conv.weight.copy_(weight)
        else:
            nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))
        if self.use_bias and self.conv.bias is not None:
            bias = self.bias_initializer((self.out_channels,), device=device, dtype=dtype)
            with torch.no_grad():
                self.conv.bias.copy_(bias)

        self.built = True

    def _apply_custom_padding(self, x: Tensor) -> Tensor:
        kh, kw = self.kernel_size
        pad_h, pad_w = kh // 2, kw // 2
        padding = (pad_w, pad_w, pad_h, pad_h)
        if self.padding == "constant":
            return F.pad(x, padding, mode="constant", value=self.padding_value)
        if self.padding == "reflect":
            return F.pad(x, padding, mode="reflect")
        if self.padding == "replicate":
            return F.pad(x, padding, mode="replicate")
        if self.padding == "symmetric":
            # Approximate symmetric padding via replicate padding which preserves boundary values.
            return F.pad(x, padding, mode="replicate")
        if self.padding == "circular":
            return _padding_circular(x, (pad_h, pad_w))
        raise ValueError(f"Unsupported padding mode: {self.padding}")

    def _pad_input(self, x: Tensor) -> Tensor:
        if self.padding == "valid":
            return x
        if self.padding == "same":
            return _apply_same_padding(x, self.kernel_size, self.stride, self.dilation)
        return self._apply_custom_padding(x)

    def forward(self, x: Tensor) -> Tensor:
        self._maybe_build(x)
        if self.data_format == "channels_last":
            x = x.permute(0, 3, 1, 2)
        x = self._pad_input(x)
        output = self.conv(x)
        if self.data_format == "channels_last":
            output = output.permute(0, 2, 3, 1)
        return self.activation_fn(output)

    def get_config(self) -> Dict[str, Any]:
        return {
            "filters": self.out_channels,
            "kernel_size": self.kernel_size,
            "strides": self.stride,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilation_rate": self.dilation,
            "activation": self.activation_name,
            "use_bias": self.use_bias,
        }

    def condense(self):
        return

    def vanilla_export(self):
        if not self.built:
            raise RuntimeError("Layer must be built before calling vanilla_export().")
        exported = PadConv2D(
            filters=self.out_channels,
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation,
            activation=self.activation_fn,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            **self._kwargs,
        )
        if self.data_format == "channels_last":
            dummy = torch.zeros(
                (1, 2, 2, self.in_channels),
                device=self.conv.weight.device,
                dtype=self.conv.weight.dtype,
            )
        else:
            dummy = torch.zeros(
                (1, self.in_channels, 2, 2),
                device=self.conv.weight.device,
                dtype=self.conv.weight.dtype,
            )
        exported._maybe_build(dummy)
        exported.load_state_dict(self.state_dict())
        return exported


__all__ = ["PadConv2D"]
