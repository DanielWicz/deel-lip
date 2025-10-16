# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier
# Toulouse III - All rights reserved. DEEL is a research program operated by
# IVADO, IRT Saint Exupéry, CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
PyTorch implementation of Lipschitz-constrained convolutional layers.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..initializers import SpectralInitializer
from ..normalizers import (
    DEFAULT_BETA_BJORCK,
    DEFAULT_EPS_BJORCK,
    DEFAULT_EPS_SPECTRAL,
    DEFAULT_MAXITER_BJORCK,
    DEFAULT_MAXITER_SPECTRAL,
    _check_RKO_params,
    reshaped_kernel_orthogonalization,
)
from .base_layer import Condensable, LipschitzLayer
from .dense import _resolve_activation


def _to_2tuple(value: int | Sequence[int]) -> Tuple[int, int]:
    if isinstance(value, Iterable):
        value = tuple(value)
        if len(value) != 2:
            raise ValueError("Value must have length 2.")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _compute_conv_lip_factor(
    kernel_size: Tuple[int, int],
    strides: Tuple[int, int],
    input_shape: Sequence[int],
) -> float:
    stride = strides[0] * strides[1]
    kh, kw = kernel_size
    kh_div2 = (kh - 1) / 2
    kw_div2 = (kw - 1) / 2
    height, width = input_shape[-2], input_shape[-1]

    if stride == 1:
        numerator = width * height
        denominator = (kh * height - kh_div2 * (kh_div2 + 1)) * (
            kw * width - kw_div2 * (kw_div2 + 1)
        )
        return float(math.sqrt(numerator / denominator))
    return float(
        math.sqrt(
            1.0 / (math.ceil(kh / strides[0]) * math.ceil(kw / strides[1]))
        )
    )


def _apply_same_padding(
    inputs: Tensor,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    dilation: Tuple[int, int],
) -> Tensor:
    height, width = inputs.shape[-2:]
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation
    kernel_h, kernel_w = kernel_size

    out_height = math.ceil(height / stride_h)
    out_width = math.ceil(width / stride_w)

    pad_h_total = max(
        (out_height - 1) * stride_h + dilation_h * (kernel_h - 1) + 1 - height, 0
    )
    pad_w_total = max(
        (out_width - 1) * stride_w + dilation_w * (kernel_w - 1) + 1 - width, 0
    )

    pad_top = pad_h_total // 2
    pad_bottom = pad_h_total - pad_top
    pad_left = pad_w_total // 2
    pad_right = pad_w_total - pad_left

    if pad_h_total == 0 and pad_w_total == 0:
        return inputs

    return F.pad(inputs, (pad_left, pad_right, pad_top, pad_bottom))


def _initialize_conv_weight(
    in_channels: int,
    out_channels: int,
    kernel_size: Tuple[int, int],
    initializer: Optional[Callable],
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    init = initializer or SpectralInitializer()
    kernel_shape = (*kernel_size, in_channels, out_channels)
    weight_tf = init(kernel_shape, dtype=dtype, device=device)
    return weight_tf.permute(3, 2, 0, 1).contiguous()


class _Conv2dBase(nn.Module, LipschitzLayer, Condensable):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: str = "same",
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        activation: Optional[Callable | str] = None,
        kernel_initializer: Optional[Callable] = None,
        bias_initializer: Callable | str = "zeros",
        k_coef_lip: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        if padding_mode != "zeros":
            raise ValueError("Only zero padding mode is supported.")
        if padding not in {"same", "valid"}:
            raise ValueError("Only 'same' and 'valid' padding are supported.")
        if groups != 1:
            raise ValueError("Grouped convolutions are not supported.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _to_2tuple(kernel_size)
        self.stride = _to_2tuple(stride)
        self.padding = padding
        self.dilation = _to_2tuple(dilation)
        self.groups = groups
        self.use_bias = bias
        self.padding_mode = padding_mode
        self.activation_fn, self.activation_name = _resolve_activation(activation)
        self.original_activation = activation
        self._kwargs = kwargs

        self.weight = nn.Parameter(
            _initialize_conv_weight(
                in_channels,
                out_channels,
                self.kernel_size,
                kernel_initializer,
                dtype=torch.float32,
                device=torch.device("cpu"),
            )
        )
        if bias:
            bias_init = bias_initializer
            if isinstance(bias_init, str):
                if bias_init == "zeros":
                    bias_tensor = torch.zeros(out_channels)
                elif bias_init == "ones":
                    bias_tensor = torch.ones(out_channels)
                else:
                    raise ValueError(f"Unsupported bias initializer '{bias_init}'.")
            else:
                tensor = torch.empty(out_channels)
                result = bias_init(tensor)
                bias_tensor = result if isinstance(result, Tensor) else tensor
            self.bias = nn.Parameter(bias_tensor)
        else:
            self.register_parameter("bias", None)

        self.set_klip_factor(k_coef_lip)
        self.register_buffer("wbar", self.weight.detach().clone())
        self.coef_initialized = False

    def _ensure_device_dtype(self, device: torch.device, dtype: torch.dtype) -> None:
        if self.weight.device != device or self.weight.dtype != dtype:
            self.weight.data = self.weight.data.to(device=device, dtype=dtype)
            if self.bias is not None:
                self.bias.data = self.bias.data.to(device=device, dtype=dtype)
            self.register_buffer("wbar", self.wbar.to(device=device, dtype=dtype))

    def _ensure_lip_coef(self, input_tensor: Tensor) -> None:
        if not self.coef_initialized:
            self._init_lip_coef(input_tensor.shape)
            self.coef_initialized = True

    def get_config(self) -> Dict[str, object]:
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "dilation": self.dilation,
            "groups": self.groups,
            "use_bias": self.use_bias,
            "activation": self.activation_name,
            "k_coef_lip": self.k_coef_lip,
            **self._kwargs,
        }

    def _compute_weight(self, training: bool) -> Tensor:
        raise NotImplementedError

    def _apply_convolution(self, inputs: Tensor, weight: Tensor) -> Tensor:
        if self.padding == "same":
            inputs = _apply_same_padding(inputs, self.kernel_size, self.stride, self.dilation)
            padding = (0, 0)
        else:
            padding = tuple(0 for _ in self.kernel_size)
        return F.conv2d(
            inputs,
            weight,
            bias=None,
            stride=self.stride,
            padding=padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def condense(self) -> None:
        with torch.no_grad():
            weight = self._compute_weight(training=True)
            self.weight.copy_(weight)
            self.wbar.copy_(weight)

    def vanilla_export(self) -> nn.Conv2d:
        class VanillaConv2D(nn.Module):
            def __init__(self, outer: "_Conv2dBase"):
                super().__init__()
                self.padding = outer.padding
                self.kernel_size = outer.kernel_size
                self.stride = outer.stride
                self.dilation = outer.dilation
                self.groups = outer.groups
                self.conv = nn.Conv2d(
                    outer.in_channels,
                    outer.out_channels,
                    outer.kernel_size,
                    stride=outer.stride,
                    padding=0,
                    dilation=outer.dilation,
                    groups=outer.groups,
                    bias=outer.use_bias,
                )

            def forward(self, x: Tensor) -> Tensor:
                if self.padding == "same":
                    x = _apply_same_padding(x, self.kernel_size, self.stride, self.dilation)
                return self.conv(x)

        conv = VanillaConv2D(self)
        with torch.no_grad():
            weight = self._compute_weight(training=True)
            conv.conv.weight.copy_(weight)
            if self.use_bias and self.bias is not None:
                conv.conv.bias.copy_(self.bias)
        return conv

    def forward(self, inputs: Tensor) -> Tensor:
        self._ensure_device_dtype(inputs.device, inputs.dtype)
        self._ensure_lip_coef(inputs)
        weight = self._compute_weight(training=self.training)
        outputs = self._apply_convolution(inputs, weight)
        if self.bias is not None:
            outputs = outputs + self.bias.view(1, -1, 1, 1)
        return self.activation_fn(outputs)


class SpectralConv2D(_Conv2dBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: str = "same",
        dilation: int | Sequence[int] = 1,
        activation: Optional[Callable | str] = None,
        use_bias: bool = True,
        kernel_initializer: Optional[Callable] = None,
        bias_initializer: Callable | str = "zeros",
        k_coef_lip: float = 1.0,
        eps_spectral: float = DEFAULT_EPS_SPECTRAL,
        eps_bjorck: float | None = DEFAULT_EPS_BJORCK,
        beta_bjorck: float | None = DEFAULT_BETA_BJORCK,
        maxiter_spectral: int = DEFAULT_MAXITER_SPECTRAL,
        maxiter_bjorck: int = DEFAULT_MAXITER_BJORCK,
        **kwargs,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            k_coef_lip=k_coef_lip,
            **kwargs,
        )

        _check_RKO_params(eps_spectral, eps_bjorck, beta_bjorck)
        self.eps_spectral = eps_spectral
        self.eps_bjorck = eps_bjorck
        self.beta_bjorck = beta_bjorck
        self.maxiter_spectral = maxiter_spectral
        self.maxiter_bjorck = maxiter_bjorck

        self.register_buffer("u", torch.randn((1, out_channels)))
        self.register_buffer("sigma", torch.ones((1, 1)))

    def _compute_lip_coef(self, input_shape=None):
        return _compute_conv_lip_factor(self.kernel_size, self.stride, input_shape)

    def _compute_weight(self, training: bool) -> Tensor:
        if not training:
            return self.wbar

        kernel_tf = self.weight.permute(2, 3, 1, 0)
        wbar_tf, u, sigma = reshaped_kernel_orthogonalization(
            kernel_tf,
            self.u,
            self._get_coef(),
            self.eps_spectral,
            self.eps_bjorck,
            self.beta_bjorck,
            self.maxiter_spectral,
            self.maxiter_bjorck,
        )
        wbar = wbar_tf.permute(3, 2, 0, 1).contiguous()
        with torch.no_grad():
            self.wbar.copy_(wbar)
            self.u.copy_(u)
            self.sigma.copy_(sigma)
        return wbar

    def _ensure_device_dtype(self, device: torch.device, dtype: torch.dtype) -> None:
        super()._ensure_device_dtype(device, dtype)
        self.register_buffer("u", self.u.to(device=device, dtype=dtype))
        self.register_buffer("sigma", self.sigma.to(device=device, dtype=dtype))

    def get_config(self) -> Dict[str, object]:
        config = {
            "eps_spectral": self.eps_spectral,
            "eps_bjorck": self.eps_bjorck,
            "beta_bjorck": self.beta_bjorck,
            "maxiter_spectral": self.maxiter_spectral,
            "maxiter_bjorck": self.maxiter_bjorck,
        }
        base = super().get_config()
        return {**base, **config}


class FrobeniusConv2D(_Conv2dBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: str = "same",
        dilation: int | Sequence[int] = 1,
        activation: Optional[Callable | str] = None,
        use_bias: bool = True,
        kernel_initializer: Optional[Callable] = None,
        bias_initializer: Callable | str = "zeros",
        k_coef_lip: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            k_coef_lip=k_coef_lip,
            **kwargs,
        )

    def _compute_lip_coef(self, input_shape=None):
        return _compute_conv_lip_factor(self.kernel_size, self.stride, input_shape)

    def _compute_weight(self, training: bool) -> Tensor:
        if not training:
            return self.wbar
        norm = torch.linalg.norm(self.weight)
        wbar = self.weight / (norm + 1e-12) * self._get_coef()
        with torch.no_grad():
            self.wbar.copy_(wbar)
        return wbar


class SpectralConv2DTranspose(SpectralConv2D):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: str = "same",
        activation: Optional[Callable | str] = None,
        use_bias: bool = True,
        kernel_initializer: Optional[Callable] = None,
        bias_initializer: Callable | str = "zeros",
        k_coef_lip: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            k_coef_lip=k_coef_lip,
            **kwargs,
        )

    def _apply_convolution(self, inputs: Tensor, weight: Tensor) -> Tensor:
        if self.padding == "same":
            output_padding = 0
        else:
            output_padding = 0
        return F.conv_transpose2d(
            inputs,
            weight,
            bias=None,
            stride=self.stride,
            padding=0,
            output_padding=output_padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def vanilla_export(self) -> nn.ConvTranspose2d:
        class VanillaConvTranspose2D(nn.Module):
            def __init__(self, outer: "SpectralConv2DTranspose"):
                super().__init__()
                self.stride = outer.stride
                self.dilation = outer.dilation
                self.groups = outer.groups
                self.conv = nn.ConvTranspose2d(
                    outer.in_channels,
                    outer.out_channels,
                    outer.kernel_size,
                    stride=outer.stride,
                    padding=0,
                    dilation=outer.dilation,
                    groups=outer.groups,
                    bias=outer.use_bias,
                )

            def forward(self, x: Tensor) -> Tensor:
                return self.conv(x)

        conv = VanillaConvTranspose2D(self)
        with torch.no_grad():
            weight = self._compute_weight(training=True)
            conv.conv.weight.copy_(weight)
            if self.use_bias and self.bias is not None:
                conv.conv.bias.copy_(self.bias)
        return conv


class FrobeniusConv2DTranspose(FrobeniusConv2D):
    def _apply_convolution(self, inputs: Tensor, weight: Tensor) -> Tensor:
        return F.conv_transpose2d(
            inputs,
            weight,
            bias=None,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            groups=self.groups,
        )

    def vanilla_export(self) -> nn.ConvTranspose2d:
        class VanillaConvTranspose2D(nn.Module):
            def __init__(self, outer: "FrobeniusConv2DTranspose"):
                super().__init__()
                self.conv = nn.ConvTranspose2d(
                    outer.in_channels,
                    outer.out_channels,
                    outer.kernel_size,
                    stride=outer.stride,
                    padding=0,
                    dilation=outer.dilation,
                    groups=outer.groups,
                    bias=outer.use_bias,
                )

            def forward(self, x: Tensor) -> Tensor:
                return self.conv(x)

        conv = VanillaConvTranspose2D(self)
        with torch.no_grad():
            weight = self._compute_weight(training=True)
            conv.conv.weight.copy_(weight)
            if self.use_bias and self.bias is not None:
                conv.conv.bias.copy_(self.bias)
        return conv


__all__ = [
    "SpectralConv2D",
    "FrobeniusConv2D",
    "SpectralConv2DTranspose",
    "FrobeniusConv2DTranspose",
]
