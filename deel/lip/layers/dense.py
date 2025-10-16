# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier
# Toulouse III - All rights reserved. DEEL is a research program operated by
# IVADO, IRT Saint Exupéry, CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
PyTorch implementation of Lipschitz constrained dense layers.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
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


def _identity(x: Tensor) -> Tensor:
    return x


def _resolve_activation(activation: Optional[Callable | str]) -> Tuple[Callable[[Tensor], Tensor], Optional[str]]:
    if activation is None or activation == "linear":
        return _identity, None
    if isinstance(activation, str):
        name = activation.lower()
        activations: Dict[str, Callable[[Tensor], Tensor]] = {
            "relu": torch.relu,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
            "gelu": torch.nn.functional.gelu,
        }
        if name not in activations:
            raise ValueError(f"Unsupported activation '{activation}'.")
        return activations[name], name
    if callable(activation):
        return activation, getattr(activation, "__name__", activation.__class__.__name__)
    raise TypeError("activation must be None, a string identifier or a callable.")


def _resolve_bias_initializer(
    initializer: Callable | str
) -> Callable[[Tuple[int, ...], torch.device, torch.dtype], Tensor]:
    if callable(initializer):
        def wrapper(shape: Tuple[int, ...], device: torch.device, dtype: torch.dtype) -> Tensor:
            tensor = torch.empty(shape, device=device, dtype=dtype)
            result = initializer(tensor)
            return result if isinstance(result, Tensor) else tensor
        return wrapper
    if isinstance(initializer, str):
        key = initializer.lower()
        if key == "zeros":
            return lambda shape, device, dtype: torch.zeros(shape, device=device, dtype=dtype)
        if key == "ones":
            return lambda shape, device, dtype: torch.ones(shape, device=device, dtype=dtype)
        raise ValueError(f"Unsupported bias initializer '{initializer}'.")
    raise TypeError("bias_initializer must be callable or a supported string.")


def _apply_kernel_initializer(
    initializer: Optional[Callable],
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    if initializer is None:
        initializer = SpectralInitializer()
    if isinstance(initializer, SpectralInitializer):
        return initializer(shape, dtype=dtype, device=device)
    if callable(initializer):
        tensor = torch.empty(shape, dtype=dtype, device=device)
        result = initializer(tensor)
        return result if isinstance(result, Tensor) else tensor
    raise TypeError("kernel_initializer must be a callable or SpectralInitializer.")


class _DenseBase(nn.Module, LipschitzLayer, Condensable):
    def __init__(
        self,
        units: int,
        activation: Optional[Callable | str],
        use_bias: bool,
        kernel_initializer: Optional[Callable],
        bias_initializer: Callable | str,
        k_coef_lip: float,
        **kwargs,
    ) -> None:
        super().__init__()
        self.units = units
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = _resolve_bias_initializer(bias_initializer)
        self.activation_fn, self.activation_name = _resolve_activation(activation)
        self.original_activation = activation

        self.in_features: Optional[int] = None
        self.kernel: Optional[nn.Parameter] = None
        self.bias: Optional[nn.Parameter] = None
        self.built = False
        self._extra_kwargs = kwargs

        self.set_klip_factor(k_coef_lip)

    def _ensure_built(self, x: Tensor) -> None:
        if self.built:
            return
        self.in_features = x.shape[-1]
        device, dtype = x.device, x.dtype

        weight = _apply_kernel_initializer(
            self.kernel_initializer, (self.in_features, self.units), dtype, device
        )
        self.kernel = nn.Parameter(weight)

        if self.use_bias:
            bias = self.bias_initializer((self.units,), device=device, dtype=dtype)
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter("bias", None)

        self._post_build(dtype, device)
        self._init_lip_coef(x.shape)
        self.built = True

    def _compute_lip_coef(self, input_shape=None):
        return 1.0

    def _post_build(self, dtype: torch.dtype, device: torch.device) -> None:
        raise NotImplementedError

    def _compute_weight(self, training: bool) -> Tensor:
        raise NotImplementedError

    def condense(self) -> None:
        if not self.built:
            raise RuntimeError("Layer must be built before calling condense().")
        with torch.no_grad():
            wbar = self._compute_weight(training=True)
            self.kernel.copy_(wbar)

    def vanilla_export(self) -> nn.Linear:
        if not self.built:
            raise RuntimeError("Layer must be built before calling vanilla_export().")
        linear = nn.Linear(self.in_features, self.units, bias=self.use_bias)
        with torch.no_grad():
            wbar = self._compute_weight(training=True)
            linear.weight.copy_(wbar.transpose(0, 1))
            if self.use_bias and self.bias is not None:
                linear.bias.copy_(self.bias)
        return linear

    def get_config(self) -> Dict[str, object]:
        return {
            "units": self.units,
            "use_bias": self.use_bias,
            "activation": self.activation_name,
            "k_coef_lip": self.k_coef_lip,
            **self._extra_kwargs,
        }

    def forward(self, x: Tensor) -> Tensor:
        self._ensure_built(x)
        wbar = self._compute_weight(training=self.training)
        output = torch.matmul(x, wbar)
        if self.use_bias and self.bias is not None:
            output = output + self.bias
        return self.activation_fn(output)


class SpectralDense(_DenseBase):
    def __init__(
        self,
        units: int,
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
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer or SpectralInitializer(),
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

    def _post_build(self, dtype: torch.dtype, device: torch.device) -> None:
        u = torch.randn((1, self.units), device=device, dtype=dtype)
        sigma = torch.ones((1, 1), device=device, dtype=dtype)
        wbar = self.kernel.detach().clone()
        self.register_buffer("u", u)
        self.register_buffer("sigma", sigma)
        self.register_buffer("wbar", wbar)

    def _compute_weight(self, training: bool) -> Tensor:
        if not training:
            return self.wbar

        wbar, u, sigma = reshaped_kernel_orthogonalization(
            self.kernel,
            self.u,
            self._get_coef(),
            self.eps_spectral,
            self.eps_bjorck,
            self.beta_bjorck,
            self.maxiter_spectral,
            self.maxiter_bjorck,
        )
        with torch.no_grad():
            self.wbar.copy_(wbar)
            self.u.copy_(u)
            self.sigma.copy_(sigma)
        return wbar

    def get_config(self) -> Dict[str, object]:
        config = {
            "eps_spectral": self.eps_spectral,
            "eps_bjorck": self.eps_bjorck,
            "beta_bjorck": self.beta_bjorck,
            "maxiter_spectral": self.maxiter_spectral,
            "maxiter_bjorck": self.maxiter_bjorck,
        }
        base_config = super().get_config()
        return {**base_config, **config}


class FrobeniusDense(_DenseBase):
    def __init__(
        self,
        units: int,
        activation: Optional[Callable | str] = None,
        use_bias: bool = True,
        kernel_initializer: Optional[Callable] = None,
        bias_initializer: Callable | str = "zeros",
        disjoint_neurons: bool = True,
        k_coef_lip: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer or SpectralInitializer(),
            bias_initializer=bias_initializer,
            k_coef_lip=k_coef_lip,
            **kwargs,
        )
        self.disjoint_neurons = disjoint_neurons
    def _post_build(self, dtype: torch.dtype, device: torch.device) -> None:
        self.register_buffer("wbar", self.kernel.detach().clone())

    def _compute_weight(self, training: bool) -> Tensor:
        if not training:
            return self.wbar

        if self.disjoint_neurons:
            norms = torch.linalg.norm(self.kernel, dim=0, keepdim=True)
        else:
            norms = torch.linalg.norm(self.kernel)
        norms = norms + 1e-12
        wbar = self.kernel / norms * self._get_coef()

        with torch.no_grad():
            self.wbar.copy_(wbar)
        return wbar

    def get_config(self) -> Dict[str, object]:
        config = {"disjoint_neurons": self.disjoint_neurons}
        base_config = super().get_config()
        return {**base_config, **config}
