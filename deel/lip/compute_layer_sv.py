# "Copyright" Daniel Wiczew, NEBULA IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III -
# All rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Singular value utilities for PyTorch modules.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional, Tuple, Type

import torch
from torch import Tensor, nn

from .layers import Condensable, GroupSort, MaxMin, PadConv2D


def _compute_sv_linear(layer: nn.Linear, *_):
    weights = layer.weight
    svd = torch.linalg.svdvals(weights)
    return (svd.min().item(), svd.max().item())


def _generate_conv_matrix(layer: nn.Module, input_shape: Tuple[int, ...]) -> Tensor:
    # input_shape assumed NCHW with batch dimension first.
    batch, channels, height, width = input_shape
    in_dim = channels * height * width
    device = next(layer.parameters()).device
    dtype = next(layer.parameters()).dtype

    basis = torch.eye(in_dim, device=device, dtype=dtype).view(
        in_dim, channels, height, width
    )
    with torch.no_grad():
        outputs = layer(basis)
    return outputs.reshape(in_dim, -1)


def _compute_sv_conv(layer: nn.Module, input_shape: Tuple[int, ...]):
    if input_shape is None:
        return (None, None)
    matrix = _generate_conv_matrix(layer, input_shape)
    svd = torch.linalg.svdvals(matrix)
    return (svd.min().item(), svd.max().item())


def _compute_sv_activation(layer: nn.Module, *_):
    if isinstance(layer, nn.ReLU):
        return (0.0, 1.0)
    if isinstance(layer, (GroupSort, MaxMin)):
        return (1.0, 1.0)
    return (None, None)


def _compute_sv_add(layer: nn.Module, input_sizes):
    if not isinstance(input_sizes, Iterable):
        return (None, None)
    count = len(list(input_sizes))
    return (float(count), float(count))


def compute_layer_sv(
    layer: nn.Module,
    input_shape: Optional[Tuple[int, ...]] = None,
    supplementary_type2sv: Optional[Dict[Type[nn.Module], Callable[[nn.Module, Optional[Tuple[int, ...]]], Tuple[Optional[float], Optional[float]]]]] = None,
):
    """
    Compute min and max singular values (or bounds) of a torch module.
    """
    supplementary_type2sv = supplementary_type2sv or {}

    if isinstance(layer, Condensable):
        layer.condense()
        layer = layer.vanilla_export()

    default_type2sv = {
        nn.Linear: _compute_sv_linear,
        nn.Conv2d: _compute_sv_conv,
        nn.ConvTranspose2d: _compute_sv_conv,
        PadConv2D: _compute_sv_conv,
        nn.ReLU: _compute_sv_activation,
        nn.Sigmoid: lambda *_: (0.0, 0.25),
        nn.Tanh: lambda *_: (0.0, 1.0),
        GroupSort: _compute_sv_activation,
        MaxMin: _compute_sv_activation,
    }

    layer_type = type(layer)
    if layer_type in supplementary_type2sv:
        return supplementary_type2sv[layer_type](layer, input_shape)
    if layer_type in default_type2sv:
        return default_type2sv[layer_type](layer, input_shape)
    return (None, None)


def compute_model_sv(
    model: nn.Module,
    input_shape: Optional[Tuple[int, ...]] = None,
    supplementary_type2sv: Optional[Dict[Type[nn.Module], Callable[[nn.Module, Optional[Tuple[int, ...]]], Tuple[Optional[float], Optional[float]]]]] = None,
):
    """Compute singular values for each child module in a model."""
    results = []
    for name, module in model.named_children():
        result = compute_layer_sv(
            module, input_shape=input_shape, supplementary_type2sv=supplementary_type2sv
        )
        results.append((name, result))
    return results


__all__ = ["compute_layer_sv", "compute_model_sv"]
