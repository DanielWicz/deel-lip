# "Copyright" Daniel Wiczew, NEBULA IRT Antoine de Saint Exupéry et Université Paul Sabatier
# Toulouse III - All rights reserved. DEEL is a research program operated by
# IVADO, IRT Saint Exupéry, CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Contains utility functions implemented with PyTorch tensors.
"""
from __future__ import annotations

from typing import Any, Generator, Iterable, Tuple

import torch
from torch import Tensor, nn


def _infer_device_and_dtype(model: nn.Module) -> Tuple[torch.device, torch.dtype]:
    """
    Infer default device and dtype from a module. If the module does not own any
    parameters, defaults to CPU / float32.
    """
    try:
        parameter = next(model.parameters())
        return parameter.device, parameter.dtype
    except StopIteration:
        buffer = next(model.buffers(), None)
        if buffer is not None:
            return buffer.device, buffer.dtype
    return torch.device("cpu"), torch.float32


def evaluate_lip_const_gen(
    model: nn.Module,
    generator: Generator[Tuple[Iterable[Any], Iterable[Any]], Any, None],
) -> Tensor:
    """
    Evaluate the Lipschitz constant of a model on the first batch yielded by a
    generator.
    """
    try:
        batch = generator.send(None)
    except (AttributeError, TypeError):
        batch = next(generator)
    x, _ = batch
    return evaluate_lip_const(model, x)


def _prepare_input(model: nn.Module, x: Any) -> Tensor:
    device, dtype = _infer_device_and_dtype(model)
    x_tensor = torch.as_tensor(x, dtype=dtype, device=device)
    if x_tensor.dim() == 1:
        x_tensor = x_tensor.unsqueeze(0)
    return x_tensor


def evaluate_lip_const(model: nn.Module, x: Any) -> Tensor:
    """
    Evaluate the Lipschitz constant of a model, using the Jacobian of the model.
    Note that the estimation of the Lipschitz constant is done locally around input
    samples and may not capture the behaviour on the entire domain.
    """
    model_was_training = model.training
    model.eval()

    x_tensor = _prepare_input(model, x)
    batch_size = x_tensor.shape[0]

    x_tensor = x_tensor.requires_grad_(True)
    outputs = model(x_tensor)

    input_dim = int(torch.prod(torch.tensor(x_tensor.shape[1:], device=x_tensor.device)))
    output_dim = int(torch.prod(torch.tensor(outputs.shape[1:], device=outputs.device)))

    jacobian_norms = []
    for sample_id in range(batch_size):
        input_sample = x_tensor[sample_id : sample_id + 1]

        def model_fn(inp: Tensor) -> Tensor:
            return model(inp).reshape(-1)

        jacobian = torch.autograd.functional.jacobian(
            model_fn,
            input_sample,
            create_graph=False,
            vectorize=False,
        )
        jacobian = jacobian.reshape(output_dim, input_dim)
        sigma_max = torch.linalg.svdvals(jacobian).max()
        jacobian_norms.append(sigma_max)

    lip_const = torch.stack(jacobian_norms).max()
    if model_was_training:
        model.train()
    return lip_const.detach()


def _padding_circular(x: Tensor, circular_paddings: Tuple[int, int] | None) -> Tensor:
    """Add circular padding to a 4-D tensor (NCHW data format)."""
    if circular_paddings is None:
        return x
    pad_h, pad_w = circular_paddings
    if pad_h > 0:
        x = torch.cat((x[:, :, -pad_h:, :], x, x[:, :, :pad_h, :]), dim=2)
    if pad_w > 0:
        x = torch.cat((x[:, :, :, -pad_w:], x, x[:, :, :, :pad_w]), dim=3)
    return x


def _zero_upscale2D(x: Tensor, strides: Tuple[int, int]) -> Tensor:
    """
    Insert zeros between elements according to an (stride_h, stride_w) tuple for
    4-D tensors in NCHW format.
    """
    stride_h, stride_w = strides
    if stride_h == 1 and stride_w == 1:
        return x

    batch, channels, height, width = x.shape
    device, dtype = x.device, x.dtype

    if stride_w > 1:
        x = x.unsqueeze(-1)
        zeros = torch.zeros(
            (batch, channels, height, width, stride_w - 1), device=device, dtype=dtype
        )
        x = torch.cat((x, zeros), dim=-1)
        x = x.reshape(batch, channels, height, width * stride_w)

    if stride_h > 1:
        x = x.unsqueeze(3)
        zeros = torch.zeros(
            (batch, channels, height, stride_h - 1, width * stride_w),
            device=device,
            dtype=dtype,
        )
        x = torch.cat((x, zeros), dim=3)
        x = x.reshape(batch, channels, height * stride_h, width * stride_w)

    return x


def _maybe_transpose_kernel(w: Tensor, transpose: bool = False) -> Tensor:
    """Transpose 4-D convolution kernel from OIHW to IOHW with spatial flip."""
    if not transpose:
        return w
    w_adj = w.permute(1, 0, 2, 3)
    w_adj = torch.flip(w_adj, dims=(2, 3))
    return w_adj


def process_labels_for_multi_gpu(labels: Tensor) -> Tensor:
    """Process labels to be fed to any loss based on KR estimation with a
    multi-GPU/TPU strategy."""
    eps = 1e-7
    dtype = labels.dtype if labels.is_floating_point() else torch.float32
    labels = labels.to(dtype=dtype)
    labels = torch.where(labels > 0, torch.ones_like(labels), torch.zeros_like(labels))

    batch_size = labels.shape[0]
    counts = labels.sum(dim=0)

    pos = labels / (counts + eps)
    neg = (1.0 - labels) / (batch_size - counts + eps)

    return batch_size * (pos - neg)


__all__ = [
    "evaluate_lip_const_gen",
    "evaluate_lip_const",
    "_padding_circular",
    "_zero_upscale2D",
    "_maybe_transpose_kernel",
    "process_labels_for_multi_gpu",
]
