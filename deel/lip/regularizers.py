# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III -
# All rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Regularization utilities implemented with PyTorch tensors.
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


class Lorth(ABC):
    def __init__(self, dim: int, kernel_shape=None, stride: int = 1, conv_transpose: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.stride = stride
        self.conv_transpose = conv_transpose
        self.kernel_shape = None
        self.padding = None
        self.delta = None
        self.alphaNormSpectral = None
        self.set_kernel_shape(kernel_shape)

    def _get_kernel_shape(self):
        return [self.kernel_shape[i] for i in (0, -2, -1)]

    def _compute_delta(self):
        _, c_in, c_out = self._get_kernel_shape()
        if not self.conv_transpose:
            delta = c_out - (self.stride**self.dim) * c_in
        else:
            delta = c_in - (self.stride**self.dim) * c_out
        return max(0, delta)

    def _alpha_norm_spectral(self):
        r, c1, m1 = self._get_kernel_shape()
        if not self.conv_transpose:
            c, m = c1, m1
        else:
            c, m = m1, c1
        alpha = r * c * m
        if m - (self.stride**self.dim) * c <= 0:
            alpha = m * (2 * ((r - 1) // self.stride) + 1) ** self.dim
        if m - (self.stride**self.dim) * c >= 0:
            alpha = min(alpha, c * (2 * r - 1) ** self.dim)
        return alpha

    def _check_if_orthconv_exists(self):
        r, c, m = self._get_kernel_shape()
        msg = "Impossible {} configuration for orthogonal convolution."
        if c * self.stride**self.dim >= m:
            if m > c * (r**self.dim):
                raise RuntimeError(msg.format("RO"))
        else:
            if self.stride > r:
                raise RuntimeError(msg.format("CO"))
        if c * (self.stride**self.dim) == m:
            warnings.warn("LorthRegularizer: configuration C*S^2=M is hard to optimize.")

    def set_kernel_shape(self, shape):
        if shape is None:
            self.kernel_shape, self.padding, self.delta = None, None, None
            return
        r = shape[0]
        self.kernel_shape = shape
        self.padding = ((r - 1) // self.stride) * self.stride
        self.delta = self._compute_delta()
        self.alphaNormSpectral = self._alpha_norm_spectral()
        assert r & 1, "Lorth regularizer requires odd kernels. Received {}".format(r)
        self._check_if_orthconv_exists()

    @abstractmethod
    def _compute_conv_kk(self, w: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def _compute_target(self, w: Tensor, output_shape: Tuple[int, ...]) -> Tensor:
        raise NotImplementedError

    def compute_lorth(self, w: Tensor) -> Tensor:
        output = self._compute_conv_kk(w)
        target = self._compute_target(w, output.shape)
        return (output - target).pow(2).sum() - self.delta


class Lorth2D(Lorth):
    def __init__(self, kernel_shape=None, stride: int = 1, conv_transpose: bool = False) -> None:
        super().__init__(dim=2, kernel_shape=kernel_shape, stride=stride, conv_transpose=conv_transpose)

    def _prepare_kernel(self, w: Tensor) -> Tensor:
        if w.shape == self.kernel_shape:
            return w
        return w.permute(2, 3, 1, 0)

    def _compute_conv_kk(self, w: Tensor) -> Tensor:
        w_tf = self._prepare_kernel(w)
        input_tensor = w_tf.permute(3, 2, 0, 1)  # (C_out, C_in, kh, kw)
        if self.padding > 0:
            input_tensor = F.pad(input_tensor, (self.padding, self.padding, self.padding, self.padding))
        weight = w_tf.permute(3, 2, 0, 1)
        conv = F.conv2d(input_tensor, weight, stride=self.stride, padding=0)
        return conv.permute(0, 2, 3, 1)

    def _compute_target(self, w: Tensor, output_shape: Tuple[int, ...]) -> Tensor:
        w_tf = self._prepare_kernel(w)
        c_out = w_tf.shape[-1]
        out_h = output_shape[1]
        out_w = output_shape[2]
        target = torch.zeros((out_h, out_w, c_out, c_out), dtype=w_tf.dtype, device=w_tf.device)
        center = out_h // 2
        target[center, center] = torch.eye(c_out, dtype=w_tf.dtype, device=w_tf.device)
        return target.permute(2, 0, 1, 3)


class LorthRegularizer:
    def __init__(
        self,
        kernel_shape=None,
        stride: int = 1,
        lambda_lorth: float = 1.0,
        dim: int = 2,
        conv_transpose: bool = False,
    ) -> None:
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.lambda_lorth = lambda_lorth
        self.dim = dim
        self.conv_transpose = conv_transpose
        if dim != 2:
            raise NotImplementedError("Only 2D convolutions are supported for Lorth.")
        self.lorth = Lorth2D(kernel_shape, stride, conv_transpose)

    def set_kernel_shape(self, shape):
        self.kernel_shape = shape
        self.lorth.set_kernel_shape(shape)

    def __call__(self, w: Tensor) -> Tensor:
        penalty = self.lorth.compute_lorth(w)
        return self.lambda_lorth * penalty

    def get_config(self):
        return {
            "kernel_shape": self.kernel_shape,
            "stride": self.stride,
            "lambda_lorth": self.lambda_lorth,
            "dim": self.dim,
            "conv_transpose": self.conv_transpose,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class OrthDenseRegularizer:
    def __init__(self, lambda_orth: float = 1.0) -> None:
        self.lambda_orth = lambda_orth

    def _dense_orth_dist(self, w: Tensor) -> Tensor:
        if w.shape[0] <= w.shape[1]:
            gram = w @ w.transpose(0, 1)
        else:
            gram = w.transpose(0, 1) @ w
        eye = torch.eye(gram.shape[0], dtype=w.dtype, device=w.device)
        return (gram - eye).pow(2).sum()

    def __call__(self, w: Tensor) -> Tensor:
        return self.lambda_orth * self._dense_orth_dist(w)

    def get_config(self):
        return {"lambda_orth": self.lambda_orth}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


__all__ = ["Lorth2D", "LorthRegularizer", "OrthDenseRegularizer"]
