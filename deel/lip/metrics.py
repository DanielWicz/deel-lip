# "Copyright" Daniel Wiczew, NEBULA IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III -
# All rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Provable robustness metrics implemented with PyTorch tensors.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn


class _BaseMetric(nn.Module):
    def __init__(self, reduction: str = "mean", name: Optional[str] = None) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of {'mean', 'sum', 'none'}.")
        self.reduction = reduction
        self.name = name or self.__class__.__name__

    def _apply_reduction(self, values: Tensor) -> Tensor:
        if self.reduction == "mean":
            return values.mean()
        if self.reduction == "sum":
            return values.sum()
        return values

    def get_config(self):
        return {"reduction": self.reduction, "name": self.name}


def _delta_multiclass(y_true: Tensor, y_pred: Tensor) -> Tensor:
    mask = torch.where(y_true > 0, torch.ones_like(y_pred), torch.zeros_like(y_pred))
    mask = mask.to(dtype=y_pred.dtype)
    true_scores = (y_pred * mask).sum(dim=-1)
    neg_inf = torch.finfo(y_pred.dtype).min
    other_scores = torch.where(mask > 0, torch.full_like(y_pred, neg_inf), y_pred).max(dim=-1).values
    return true_scores - other_scores


def _delta_binary(y_true: Tensor, y_pred: Tensor) -> Tensor:
    dtype = y_pred.dtype
    signed_targets = torch.sign(y_true.to(dtype) - 1e-3)
    return signed_targets * y_pred.squeeze(-1)


class CategoricalProvableRobustAccuracy(_BaseMetric):
    def __init__(
        self,
        epsilon: float = 36 / 255,
        lip_const: float = 1.0,
        disjoint_neurons: bool = True,
        reduction: str = "mean",
        name: str = "CategoricalProvableRobustAccuracy",
    ):
        super().__init__(reduction=reduction, name=name)
        self.epsilon = epsilon
        self.lip_const = lip_const
        if disjoint_neurons:
            self.certificate_factor = 2 * lip_const
        else:
            self.certificate_factor = math.sqrt(2) * lip_const

    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        delta = _delta_multiclass(y_true, y_pred)
        certified = (delta / self.certificate_factor) > self.epsilon
        values = certified.to(dtype=y_pred.dtype)
        return self._apply_reduction(values)

    call = forward

    def get_config(self):
        base = super().get_config()
        base.update(
            {
                "epsilon": self.epsilon,
                "lip_const": self.lip_const,
                "certificate_factor": self.certificate_factor,
            }
        )
        return base


class BinaryProvableRobustAccuracy(_BaseMetric):
    def __init__(
        self,
        epsilon: float = 36 / 255,
        lip_const: float = 1.0,
        reduction: str = "mean",
        name: str = "BinaryProvableRobustAccuracy",
    ):
        super().__init__(reduction=reduction, name=name)
        self.epsilon = epsilon
        self.lip_const = lip_const

    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        delta = _delta_binary(y_true, y_pred)
        certified = (delta / self.lip_const) > self.epsilon
        values = certified.to(dtype=y_pred.dtype)
        return self._apply_reduction(values)

    call = forward

    def get_config(self):
        base = super().get_config()
        base.update({"epsilon": self.epsilon, "lip_const": self.lip_const})
        return base


class CategoricalProvableAvgRobustness(_BaseMetric):
    def __init__(
        self,
        lip_const: float = 1.0,
        disjoint_neurons: bool = True,
        negative_robustness: bool = False,
        reduction: str = "mean",
        name: str = "CategoricalProvableAvgRobustness",
    ):
        super().__init__(reduction=reduction, name=name)
        self.lip_const = lip_const
        self.negative_robustness = negative_robustness
        if disjoint_neurons:
            self.certificate_factor = 2 * lip_const
        else:
            self.certificate_factor = math.sqrt(2) * lip_const

    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        delta = _delta_multiclass(y_true, y_pred)
        values = delta / self.certificate_factor
        if not self.negative_robustness:
            values = torch.relu(values)
        return self._apply_reduction(values.to(dtype=y_pred.dtype))

    call = forward

    def get_config(self):
        base = super().get_config()
        base.update(
            {
                "lip_const": self.lip_const,
                "certificate_factor": self.certificate_factor,
                "negative_robustness": self.negative_robustness,
            }
        )
        return base


class BinaryProvableAvgRobustness(_BaseMetric):
    def __init__(
        self,
        lip_const: float = 1.0,
        negative_robustness: bool = False,
        reduction: str = "mean",
        name: str = "BinaryProvableAvgRobustness",
    ):
        super().__init__(reduction=reduction, name=name)
        self.lip_const = lip_const
        self.negative_robustness = negative_robustness

    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        delta = _delta_binary(y_true, y_pred)
        values = delta / self.lip_const
        if not self.negative_robustness:
            values = torch.relu(values)
        return self._apply_reduction(values.to(dtype=y_pred.dtype))

    call = forward

    def get_config(self):
        base = super().get_config()
        base.update(
            {
                "lip_const": self.lip_const,
                "negative_robustness": self.negative_robustness,
            }
        )
        return base


__all__ = [
    "CategoricalProvableRobustAccuracy",
    "BinaryProvableRobustAccuracy",
    "CategoricalProvableAvgRobustness",
    "BinaryProvableAvgRobustness",
]
