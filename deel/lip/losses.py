# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III -
# All rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Loss functions implemented with PyTorch tensors.
"""
from __future__ import annotations

import math
from functools import partial
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


_EPS = 1e-7


def _to_tensor(x: Tensor | Iterable, dtype: torch.dtype, device: torch.device) -> Tensor:
    if isinstance(x, Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, dtype=dtype, device=device)


class _BaseLoss(nn.Module):
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


def _kr(y_true: Tensor, y_pred: Tensor, epsilon: float) -> Tensor:
    dtype = y_pred.dtype
    y_true = y_true.to(dtype=dtype)
    batch_size = y_true.shape[0]
    s1 = torch.where(y_true > 0, torch.ones_like(y_true), torch.zeros_like(y_true))
    num_per_class = s1.sum(dim=0)
    pos = s1 / (num_per_class + epsilon)
    neg = (1.0 - s1) / (batch_size - num_per_class + epsilon)
    elementwise = (batch_size * y_pred * (pos - neg)).mean(dim=-1)
    return elementwise


def _kr_multi_gpu(y_true: Tensor, y_pred: Tensor) -> Tensor:
    y_true = y_true.to(dtype=y_pred.dtype)
    return (y_pred * y_true).mean(dim=-1)


def hinge_margin(y_true: Tensor, y_pred: Tensor, min_margin: Tensor | float) -> Tensor:
    sign = torch.where(y_true > 0, torch.ones_like(y_pred), -torch.ones_like(y_pred))
    sign = sign.to(dtype=y_pred.dtype)
    margin = torch.as_tensor(min_margin, dtype=y_pred.dtype, device=y_pred.device)
    hinge = F.relu(margin / 2.0 - sign * y_pred)
    return hinge.mean(dim=-1)


def multiclass_hinge(y_true: Tensor, y_pred: Tensor, min_margin: Tensor | float) -> Tensor:
    sign = torch.where(y_true > 0, torch.ones_like(y_pred), -torch.ones_like(y_pred))
    sign = sign.to(dtype=y_pred.dtype)
    margin = torch.as_tensor(min_margin, dtype=y_pred.dtype, device=y_pred.device)
    hinge = F.relu(margin / 2.0 - sign * y_pred)
    factor = y_pred.shape[-1] - 1.0
    hinge = torch.where(sign > 0, hinge * factor, hinge)
    return hinge.mean(dim=-1)


class KR(_BaseLoss):
    def __init__(self, multi_gpu: bool = False, reduction: str = "mean", name: str = "KR"):
        super().__init__(reduction=reduction, name=name)
        self.multi_gpu = multi_gpu
        self.eps = _EPS
        self.kr_function = _kr_multi_gpu if multi_gpu else partial(_kr, epsilon=self.eps)

    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        values = self.kr_function(y_true, y_pred)
        return self._apply_reduction(values)

    call = forward

    def get_config(self):
        base = super().get_config()
        base.update({"multi_gpu": self.multi_gpu})
        return base


class HingeMargin(_BaseLoss):
    def __init__(self, min_margin: float = 1.0, reduction: str = "mean", name: str = "HingeMargin"):
        super().__init__(reduction=reduction, name=name)
        self.min_margin = nn.Parameter(torch.tensor(min_margin, dtype=torch.float32), requires_grad=False)

    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        values = hinge_margin(y_true, y_pred, self.min_margin.to(device=y_pred.device, dtype=y_pred.dtype))
        return self._apply_reduction(values)

    call = forward

    def get_config(self):
        base = super().get_config()
        base.update({"min_margin": float(self.min_margin.item())})
        return base


class HKR(_BaseLoss):
    def __init__(
        self,
        alpha: float,
        min_margin: float = 1.0,
        multi_gpu: bool = False,
        reduction: str = "mean",
        name: str = "HKR",
    ):
        super().__init__(reduction=reduction, name=name)
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32), requires_grad=False)
        self.min_margin = nn.Parameter(torch.tensor(min_margin, dtype=torch.float32), requires_grad=False)
        self.multi_gpu = multi_gpu
        self.kr_loss = KR(multi_gpu=multi_gpu, reduction="none")
        self._hinge_only = math.isinf(alpha)

    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        margin = self.min_margin.to(device=y_pred.device, dtype=y_pred.dtype)
        if self._hinge_only:
            values = hinge_margin(y_true, y_pred, margin)
        else:
            kr_val = -self.kr_loss.call(y_true, y_pred)
            hinge_val = hinge_margin(y_true, y_pred, margin)
            alpha = self.alpha.to(device=y_pred.device, dtype=y_pred.dtype)
            values = kr_val + alpha * hinge_val
        return self._apply_reduction(values)

    call = forward

    def get_config(self):
        base = super().get_config()
        base.update(
            {
                "alpha": float(self.alpha.item()),
                "min_margin": float(self.min_margin.item()),
                "multi_gpu": self.multi_gpu,
            }
        )
        return base


class MulticlassKR(_BaseLoss):
    def __init__(self, multi_gpu: bool = False, reduction: str = "mean", name: str = "MulticlassKR"):
        super().__init__(reduction=reduction, name=name)
        self.multi_gpu = multi_gpu
        self.eps = _EPS
        self.kr_function = _kr_multi_gpu if multi_gpu else partial(_kr, epsilon=self.eps)

    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        values = self.kr_function(y_true, y_pred)
        return self._apply_reduction(values)

    call = forward

    def get_config(self):
        base = super().get_config()
        base.update({"multi_gpu": self.multi_gpu})
        return base


class MulticlassHinge(_BaseLoss):
    def __init__(self, min_margin: float = 1.0, reduction: str = "mean", name: str = "MulticlassHinge"):
        super().__init__(reduction=reduction, name=name)
        self.min_margin = nn.Parameter(torch.tensor(min_margin, dtype=torch.float32), requires_grad=False)

    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        values = multiclass_hinge(
            y_true,
            y_pred,
            self.min_margin.to(device=y_pred.device, dtype=y_pred.dtype),
        )
        return self._apply_reduction(values)

    call = forward

    def get_config(self):
        base = super().get_config()
        base.update({"min_margin": float(self.min_margin.item())})
        return base


class MulticlassHKR(_BaseLoss):
    def __init__(
        self,
        alpha: float = 10.0,
        min_margin: float = 1.0,
        multi_gpu: bool = False,
        reduction: str = "mean",
        name: str = "MulticlassHKR",
    ):
        super().__init__(reduction=reduction, name=name)
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32), requires_grad=False)
        self.min_margin = nn.Parameter(torch.tensor(min_margin, dtype=torch.float32), requires_grad=False)
        self.multi_gpu = multi_gpu
        self.kr_loss = MulticlassKR(multi_gpu=multi_gpu, reduction="none")
        self._hinge_only = math.isinf(alpha)

    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        margin = self.min_margin.to(device=y_pred.device, dtype=y_pred.dtype)
        if self._hinge_only:
            values = multiclass_hinge(y_true, y_pred, margin)
        else:
            kr_val = -self.kr_loss.call(y_true, y_pred)
            hinge_val = multiclass_hinge(y_true, y_pred, margin)
            alpha = self.alpha.to(device=y_pred.device, dtype=y_pred.dtype)
            values = kr_val + alpha * hinge_val
        return self._apply_reduction(values)

    call = forward

    def get_config(self):
        base = super().get_config()
        base.update(
            {
                "alpha": float(self.alpha.item()),
                "min_margin": float(self.min_margin.item()),
                "multi_gpu": self.multi_gpu,
            }
        )
        return base


class MulticlassSoftHKR(_BaseLoss):
    def __init__(
        self,
        alpha: float = 10.0,
        min_margin: float = 1.0,
        alpha_mean: float = 0.99,
        temperature: float = 1.0,
        reduction: str = "mean",
        name: str = "MulticlassSoftHKR",
    ):
        super().__init__(reduction=reduction, name=name)
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32), requires_grad=False)
        self.min_margin = float(min_margin)
        self.alpha_mean = alpha_mean
        self.temperature_scale = temperature * self.min_margin
        self.register_buffer(
            "current_mean",
            torch.tensor([self.min_margin], dtype=torch.float32),
            persistent=False,
        )
        self._hinge_only = math.isinf(alpha)

    def _update_mean(self, y_pred: Tensor) -> Tensor:
        with torch.no_grad():
            current = y_pred.abs().mean()
            new_mean = self.alpha_mean * self.current_mean + (1 - self.alpha_mean) * current
            new_mean = new_mean.clamp(0.005, 1000.0)
            self.current_mean.copy_(new_mean)
            total_mean = new_mean.clamp(self.min_margin, 20000.0)
        return total_mean.to(device=y_pred.device, dtype=y_pred.dtype)

    def _compute_temperature_softmax(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        total_mean = self._update_mean(y_pred)
        temp = (self.temperature_scale / total_mean).clamp(0.005, 250.0)
        finfo_min = -torch.finfo(y_pred.dtype).max
        opposite = torch.where(y_true > 0, torch.full_like(y_pred, finfo_min), temp * y_pred)
        f_soft = torch.softmax(opposite, dim=-1)
        f_soft = torch.where(y_true > 0, torch.ones_like(f_soft), f_soft)
        return f_soft

    def _signed_logits(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        sign = torch.where(y_true > 0, torch.ones_like(y_pred), -torch.ones_like(y_pred))
        return y_pred * sign.to(dtype=y_pred.dtype)

    def _hinge_preproc(self, signed_logits: Tensor) -> Tensor:
        margin = torch.tensor(self.min_margin, dtype=signed_logits.dtype, device=signed_logits.device)
        return F.relu(margin / 2.0 - signed_logits)

    def _multiclass_hinge_soft(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        f_soft = self._compute_temperature_softmax(y_true, y_pred)
        signed = self._signed_logits(y_true, y_pred)
        hinge = self._hinge_preproc(signed)
        return (hinge * f_soft).sum(dim=-1)

    def _hkr(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        f_soft = self._compute_temperature_softmax(y_true, y_pred)
        signed = self._signed_logits(y_true, y_pred)
        kr_term = (-signed * f_soft).sum(dim=-1)
        hinge = self._hinge_preproc(signed)
        hinge_term = (hinge * f_soft).sum(dim=-1)
        alpha = self.alpha.to(device=y_pred.device, dtype=y_pred.dtype)
        beta = torch.where(torch.isinf(alpha), torch.zeros_like(alpha), 1.0 / alpha)
        return beta * kr_term + hinge_term

    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        if not isinstance(y_pred, Tensor):
            y_pred = torch.as_tensor(y_pred)
        if not isinstance(y_true, Tensor):
            y_true = torch.as_tensor(y_true, dtype=y_pred.dtype, device=y_pred.device)
        if self._hinge_only:
            values = self._multiclass_hinge_soft(y_true, y_pred)
        else:
            values = self._hkr(y_true, y_pred)
        return self._apply_reduction(values)

    call = forward

    def get_config(self):
        base = super().get_config()
        base.update(
            {
                "alpha": float(self.alpha.item()),
                "min_margin": self.min_margin,
                "alpha_mean": self.alpha_mean,
                "temperature": self.temperature_scale / self.min_margin,
            }
        )
        return base


class MultiMargin(_BaseLoss):
    def __init__(self, min_margin: float = 1.0, reduction: str = "mean", name: str = "MultiMargin"):
        super().__init__(reduction=reduction, name=name)
        self.min_margin = nn.Parameter(torch.tensor(min_margin, dtype=torch.float32), requires_grad=False)

    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        mask = torch.where(y_true > 0, torch.ones_like(y_pred), torch.zeros_like(y_pred))
        mask = mask.to(dtype=y_pred.dtype)
        v_ytrue = (y_pred * mask).sum(dim=-1, keepdim=True)
        margin = torch.as_tensor(self.min_margin, dtype=y_pred.dtype, device=y_pred.device)
        loss = F.relu(margin - v_ytrue + y_pred)
        values = ((1.0 - mask) * loss).mean(dim=-1)
        return self._apply_reduction(values)

    call = forward

    def get_config(self):
        base = super().get_config()
        base.update({"min_margin": float(self.min_margin.item())})
        return base


class CategoricalHinge(_BaseLoss):
    def __init__(self, min_margin: float, reduction: str = "mean", name: str = "CategoricalHinge"):
        super().__init__(reduction=reduction, name=name)
        self.min_margin = nn.Parameter(torch.tensor(min_margin, dtype=torch.float32), requires_grad=False)

    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        mask = torch.where(y_true > 0, torch.ones_like(y_pred), torch.zeros_like(y_pred))
        mask = mask.to(dtype=y_pred.dtype)
        pos = (mask * y_pred).sum(dim=-1)
        finfo_min = torch.finfo(y_pred.dtype).min
        neg = torch.where(mask > 0, torch.full_like(y_pred, finfo_min), y_pred).max(dim=-1).values
        margin = self.min_margin.to(device=y_pred.device, dtype=y_pred.dtype)
        values = F.relu(margin - (pos - neg))
        return self._apply_reduction(values)

    call = forward

    def get_config(self):
        base = super().get_config()
        base.update({"min_margin": float(self.min_margin.item())})
        return base


class TauCategoricalCrossentropy(_BaseLoss):
    def __init__(self, tau: float, reduction: str = "mean", name: str = "TauCategoricalCrossentropy"):
        super().__init__(reduction=reduction, name=name)
        self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32), requires_grad=False)

    def forward(self, y_true: Tensor, y_pred: Tensor, *_, **__):
        tau = self.tau.to(device=y_pred.device, dtype=y_pred.dtype)
        logits = tau * y_pred
        log_probs = F.log_softmax(logits, dim=-1)
        y_true = y_true.to(dtype=y_pred.dtype)
        values = -(y_true * log_probs).sum(dim=-1) / tau
        return self._apply_reduction(values)

    call = forward

    def get_config(self):
        base = super().get_config()
        base.update({"tau": float(self.tau.item())})
        return base


class TauSparseCategoricalCrossentropy(_BaseLoss):
    def __init__(self, tau: float, reduction: str = "mean", name: str = "TauSparseCategoricalCrossentropy"):
        super().__init__(reduction=reduction, name=name)
        self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32), requires_grad=False)

    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        tau = self.tau.to(device=y_pred.device, dtype=y_pred.dtype)
        targets = y_true.to(dtype=torch.long, device=y_pred.device)
        values = F.cross_entropy(tau * y_pred, targets, reduction="none") / tau
        return self._apply_reduction(values)

    call = forward

    def get_config(self):
        base = super().get_config()
        base.update({"tau": float(self.tau.item())})
        return base


class TauBinaryCrossentropy(_BaseLoss):
    def __init__(self, tau: float, reduction: str = "mean", name: str = "TauBinaryCrossentropy"):
        super().__init__(reduction=reduction, name=name)
        self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32), requires_grad=False)

    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        tau = self.tau.to(device=y_pred.device, dtype=y_pred.dtype)
        labels = torch.where(y_true > 0, torch.ones_like(y_pred), torch.zeros_like(y_pred))
        logits = tau * y_pred
        values = F.binary_cross_entropy_with_logits(logits, labels, reduction="none") / tau
        return self._apply_reduction(values.mean(dim=-1))

    call = forward

    def get_config(self):
        base = super().get_config()
        base.update({"tau": float(self.tau.item())})
        return base


__all__ = [
    "KR",
    "HKR",
    "HingeMargin",
    "MulticlassKR",
    "MulticlassHinge",
    "MulticlassHKR",
    "MulticlassSoftHKR",
    "MultiMargin",
    "CategoricalHinge",
    "TauCategoricalCrossentropy",
    "TauSparseCategoricalCrossentropy",
    "TauBinaryCrossentropy",
    "hinge_margin",
    "multiclass_hinge",
]
