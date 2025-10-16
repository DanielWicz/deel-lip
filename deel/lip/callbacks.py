# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III -
# All rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Lightweight callback utilities for PyTorch training loops.
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, Optional

import numpy as np
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    SummaryWriter = None

from .layers import Condensable


class Callback:
    def set_model(self, model):
        self.model = model

    def on_train_batch_begin(self, batch: int, logs: Optional[Dict[str, float]] = None):
        pass

    def on_train_batch_end(self, batch: int, logs: Optional[Dict[str, float]] = None):
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None):
        pass


class CondenseCallback(Callback):
    def __init__(self, on_epoch: bool = True, on_batch: bool = False):
        self.on_epoch = on_epoch
        self.on_batch = on_batch

    def _condense_model(self):
        for module in self.model.modules():
            if isinstance(module, Condensable):
                module.condense()

    def on_train_batch_end(self, batch: int, logs: Optional[Dict[str, float]] = None):
        if self.on_batch:
            self._condense_model()

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None):
        if self.on_epoch:
            self._condense_model()

    def get_config(self):
        return {"on_epoch": self.on_epoch, "on_batch": self.on_batch}


class MonitorCallback(Callback):
    def __init__(
        self,
        monitored_layers: Iterable[str],
        logdir: str,
        target: str = "kernel",
        what: str = "max",
        on_epoch: bool = True,
        on_batch: bool = False,
    ):
        self.on_epoch = on_epoch
        self.on_batch = on_batch
        if SummaryWriter is None:
            raise ImportError(
                "tensorboard is required to use MonitorCallback. "
                "Install it with `pip install tensorboard`."
            )
        if target not in {"kernel", "wbar"}:
            raise ValueError("target must be 'kernel' or 'wbar'.")
        if what not in {"max", "all"}:
            raise ValueError("what must be 'max' or 'all'.")
        self.target = target
        self.what = what
        self.logdir = logdir
        os.makedirs(os.path.join(logdir, "metrics"), exist_ok=True)
        self.writer = SummaryWriter(os.path.join(logdir, "metrics"))
        self.monitored_layers = list(monitored_layers)
        if on_batch and on_epoch:
            self.on_epoch = False
        self.epochs = 0

    def set_model(self, model):
        super().set_model(model)
        self._layers = dict(model.named_modules())
        self._steps_per_epoch = getattr(model, "steps_per_epoch", None)

    def _get_tensor(self, layer, target: str) -> torch.Tensor:
        if target == "kernel":
            if hasattr(layer, "weight"):
                return layer.weight
        if hasattr(layer, target):
            attr = getattr(layer, target)
            if isinstance(attr, torch.nn.Parameter):
                return attr.data
            if isinstance(attr, torch.Tensor):
                return attr
        raise RuntimeError(f"[MonitorCallback] layer {layer.__class__.__name__} has no attribute {target}")

    def _monitor(self, step: int):
        for layer_name in self.monitored_layers:
            if layer_name not in self._layers:
                continue
            layer = self._layers[layer_name]
            tensor = self._get_tensor(layer, self.target)
            matrix = tensor.reshape(-1, tensor.shape[-1])
            sigmas = torch.linalg.svdvals(matrix.detach())
            if self.what == "max":
                self.writer.add_scalar(f"{layer_name}_{self.target}_sigma_max", sigmas.max().item(), step)
            else:
                self.writer.add_histogram(f"{layer_name}_{self.target}_sigmas", sigmas.cpu().numpy(), step)

    def on_train_batch_end(self, batch: int, logs: Optional[Dict[str, float]] = None):
        if self.on_batch:
            step = self.epochs * (self._steps_per_epoch or 0) + batch
            self._monitor(step)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None):
        self.epochs += 1
        if self.on_epoch:
            step = self.epochs * (self._steps_per_epoch or 0)
            self._monitor(step)


class LossParamScheduler(Callback):
    def __init__(self, param_name: str, fp, xp, step: int = 0):
        self.xp = xp
        self.fp = fp
        self.step = step
        self.param_name = param_name

    def on_train_batch_begin(self, batch: int, logs=None):
        new_value = float(np.interp(self.step, self.xp, self.fp))
        param = getattr(self.model.loss, self.param_name)
        if isinstance(param, torch.nn.Parameter):
            param.data.fill_(new_value)
        else:
            setattr(self.model.loss, self.param_name, new_value)
        self.step += 1

    def get_config(self):
        return {"xp": self.xp, "fp": self.fp, "step": self.step, "param_name": self.param_name}


class LossParamLog(Callback):
    def __init__(self, param_name: str, rate: int = 1):
        self.param_name = param_name
        self.rate = rate

    def on_epoch_end(self, epoch: int, logs=None):
        if epoch % self.rate == 0:
            value = getattr(self.model.loss, self.param_name)
            if isinstance(value, torch.nn.Parameter):
                value = value.item()
            print(f"{self.model.loss.name}.{self.param_name} = {value}")
