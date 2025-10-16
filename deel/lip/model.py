# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III -
# All rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
PyTorch helpers providing Lipschitz-aware Sequential/Model containers.
"""
from __future__ import annotations

import copy
import math
from warnings import warn

import torch.nn as nn

from .layers import Condensable, LipschitzLayer


_MSG_NOT_LIP = "Sequential model contains a layer which is not a 1-Lipschitz layer: {}"


def _is_supported_1lip_layer(layer: nn.Module) -> bool:
    supported = (nn.Flatten, nn.Identity, nn.Softmax, nn.Sigmoid, nn.Tanh)
    if isinstance(layer, supported):
        return True
    if isinstance(layer, nn.ReLU):
        return layer.inplace in {False, True}
    if isinstance(layer, nn.MaxPool2d):
        return layer.kernel_size <= layer.stride
    return False


class Sequential(nn.Sequential, LipschitzLayer, Condensable):
    def __init__(self, *args, k_coef_lip: float = 1.0):
        super().__init__(*args)
        self.set_klip_factor(k_coef_lip)

    def set_klip_factor(self, klip_factor: float):
        super().set_klip_factor(klip_factor)
        lip_layers = [m for m in self if isinstance(m, LipschitzLayer)]
        nb_layers = len(lip_layers) or 1
        for layer in self:
            if isinstance(layer, LipschitzLayer):
                layer.set_klip_factor(math.pow(klip_factor, 1.0 / nb_layers))
            elif not _is_supported_1lip_layer(layer):
                warn(_MSG_NOT_LIP.format(layer.__class__.__name__))

    def _compute_lip_coef(self, input_shape=None):
        for layer in self:
            if isinstance(layer, LipschitzLayer):
                layer._compute_lip_coef(input_shape)
            elif not _is_supported_1lip_layer(layer):
                warn(_MSG_NOT_LIP.format(layer.__class__.__name__))

    def _init_lip_coef(self, input_shape):
        for layer in self:
            if isinstance(layer, LipschitzLayer):
                layer._init_lip_coef(input_shape)
            elif not _is_supported_1lip_layer(layer):
                warn(_MSG_NOT_LIP.format(layer.__class__.__name__))

    def _get_coef(self):
        global_coef = 1.0
        for layer in self:
            if isinstance(layer, LipschitzLayer) and global_coef is not None:
                global_coef *= layer._get_coef()
            elif not _is_supported_1lip_layer(layer):
                warn(_MSG_NOT_LIP.format(layer.__class__.__name__))
                global_coef = None
        return global_coef

    def condense(self):
        for layer in self:
            if isinstance(layer, Condensable):
                layer.condense()

    def vanilla_export(self):
        return vanillaModel(self)


class Model(nn.Module):
    def condense(self):
        for module in self.modules():
            if isinstance(module, Condensable):
                module.condense()

    def vanilla_export(self):
        return vanillaModel(self)


def vanillaModel(model: nn.Module) -> nn.Module:
    def _clone(module: nn.Module) -> nn.Module:
        if isinstance(module, Condensable):
            return module.vanilla_export()
        cloned = copy.deepcopy(module)
        for name, child in module.named_children():
            setattr(cloned, name, _clone(child))
        return cloned

    return _clone(model)
