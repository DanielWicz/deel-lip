import random
from typing import Iterable

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_tensor(array, dtype=torch.float32):
    return torch.as_tensor(array, dtype=dtype)


def to_numpy(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy()


def to_NCHW(arr: np.ndarray):
    return arr


def to_NCHW_inv(arr: np.ndarray):
    return arr


__all__: Iterable[str] = []
