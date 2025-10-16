import torch

from deel.lip.regularizers import Lorth2D, LorthRegularizer, OrthDenseRegularizer


def test_orth_dense_regularizer_zero_for_identity():
    regularizer = OrthDenseRegularizer(lambda_orth=1.0)
    weight = torch.eye(4)
    penalty = regularizer(weight)
    assert penalty.abs() < 1e-6


def test_lorth_regularizer_identity_kernel():
    kernel = torch.zeros(3, 3, 2, 2)
    kernel[1, 1] = torch.eye(2)
    reg = LorthRegularizer(kernel_shape=kernel.shape, stride=1)
    penalty = reg(kernel)
    assert penalty.abs() < 1e-4


def test_lorth2d_updates_kernel_shape():
    lorth = Lorth2D(kernel_shape=(3, 3, 2, 2))
    lorth.set_kernel_shape((3, 3, 2, 2))
    assert lorth.kernel_shape == (3, 3, 2, 2)
