import torch

from deel.lip.layers.activations import (
    GroupSort,
    GroupSort2,
    Householder,
    MaxMin,
    PReLUlip,
)


def test_maxmin_doubles_channels():
    layer = MaxMin()
    x = torch.randn(4, 6)
    y = layer(x)
    assert y.shape == (4, 12)
    assert torch.allclose(y[:, :6], torch.relu(x))
    assert torch.allclose(y[:, 6:], torch.relu(-x))


def test_groupsort_preserves_norm():
    layer = GroupSort(n=2)
    x = torch.randn(2, 8)
    y = layer(x)
    assert y.shape == x.shape
    assert torch.all(torch.sort(x, dim=-1)[0].reshape_as(y) == torch.sort(y, dim=-1)[0])


def test_householder_reflection():
    layer = Householder()
    x = torch.randn(3, 4)
    y = layer(x)
    assert y.shape == x.shape
    # Norm is preserved by the reflection.
    assert torch.allclose(torch.linalg.norm(y, dim=-1), torch.linalg.norm(x, dim=-1))


def test_prelu_lip_limits_alpha():
    activation = PReLUlip(k_coef_lip=0.5, num_parameters=3)
    x = torch.randn(5, 3)
    for _ in range(3):
        _ = activation(x)
        assert torch.all(activation.prelu.weight.abs() <= 0.5 + 1e-6)
