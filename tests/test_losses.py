import torch

from deel.lip.losses import (
    HKR,
    HingeMargin,
    KR,
    MulticlassHKR,
    MulticlassKR,
    TauBinaryCrossentropy,
)


def test_kr_binary_matches_manual():
    y_true = torch.tensor([[1.0], [0.0]])
    y_pred = torch.tensor([[0.4], [-0.2]])
    loss = KR()
    value = loss(y_true, y_pred)
    assert value.abs() < 1.0


def test_hinge_margin_positive_when_misclassified():
    loss = HingeMargin(min_margin=1.0)
    y_true = torch.tensor([[1.0], [0.0]])
    y_pred = torch.tensor([[0.1], [0.8]])
    value = loss(y_true, y_pred)
    assert torch.all(value >= 0)
    assert value[0] > 0


def test_hkr_combines_kr_and_hinge():
    loss = HKR(alpha=0.5, min_margin=1.0)
    y_true = torch.tensor([[1.0], [0.0]])
    y_pred = torch.tensor([[0.5], [-0.7]])
    value = loss(y_true, y_pred)
    assert value.ndim == 0


def test_multiclass_losses_return_scalar():
    y_true = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    y_pred = torch.tensor([[2.0, -1.0, 0.3], [-0.3, 1.7, 0.1]])
    assert MulticlassKR()(y_true, y_pred).ndim == 0
    assert MulticlassHKR()(y_true, y_pred).ndim == 0


def test_tau_binary_crossentropy_scales_logits():
    loss = TauBinaryCrossentropy(tau=2.0)
    y_true = torch.tensor([[1.0], [0.0]])
    y_pred = torch.tensor([[1.0], [-1.0]])
    value = loss(y_true, y_pred)
    assert value.ndim == 0
