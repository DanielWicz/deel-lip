import torch

from deel.lip.metrics import (
    BinaryProvableAvgRobustness,
    BinaryProvableRobustAccuracy,
    CategoricalProvableAvgRobustness,
    CategoricalProvableRobustAccuracy,
)


def test_categorical_metrics_shapes():
    metric_acc = CategoricalProvableRobustAccuracy(lip_const=2.0)
    metric_avg = CategoricalProvableAvgRobustness(lip_const=2.0)
    y_true = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    y_pred = torch.tensor([[2.0, -0.5], [-0.3, 1.5]])
    assert metric_acc(y_true, y_pred).ndim == 0
    assert metric_avg(y_true, y_pred).ndim == 0


def test_binary_metrics_behaviour():
    y_true = torch.tensor([[1.0], [0.0]])
    y_pred = torch.tensor([[2.0], [-1.0]])
    metric_acc = BinaryProvableRobustAccuracy(lip_const=1.0)
    metric_avg = BinaryProvableAvgRobustness(lip_const=1.0)
    assert metric_acc(y_true, y_pred).ndim == 0
    assert metric_avg(y_true, y_pred).ndim == 0
