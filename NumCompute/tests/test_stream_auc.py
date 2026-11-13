
import numpy as np
import pytest

from numcompute.metrics import StreamingAUC


def test_auc_perfect_separation():
    metric = StreamingAUC()

    metric.update([0, 0], [0.1, 0.2])
    metric.update([1, 1], [0.8, 0.9])

    assert metric.auc() == pytest.approx(1.0)


def test_auc_reversed_separation():
    metric = StreamingAUC()

    metric.update([0, 0, 1, 1], [0.9, 0.8, 0.1, 0.2])

    assert metric.auc() == pytest.approx(0.0)


def test_auc_tied_scores():
    metric = StreamingAUC()

    metric.update([0, 1], [0.5, 0.5])

    assert metric.auc() == pytest.approx(0.5)


def test_auc_requires_both_classes():
    metric = StreamingAUC()

    metric.update([1, 1], [0.8, 0.9])

    assert np.isnan(metric.auc())

    with pytest.raises(ValueError):
        metric.update([0, 1], [0.2, 1.5])
