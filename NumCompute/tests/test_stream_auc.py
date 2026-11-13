
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



def test_streaming_auc_result():
    metric = StreamingAUC()

    metric.update([0, 0], [0.1, 0.2])
    metric.update([1, 1], [0.8, 0.9])

    result = metric.result()

    assert result["auc"] == pytest.approx(1.0)
    assert result["total_samples"] == 4
    assert result["positive_samples"] == 2
    assert result["negative_samples"] == 2


def test_streaming_auc_reset():
    metric = StreamingAUC()

    metric.update([0, 1], [0.2, 0.8])
    returned = metric.reset()

    assert returned is metric
    assert metric.total_samples == 0
    assert metric.positive_counts.sum() == 0
    assert metric.negative_counts.sum() == 0
    assert np.isnan(metric.auc())
