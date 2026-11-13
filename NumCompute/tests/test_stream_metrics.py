
import numpy as np
import pytest

from numcompute.metrics import StreamingClassificationMetrics


def test_metrics_update_across_chunks():
    metrics = StreamingClassificationMetrics()

    metrics.update([0, 1], [0, 1])
    metrics.update([0, 1], [1, 1])

    assert metrics.total_samples == 4
    assert metrics.accuracy() == pytest.approx(0.75)

    np.testing.assert_array_equal(
        metrics.confusion_matrix(),
        [[1, 1], [0, 2]]
    )


def test_metrics_perfect_predictions():
    metrics = StreamingClassificationMetrics()

    metrics.update([0, 1, 0, 1], [0, 1, 0, 1])

    assert metrics.accuracy() == 1.0
    assert metrics.precision() == 1.0
    assert metrics.recall() == 1.0
    assert metrics.f1_score() == 1.0


def test_metrics_new_class_expansion():
    metrics = StreamingClassificationMetrics()

    metrics.update([0, 1], [0, 1])
    metrics.update([2], [2])

    assert metrics.confusion_matrix().shape == (3, 3)
    assert metrics.total_samples == 3


def test_metrics_rejects_mismatched_lengths():
    metrics = StreamingClassificationMetrics()

    with pytest.raises(ValueError):
        metrics.update([0, 1], [0])



def test_streaming_metrics_result():
    metrics = StreamingClassificationMetrics()
    metrics.update([0, 1, 1, 0], [0, 1, 0, 0])

    result = metrics.result()

    assert result["accuracy"] == pytest.approx(0.75)
    assert result["total_samples"] == 4
    assert result["confusion_matrix"].shape == (2, 2)


def test_streaming_metrics_reset():
    metrics = StreamingClassificationMetrics()
    metrics.update([0, 1], [0, 1])

    returned = metrics.reset()

    assert returned is metrics
    assert metrics.total_samples == 0
    assert metrics.accuracy() == 0.0
    assert metrics.confusion_matrix().shape == (0, 0)
