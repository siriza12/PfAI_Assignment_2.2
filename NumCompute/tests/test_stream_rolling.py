
import pytest

from numcompute.metrics import RollingWindowMetrics


def test_rolling_accuracy_across_chunks():
    metric = RollingWindowMetrics(window_size=4)

    metric.update([0, 1], [0, 1])
    metric.update([0, 1], [1, 1])

    assert metric.accuracy() == pytest.approx(0.75)
    assert metric.samples_in_window == 4


def test_rolling_window_discards_old_samples():
    metric = RollingWindowMetrics(window_size=3)

    metric.update([0, 0, 0], [1, 1, 1])
    metric.update([1, 1, 1], [1, 1, 1])

    assert metric.accuracy() == pytest.approx(1.0)
    assert metric.total_seen == 6
    assert metric.samples_in_window == 3


def test_rolling_metrics_reset():
    metric = RollingWindowMetrics(window_size=5)

    metric.update([0, 1], [0, 1])
    metric.reset()

    assert metric.total_seen == 0
    assert metric.samples_in_window == 0
    assert metric.accuracy() == 0.0


def test_rolling_rejects_invalid_window():
    with pytest.raises(ValueError):
        RollingWindowMetrics(window_size=0)

    metric = RollingWindowMetrics()

    with pytest.raises(ValueError):
        metric.update([0, 1], [0])
