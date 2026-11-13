
import numpy as np
import pytest

from numcompute.stats import StreamingQuantiles


def test_quantiles_exact_with_small_data():
    estimator = StreamingQuantiles(reservoir_size=100)

    estimator.update([1, 2, 3])
    estimator.update([4, 5])

    assert estimator.median() == pytest.approx(3.0)
    assert estimator.quantile(0.25) == pytest.approx(2.0)
    assert estimator.quantile(0.75) == pytest.approx(4.0)


def test_quantiles_bounded_memory():
    estimator = StreamingQuantiles(
        reservoir_size=20,
        random_state=5
    )

    estimator.update(np.arange(1000))

    assert estimator.n_seen == 1000
    assert estimator.stored_samples == 20


def test_quantiles_ignore_nan():
    estimator = StreamingQuantiles()

    estimator.update([1, np.nan, 3])

    assert estimator.n_seen == 2
    assert estimator.median() == pytest.approx(2.0)


def test_quantiles_invalid_request():
    estimator = StreamingQuantiles()

    with pytest.raises(RuntimeError):
        estimator.median()

    with pytest.raises(ValueError):
        estimator.quantile(1.5)

    with pytest.raises(ValueError):
        StreamingQuantiles(reservoir_size=0)
