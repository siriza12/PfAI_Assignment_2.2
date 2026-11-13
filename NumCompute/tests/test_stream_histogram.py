
import numpy as np
import pytest

from numcompute.stats import StreamingHistogram


def test_histogram_multiple_chunks():
    histogram = StreamingHistogram(
        bins=2, value_range=(0, 10)
    )

    histogram.update([1, 2, 6])
    histogram.update([3, 8, 9])

    counts, edges = histogram.histogram()

    np.testing.assert_array_equal(counts, [3, 3])
    np.testing.assert_allclose(edges, [0, 5, 10])


def test_histogram_ignores_nan():
    histogram = StreamingHistogram(
        bins=2, value_range=(0, 10)
    )

    histogram.update([1, np.nan, 9])

    np.testing.assert_array_equal(
        histogram.counts, [1, 1]
    )
    assert histogram.total_seen == 2


def test_histogram_rejects_invalid_bins():
    with pytest.raises(ValueError):
        StreamingHistogram(bins=0)

    with pytest.raises(ValueError):
        StreamingHistogram(value_range=(10, 0))
