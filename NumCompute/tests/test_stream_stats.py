
import numpy as np
import pytest

from numcompute.stats import StreamingFeatureStats


def test_feature_stats_across_chunks():
    stats = StreamingFeatureStats()

    stats.update([[1, 10], [3, 20]])
    stats.update([[5, 30]])

    np.testing.assert_allclose(stats.mean, [3, 20])
    np.testing.assert_allclose(stats.count, [3, 3])
    np.testing.assert_allclose(stats.min, [1, 10])
    np.testing.assert_allclose(stats.max, [5, 30])


def test_feature_stats_variance():
    stats = StreamingFeatureStats()

    stats.update([[1], [3]])
    stats.update([[5]])

    np.testing.assert_allclose(
        stats.variance(), [8 / 3]
    )


def test_feature_stats_skips_nan():
    stats = StreamingFeatureStats()

    stats.update([[1, np.nan], [3, 10]])
    stats.update([[5, 20]])

    np.testing.assert_allclose(stats.mean, [3, 15])
    np.testing.assert_array_equal(stats.count, [3, 2])

    with pytest.raises(ValueError):
        stats.update([[1, 2, 3]])


def test_update_stats_api():
    stats = StreamingFeatureStats()

    stats.update_stats([[1, 10], [3, 20]])
    stats.update_stats([[5, 30]])

    np.testing.assert_allclose(stats.mean, [3, 20])
    np.testing.assert_array_equal(stats.count, [3, 3])


def test_update_stats_returns_self():
    stats = StreamingFeatureStats()

    result = stats.update_stats([[2, 4], [6, 8]])

    assert result is stats
    np.testing.assert_allclose(stats.mean, [4, 6])
