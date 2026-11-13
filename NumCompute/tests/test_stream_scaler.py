
import numpy as np
import pytest

from numcompute.preprocessing import StandardScaler


def test_scaler_incremental_mean():
    scaler = StandardScaler()

    scaler.partial_fit([[1], [3]])
    scaler.partial_fit([[5], [7]])

    np.testing.assert_allclose(scaler.mean_, [4])
    np.testing.assert_allclose(scaler.scale_, [np.sqrt(5)])


def test_scaler_handles_missing_values():
    scaler = StandardScaler()

    scaler.partial_fit([[np.nan], [2]])
    scaler.partial_fit([[4], [6]])

    np.testing.assert_allclose(scaler.mean_, [4])
    assert scaler.n_seen_[0] == 3


def test_scaler_rejects_feature_change():
    scaler = StandardScaler()
    scaler.partial_fit([[1], [2]])

    with pytest.raises(ValueError):
        scaler.partial_fit([[1, 2]])
