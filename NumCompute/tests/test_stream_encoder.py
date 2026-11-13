
import numpy as np
import pytest

from numcompute.preprocessing import OneHotEncoder


def test_encoder_expands_categories():
    encoder = OneHotEncoder()

    encoder.partial_fit([["red"], ["blue"]])
    encoder.partial_fit([["green"]])

    result = encoder.transform([["red"], ["green"]])

    assert result.shape == (2, 3)
    np.testing.assert_array_equal(result.sum(axis=1), [1, 1])


def test_encoder_retains_previous_categories():
    encoder = OneHotEncoder()

    encoder.partial_fit([["cat"]])
    encoder.partial_fit([["dog"]])

    result = encoder.transform([["cat"]])

    assert result.shape == (1, 2)
    assert result.sum() == 1


def test_encoder_rejects_feature_change():
    encoder = OneHotEncoder()

    encoder.partial_fit([["red"]])

    with pytest.raises(ValueError):
        encoder.partial_fit([["blue", "large"]])
