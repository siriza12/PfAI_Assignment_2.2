
import numpy as np
import pytest

from numcompute.preprocessing import StreamingMeanImputer


def test_imputer_updates_mean_across_chunks():
    imputer = StreamingMeanImputer()

    imputer.partial_fit([[2], [4]])
    imputer.partial_fit([[6], [np.nan]])

    result = imputer.transform([[np.nan]])

    np.testing.assert_allclose(result, [[4.0]])


def test_imputer_all_missing_column():
    imputer = StreamingMeanImputer(fill_value=0)

    imputer.partial_fit([[np.nan], [np.nan]])

    np.testing.assert_allclose(
        imputer.transform([[np.nan]]),
        [[0.0]]
    )


def test_imputer_rejects_feature_change():
    imputer = StreamingMeanImputer()

    imputer.partial_fit([[1], [2]])

    with pytest.raises(ValueError):
        imputer.partial_fit([[1, 2]])
