
import numpy as np
import pytest

from numcompute.pipeline import Pipeline
from numcompute_stream.tree import DecisionTreeClassifier


class DoubleTransformer:
    """Simple incremental transformer for testing."""

    def __init__(self):
        self.updates = 0

    def fit(self, X, y=None):
        return self

    def partial_fit(self, X, y=None):
        self.updates += 1
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float) * 2


def test_pipeline_partial_fit_multiple_chunks():
    transformer = DoubleTransformer()
    model = DecisionTreeClassifier()

    pipeline = Pipeline([
        ("double", transformer),
        ("tree", model),
    ])

    pipeline.partial_fit([[1], [2]], [0, 0])
    pipeline.partial_fit([[3], [4]], [1, 1])

    assert pipeline.predict([[1], [4]]).tolist() == [0, 1]
    assert transformer.updates == 2


def test_pipeline_partial_fit_returns_self():
    pipeline = Pipeline([
        ("tree", DecisionTreeClassifier())
    ])

    result = pipeline.partial_fit(
        [[1], [2]],
        [0, 1]
    )

    assert result is pipeline


def test_pipeline_rejects_non_incremental_step():
    class NonIncrementalTransformer:
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X

    pipeline = Pipeline([
        ("old_transformer", NonIncrementalTransformer()),
        ("tree", DecisionTreeClassifier()),
    ])

    with pytest.raises(TypeError):
        pipeline.partial_fit([[1], [2]], [0, 1])
