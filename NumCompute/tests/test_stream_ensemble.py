
import pytest

from numcompute_stream.ensemble import EnsembleClassifier


def test_ensemble_has_requested_number_of_trees():
    model = EnsembleClassifier(n_estimators=7)

    assert len(model.estimators_) == 7


def test_ensemble_fit_and_predict():
    model = EnsembleClassifier(
        n_estimators=9,
        random_state=10
    )

    model.fit(
        [[1], [2], [3], [4], [5], [6], [7], [8]],
        [0, 0, 0, 0, 1, 1, 1, 1]
    )

    assert model.predict([[1], [8]]).tolist() == [0, 1]


def test_ensemble_partial_fit_across_chunks():
    model = EnsembleClassifier(
        n_estimators=7,
        random_state=4
    )

    model.partial_fit([[1], [2]], [0, 0])
    model.partial_fit([[3], [4]], [1, 1])

    assert model.predict([[1], [4]]).tolist() == [0, 1]


def test_ensemble_partial_fit_returns_self():
    model = EnsembleClassifier(random_state=0)

    result = model.partial_fit(
        [[1], [2]],
        [0, 1]
    )

    assert result is model


def test_ensemble_reproducible():
    first = EnsembleClassifier(
        n_estimators=5,
        random_state=42
    )

    second = EnsembleClassifier(
        n_estimators=5,
        random_state=42
    )

    X = [[1], [2], [3], [4], [5], [6]]
    y = [0, 0, 0, 1, 1, 1]

    first.fit(X, y)
    second.fit(X, y)

    assert first.predict(X).tolist() == second.predict(X).tolist()


def test_ensemble_rejects_invalid_tree_count():
    with pytest.raises(ValueError):
        EnsembleClassifier(n_estimators=0)


def test_ensemble_predict_before_training():
    model = EnsembleClassifier()

    with pytest.raises(RuntimeError):
        model.predict([[1]])


def test_ensemble_rejects_changed_feature_count():
    model = EnsembleClassifier(random_state=0)

    model.partial_fit([[1], [2]], [0, 1])

    with pytest.raises(ValueError):
        model.partial_fit([[1, 2]], [0])
