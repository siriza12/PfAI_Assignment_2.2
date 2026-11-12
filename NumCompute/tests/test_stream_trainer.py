
import numpy as np
import pytest

from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.ensemble import EnsembleClassifier
from numcompute_stream.stream import StreamTrainer


def test_trainer_fit_and_score():
    trainer = StreamTrainer(DecisionTreeClassifier())

    trainer.fit_chunk(
        [[1], [2], [3], [4]],
        [0, 0, 1, 1]
    )

    assert trainer.score_chunk(
        [[1], [4]], [0, 1]
    ) == pytest.approx(1.0)


def test_trainer_multiple_chunks():
    trainer = StreamTrainer(DecisionTreeClassifier())

    trainer.fit_chunk([[1], [2]], [0, 0])
    trainer.fit_chunk([[3], [4]], [1, 1])

    assert trainer.chunk_count == 2
    assert trainer.score_chunk([[1], [4]], [0, 1]) == 1.0


def test_trainer_history():
    trainer = StreamTrainer(DecisionTreeClassifier())

    trainer.fit_chunk([[1], [2]], [0, 1])
    trainer.score_chunk([[1], [2]], [0, 1])

    records = trainer.get_history()

    assert len(records) == 2
    assert records[0]["phase"] == "train"
    assert records[1]["phase"] == "evaluation"


def test_trainer_memory_logging():
    trainer = StreamTrainer(DecisionTreeClassifier())

    trainer.fit_chunk([[1], [2]], [0, 1])

    record = trainer.get_history()[0]

    assert record["input_bytes"] > 0
    assert record["traced_memory_bytes"] >= 0
    assert record["elapsed_seconds"] >= 0


def test_trainer_rejects_empty_chunk():
    trainer = StreamTrainer(DecisionTreeClassifier())

    with pytest.raises(ValueError):
        trainer.fit_chunk([], [])


def test_trainer_rejects_mismatched_rows():
    trainer = StreamTrainer(DecisionTreeClassifier())

    with pytest.raises(ValueError):
        trainer.fit_chunk([[1], [2]], [0])


def test_trainer_score_before_fit():
    trainer = StreamTrainer(DecisionTreeClassifier())

    with pytest.raises(RuntimeError):
        trainer.score_chunk([[1]], [0])


def test_trainer_with_ensemble():
    model = EnsembleClassifier(
        n_estimators=5,
        random_state=42
    )

    trainer = StreamTrainer(model)

    trainer.fit_chunk(
        [[1], [2], [3], [4]],
        [0, 0, 1, 1]
    )

    accuracy = trainer.score_chunk(
        [[1], [4]], [0, 1]
    )

    assert 0.0 <= accuracy <= 1.0


def test_trainer_with_preprocessing():
    class DoubleFeatures:
        def __init__(self):
            self.updates = 0

        def partial_fit(self, X, y=None):
            self.updates += 1
            return self

        def transform(self, X):
            return np.asarray(X) * 2

    transformer = DoubleFeatures()

    trainer = StreamTrainer(
        DecisionTreeClassifier(),
        pipeline=transformer
    )

    trainer.fit_chunk(
        [[1], [2], [3], [4]],
        [0, 0, 1, 1]
    )

    trainer.score_chunk([[1], [4]], [0, 1])

    assert transformer.updates == 1
