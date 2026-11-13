
"""End-to-end integration tests for streaming classification."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from demo.stream_demo import create_sample_csv
from numcompute.io import load_csv_chunks
from numcompute.pipeline import Pipeline
from numcompute.preprocessing import StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.visualise import (
    plot_metric_over_time,
    save_plot,
)


def test_csv_to_streaming_trainer(tmp_path):
    """Check CSV reading, chunk training and evaluation together."""
    csv_path = tmp_path / "stream.csv"
    create_sample_csv(csv_path, n_samples=50)

    trainer = StreamTrainer(
        DecisionTreeClassifier(max_depth=5, random_state=42)
    )

    evaluated = 0

    for X_chunk, y_chunk in load_csv_chunks(
        csv_path, chunk_size=10
    ):
        if trainer.is_fitted:
            accuracy = trainer.score_chunk(X_chunk, y_chunk)
            assert 0.0 <= accuracy <= 1.0
            evaluated += 1

        trainer.fit_chunk(X_chunk, y_chunk)

    assert trainer.chunk_count == 5
    assert evaluated == 4
    assert len(trainer.get_history()) == 9


def test_pipeline_and_graph_export(tmp_path):
    """Check preprocessing, classification and plot saving."""
    csv_path = tmp_path / "data.csv"
    create_sample_csv(csv_path, n_samples=40)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("tree", DecisionTreeClassifier(random_state=42)),
    ])

    accuracies = []

    for index, (X_chunk, y_chunk) in enumerate(
        load_csv_chunks(csv_path, chunk_size=10)
    ):
        if index > 0:
            predictions = pipeline.predict(X_chunk)
            accuracies.append(
                float(np.mean(predictions == y_chunk))
            )

        pipeline.partial_fit(X_chunk, y_chunk)

    assert len(accuracies) == 3

    fig, ax = plot_metric_over_time(
        accuracies,
        title="Streaming Accuracy",
        ylabel="Accuracy",
    )

    output = save_plot(
        fig, tmp_path / "figures" / "accuracy.png"
    )

    assert output.exists()
    assert output.stat().st_size > 0

    plt.close(fig)


def test_csv_rejects_malformed_rows(tmp_path):
    """Ensure invalid CSV rows are rejected clearly."""
    csv_path = tmp_path / "invalid.csv"
    csv_path.write_text(
        "feature,target\n1,0\n2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        list(load_csv_chunks(csv_path, chunk_size=2))
