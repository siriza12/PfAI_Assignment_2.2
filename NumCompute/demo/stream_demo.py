
"""Demonstrate incremental classification using CSV chunks."""

import csv
import tempfile
from pathlib import Path

import numpy as np

from numcompute_stream.csv_stream import read_csv_chunks
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.stream import StreamTrainer


def create_sample_csv(path, n_samples=120):
    """Create a reproducible numeric classification dataset."""
    rng = np.random.default_rng(42)
    X = rng.uniform(-3, 3, size=(n_samples, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["feature_1", "feature_2", "target"])

        for features, label in zip(X, y):
            writer.writerow([*features, int(label)])


def run_demo(csv_path, chunk_size=20):
    """Evaluate incoming chunks, then update the classifier."""
    model = DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )
    trainer = StreamTrainer(model)

    evaluation_results = []

    for X_chunk, y_chunk in read_csv_chunks(
        csv_path, chunk_size=chunk_size
    ):
        # Evaluate before learning from the incoming chunk.
        if trainer.is_fitted:
            accuracy = trainer.score_chunk(X_chunk, y_chunk)
            evaluation_results.append(accuracy)
            print(f"Chunk accuracy: {accuracy:.2%}")

        trainer.fit_chunk(X_chunk, y_chunk)

    print(f"Training chunks processed: {trainer.chunk_count}")

    return trainer, evaluation_results


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        csv_path = Path(directory) / "sample_stream.csv"
        create_sample_csv(csv_path)
        run_demo(csv_path)
