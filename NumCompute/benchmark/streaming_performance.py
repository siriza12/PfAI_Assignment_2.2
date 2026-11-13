
"""Compare single-tree and ensemble performance on CSV streams."""

import time
import tempfile
from pathlib import Path

from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.ensemble import EnsembleClassifier
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.csv_stream import read_csv_chunks
from demo.stream_demo import create_sample_csv


def compare_streaming_models(csv_path, chunk_size=20):
    """Measure prequential accuracy and chunk processing times."""
    trainers = {
        "single_tree": StreamTrainer(
            DecisionTreeClassifier(max_depth=5, random_state=42)
        ),
        "ensemble": StreamTrainer(
            EnsembleClassifier(
                n_estimators=5,
                max_depth=5,
                max_features="sqrt",
                random_state=42,
            )
        ),
    }

    results = {
        name: {
            "fit_seconds": 0.0,
            "score_seconds": 0.0,
            "evaluated_samples": 0,
            "correct_estimate": 0.0,
        }
        for name in trainers
    }

    chunk_count = 0

    for X_chunk, y_chunk in read_csv_chunks(
        csv_path, chunk_size=chunk_size
    ):
        chunk_count += 1

        for name, trainer in trainers.items():
            record = results[name]

            if trainer.is_fitted:
                start = time.perf_counter()
                accuracy = trainer.score_chunk(X_chunk, y_chunk)
                record["score_seconds"] += time.perf_counter() - start

                record["evaluated_samples"] += len(y_chunk)
                record["correct_estimate"] += accuracy * len(y_chunk)

            start = time.perf_counter()
            trainer.fit_chunk(X_chunk, y_chunk)
            record["fit_seconds"] += time.perf_counter() - start

    if chunk_count < 2:
        raise ValueError("Benchmark requires at least two CSV chunks.")

    for record in results.values():
        record["prequential_accuracy"] = (
            record["correct_estimate"] / record["evaluated_samples"]
        )
        record["chunks"] = chunk_count
        del record["correct_estimate"]

    return results


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        csv_path = Path(directory) / "benchmark_stream.csv"
        create_sample_csv(csv_path, n_samples=120)

        results = compare_streaming_models(csv_path)

        for name, result in results.items():
            print(f"\n{name.replace('_', ' ').title()}")
            print(f"Chunks: {result['chunks']}")
            print(f"Training: {result['fit_seconds']:.4f} seconds")
            print(f"Scoring: {result['score_seconds']:.4f} seconds")
            print(
                f"Prequential accuracy: "
                f"{result['prequential_accuracy']:.2%}"
            )
