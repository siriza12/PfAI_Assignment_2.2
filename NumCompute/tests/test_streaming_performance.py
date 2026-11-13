
import pytest

from benchmark.streaming_performance import compare_streaming_models
from demo.stream_demo import create_sample_csv


def test_streaming_benchmark_returns_both_models(tmp_path):
    path = tmp_path / "benchmark.csv"
    create_sample_csv(path, n_samples=40)

    results = compare_streaming_models(path, chunk_size=10)

    assert set(results) == {"single_tree", "ensemble"}

    for record in results.values():
        assert record["chunks"] == 4
        assert record["evaluated_samples"] == 30
        assert 0 <= record["prequential_accuracy"] <= 1
        assert record["fit_seconds"] >= 0
        assert record["score_seconds"] >= 0


def test_streaming_benchmark_requires_multiple_chunks(tmp_path):
    path = tmp_path / "small.csv"
    create_sample_csv(path, n_samples=8)

    with pytest.raises(ValueError):
        compare_streaming_models(path, chunk_size=20)
