
import numpy as np
import pytest

from benchmark.stream_vectorization import (
    loop_mean,
    vectorized_mean,
    benchmark_vectorization,
)


def test_loop_and_vectorized_results_match():
    values = np.array([1.0, 2.0, 3.0, 4.0])

    assert loop_mean(values) == pytest.approx(2.5)
    assert vectorized_mean(values) == pytest.approx(2.5)


def test_benchmark_returns_timings():
    result = benchmark_vectorization(
        n_samples=100,
        repeats=2
    )

    assert result["samples"] == 100
    assert result["loop_seconds"] >= 0
    assert result["vectorized_seconds"] >= 0
    assert result["speedup"] > 0


def test_benchmark_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        benchmark_vectorization(n_samples=0)

    with pytest.raises(ValueError):
        benchmark_vectorization(repeats=0)
