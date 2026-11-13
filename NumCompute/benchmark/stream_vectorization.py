
"""Compare Python loops with NumPy vectorised operations."""

import time
import numpy as np


def loop_mean(values):
    """Calculate the mean using a Python loop."""
    total = 0.0

    for value in values:
        total += float(value)

    return total / len(values)


def vectorized_mean(values):
    """Calculate the mean using NumPy."""
    return float(np.mean(values))


def benchmark_vectorization(n_samples=100000, repeats=5):
    """Measure execution times for both approaches."""
    if n_samples < 1 or repeats < 1:
        raise ValueError("Sample count and repeats must be positive.")

    rng = np.random.default_rng(42)
    values = rng.random(n_samples)

    loop_times = []
    vector_times = []

    for _ in range(repeats):
        start = time.perf_counter()
        loop_result = loop_mean(values)
        loop_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        vector_result = vectorized_mean(values)
        vector_times.append(time.perf_counter() - start)

    if not np.isclose(loop_result, vector_result):
        raise AssertionError("Implementations produced different results.")

    loop_time = float(np.median(loop_times))
    vector_time = float(np.median(vector_times))

    return {
        "samples": n_samples,
        "loop_seconds": loop_time,
        "vectorized_seconds": vector_time,
        "speedup": loop_time / vector_time
        if vector_time > 0 else float("inf"),
    }


if __name__ == "__main__":
    results = benchmark_vectorization()

    print("Loop vs NumPy Vectorisation Benchmark")
    print(f"Samples: {results['samples']:,}")
    print(f"Loop time: {results['loop_seconds']:.6f} seconds")
    print(f"NumPy time: {results['vectorized_seconds']:.6f} seconds")
    print(f"Speedup: {results['speedup']:.2f}x")
