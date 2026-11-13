
"""Benchmark a single decision tree against a bagging ensemble."""

import time

import numpy as np

from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.ensemble import EnsembleClassifier


def benchmark_model(model, X_train, y_train, X_test, y_test):
    """Measure training speed, prediction speed and accuracy."""
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start

    start = time.perf_counter()
    predictions = model.predict(X_test)
    predict_seconds = time.perf_counter() - start

    accuracy = float(np.mean(predictions == y_test))

    return {
        "train_seconds": train_seconds,
        "predict_seconds": predict_seconds,
        "accuracy": accuracy,
    }


def compare_models(n_samples=400, n_features=4, random_state=42):
    """Compare both models on the same reproducible dataset."""
    if n_samples < 20 or n_features < 1:
        raise ValueError(
            "n_samples must be at least 20 and n_features at least 1."
        )

    rng = np.random.default_rng(random_state)

    X = rng.normal(size=(n_samples, n_features))
    y = (X[:, 0] + 0.3 * X[:, -1] > 0).astype(int)

    # Keep training and testing samples separate.
    split = int(n_samples * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    tree = DecisionTreeClassifier(
        max_depth=5,
        random_state=random_state,
    )

    ensemble = EnsembleClassifier(
        n_estimators=5,
        max_depth=5,
        max_features="sqrt",
        random_state=random_state,
    )

    return {
        "single_tree": benchmark_model(
            tree, X_train, y_train, X_test, y_test
        ),
        "ensemble": benchmark_model(
            ensemble, X_train, y_train, X_test, y_test
        ),
    }


if __name__ == "__main__":
    results = compare_models()

    for name, metrics in results.items():
        print(f"\n{name.replace('_', ' ').title()}")
        print(f"Training time: {metrics['train_seconds']:.6f} seconds")
        print(f"Prediction time: {metrics['predict_seconds']:.6f} seconds")
        print(f"Accuracy: {metrics['accuracy']:.2%}")
