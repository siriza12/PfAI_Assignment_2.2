
"""Utilities for training and evaluating streaming models."""

import time
import tracemalloc

import numpy as np


class StreamTrainer:
    """Train a model chunk by chunk and record performance."""

    def __init__(self, model, pipeline=None):
        self.model = model
        self.pipeline = pipeline

        self.history = []
        self.logs = self.history

        self.chunk_count = 0
        self.train_correct = 0
        self.train_total = 0
        self.eval_correct = 0
        self.eval_total = 0
        self.is_fitted = False

        if not tracemalloc.is_tracing():
            tracemalloc.start()

    def _validate_chunk(self, X, y):
        """Check that features and labels are compatible."""
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")

        if y.ndim != 1:
            raise ValueError("y must be a one-dimensional array.")

        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of rows.")

        if len(y) == 0:
            raise ValueError("A chunk cannot be empty.")

        return X, y

    def _transform(self, X, y=None, update=False):
        """Apply an optional incremental preprocessing transformer."""
        if self.pipeline is None:
            return X

        if update:
            if not hasattr(self.pipeline, "partial_fit"):
                raise TypeError(
                    "The preprocessing pipeline needs partial_fit()."
                )
            self.pipeline.partial_fit(X, y)

        if not hasattr(self.pipeline, "transform"):
            raise TypeError(
                "The preprocessing pipeline needs transform()."
            )

        return self.pipeline.transform(X)

    def _log(self, phase, y, predictions, duration, input_bytes):
        """Save accuracy, timing and memory information."""
        correct = int(np.sum(y == predictions))
        accuracy = correct / len(y)

        if phase == "train":
            self.train_correct += correct
            self.train_total += len(y)
            cumulative = self.train_correct / self.train_total
        else:
            self.eval_correct += correct
            self.eval_total += len(y)
            cumulative = self.eval_correct / self.eval_total

        current_memory, peak_memory = (
            tracemalloc.get_traced_memory()
            if tracemalloc.is_tracing()
            else (0, 0)
        )

        record = {
            "phase": phase,
            "chunk": self.chunk_count,
            "samples": len(y),
            "accuracy": float(accuracy),
            "cumulative_accuracy": float(cumulative),
            "elapsed_seconds": float(duration),
            "input_bytes": int(input_bytes),
            "traced_memory_bytes": int(current_memory),
            "peak_traced_memory_bytes": int(peak_memory),
        }

        self.history.append(record)
        return record

    def fit_chunk(self, X, y):
        """Incrementally train on one new chunk."""
        X, y = self._validate_chunk(X, y)

        if not hasattr(self.model, "partial_fit"):
            raise TypeError("Model must support partial_fit().")

        start = time.perf_counter()

        X_processed = self._transform(X, y, update=True)

        self.model.partial_fit(X_processed, y)
        predictions = np.asarray(
            self.model.predict(X_processed)
        )

        duration = time.perf_counter() - start

        self.chunk_count += 1
        self.is_fitted = True

        self._log(
            phase="train",
            y=y,
            predictions=predictions,
            duration=duration,
            input_bytes=X.nbytes + y.nbytes,
        )

        return self

    def score_chunk(self, X, y):
        """Evaluate one chunk and return its accuracy."""
        if not self.is_fitted:
            raise RuntimeError("Train the model before scoring.")

        X, y = self._validate_chunk(X, y)

        start = time.perf_counter()

        X_processed = self._transform(
            X, update=False
        )

        predictions = np.asarray(
            self.model.predict(X_processed)
        )

        duration = time.perf_counter() - start

        record = self._log(
            phase="evaluation",
            y=y,
            predictions=predictions,
            duration=duration,
            input_bytes=X.nbytes + y.nbytes,
        )

        return record["accuracy"]

    def get_history(self):
        """Return a copy of the streaming performance records."""
        return list(self.history)
