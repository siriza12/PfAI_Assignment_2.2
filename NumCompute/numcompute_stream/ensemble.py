
"""Chunk-wise bagging ensemble for streaming classification."""

import numpy as np

from .tree import DecisionTreeClassifier


class EnsembleClassifier:
    """Combine decision trees trained on bootstrap samples of each chunk."""

    def __init__(
        self,
        n_estimators=5,
        max_depth=5,
        min_samples_split=2,
        criterion="gini",
        max_features=None,
        random_state=None,
    ):
        if (
            isinstance(n_estimators, (bool, np.bool_))
            or not isinstance(n_estimators, (int, np.integer))
            or n_estimators < 1
        ):
            raise ValueError("n_estimators must be a positive integer.")

        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state

        self._reset()

    def _reset(self):
        """Initialize independent decision trees."""
        self._rng = np.random.default_rng(self.random_state)

        seeds = self._rng.integers(
            0, 2**32 - 1, size=self.n_estimators
        )

        self.estimators_ = [
            DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion,
                max_features=self.max_features,
                random_state=int(seed),
            )
            for seed in seeds
        ]

        self.n_features_in_ = None

    def fit(self, X, y):
        """Start fresh and train the ensemble on one batch."""
        self._reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X_chunk, y_chunk):
        """Incrementally update every tree using bootstrap samples."""
        X = np.asarray(X_chunk, dtype=float)
        y = np.asarray(y_chunk)

        if (
            X.ndim != 2
            or y.ndim != 1
            or len(X) != len(y)
            or len(y) == 0
        ):
            raise ValueError(
                "X_chunk and y_chunk must have matching nonempty rows."
            )

        if X.shape[1] == 0 or not np.all(np.isfinite(X)):
            raise ValueError("X_chunk must contain finite features.")

        if (
            self.n_features_in_ is not None
            and X.shape[1] != self.n_features_in_
        ):
            raise ValueError("Feature count changed between chunks.")

        # Each tree receives its own bootstrap sample.
        for tree in self.estimators_:
            indices = self._rng.integers(
                0,
                len(y),
                size=len(y)
            )

            tree.partial_fit(
                X[indices],
                y[indices]
            )

        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        """Return predictions using majority voting."""
        if self.n_features_in_ is None:
            raise RuntimeError(
                "Train the ensemble before predicting."
            )

        X = np.asarray(X, dtype=float)

        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X has the wrong feature shape.")

        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain finite numeric values.")

        if len(X) == 0:
            return np.array([])

        # Collect predictions from every tree.
        votes = np.array([
            tree.predict(X)
            for tree in self.estimators_
        ])

        predictions = []

        # Select the most frequently predicted label.
        for column in votes.T:
            labels, counts = np.unique(
                column,
                return_counts=True
            )

            predictions.append(
                labels[np.argmax(counts)]
            )

        return np.asarray(predictions)
