
import numpy as np


def gini_impurity(y):
    """Calculate the Gini impurity of class labels."""
    labels = np.asarray(y)

    if labels.ndim != 1:
        raise ValueError("Labels must be one-dimensional.")

    if labels.size == 0:
        return 0.0

    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()

    return float(1.0 - np.sum(probabilities ** 2))


def entropy_impurity(y):
    """Calculate entropy for a group of class labels."""
    labels = np.asarray(y)

    if labels.ndim != 1:
        raise ValueError("Labels must be one-dimensional.")

    if labels.size == 0:
        return 0.0

    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()

    return float(-np.sum(probabilities * np.log2(probabilities)))


class TreeNode:
    """Represent a decision split or a leaf in a decision tree."""

    def __init__(
        self,
        feature_index=None,
        threshold=None,
        left=None,
        right=None,
        value=None
    ):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        """Return True when this node stores a prediction."""
        return self.value is not None


def find_best_split(X, y, criterion="gini", max_features=None, rng=None):
    """Find the feature and threshold with the highest impurity reduction."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError("X and y must have matching sample counts.")

    if not np.all(np.isfinite(X)):
        raise ValueError("Features must contain finite numeric values.")

    if criterion == "gini":
        impurity = gini_impurity
    elif criterion == "entropy":
        impurity = entropy_impurity
    else:
        raise ValueError("Criterion must be 'gini' or 'entropy'.")

    if len(y) < 2 or np.unique(y).size < 2:
        return None

    parent_impurity = impurity(y)
    best_gain = 0.0
    best_split = None

    n_features = X.shape[1]

    if n_features == 0:
        raise ValueError("X must contain at least one feature.")

    if max_features is None:
        feature_indices = range(n_features)
    else:
        if isinstance(max_features, str):
            if max_features == "sqrt":
                limit = max(1, int(np.sqrt(n_features)))
            elif max_features == "log2":
                limit = max(1, int(np.log2(n_features)))
            else:
                raise ValueError("Unknown max_features setting.")

        elif isinstance(max_features, (int, np.integer)) and not isinstance(
            max_features, (bool, np.bool_)
        ):
            limit = int(max_features)

        elif isinstance(max_features, (float, np.floating)):
            if not 0 < max_features <= 1:
                raise ValueError(
                    "Float max_features must be between 0 and 1."
                )
            limit = max(
                1, int(np.ceil(max_features * n_features))
            )

        else:
            raise ValueError("Invalid max_features setting.")

        if not 1 <= limit <= n_features:
            raise ValueError("max_features is outside the valid range.")

        generator = np.random.default_rng() if rng is None else rng
        feature_indices = generator.choice(
            n_features,
            size=limit,
            replace=False
        )

    for feature_index in feature_indices:
        values = np.unique(X[:, feature_index])

        for i in range(len(values) - 1):
            threshold = (values[i] + values[i + 1]) / 2

            left_mask = X[:, feature_index] <= threshold
            left_y = y[left_mask]
            right_y = y[~left_mask]

            left_weight = len(left_y) / len(y)
            right_weight = len(right_y) / len(y)

            weighted_impurity = (
                left_weight * impurity(left_y)
                + right_weight * impurity(right_y)
            )

            gain = parent_impurity - weighted_impurity

            if gain > best_gain + 1e-12:
                best_gain = gain
                best_split = (
                    feature_index,
                    float(threshold),
                    float(gain)
                )

    return best_split


class DecisionTreeClassifier:
    """A simple decision tree for classification."""

    def __init__(
        self,
        max_depth=5,
        min_samples_split=2,
        criterion="gini",
        max_features=None,
        random_state=None
    ):
        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth cannot be negative.")

        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")

        if criterion not in ("gini", "entropy"):
            raise ValueError("Criterion must be 'gini' or 'entropy'.")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self._rng = np.random.default_rng(random_state)
        self.root = None

    def fit(self, X, y):
        """Train the tree using labelled data."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError("X and y must have matching sample counts.")

        if len(y) == 0:
            raise ValueError("Training data cannot be empty.")

        self.root = self._grow_tree(X, y, depth=0)
        return self

    def _grow_tree(self, X, y, depth):
        """Build the tree recursively."""
        classes, counts = np.unique(y, return_counts=True)
        majority_class = classes[np.argmax(counts)]

        if (
            classes.size == 1
            or len(y) < self.min_samples_split
            or (
                self.max_depth is not None
                and depth >= self.max_depth
            )
        ):
            return TreeNode(value=majority_class)

        # Find a split after checking the stopping conditions.
        split = find_best_split(
            X,
            y,
            criterion=self.criterion,
            max_features=self.max_features,
            rng=self._rng
        )

        if split is None:
            return TreeNode(value=majority_class)

        feature_index, threshold, _ = split
        left_mask = X[:, feature_index] <= threshold

        left_child = self._grow_tree(
            X[left_mask],
            y[left_mask],
            depth + 1
        )

        right_child = self._grow_tree(
            X[~left_mask],
            y[~left_mask],
            depth + 1
        )

        return TreeNode(
            feature_index=feature_index,
            threshold=threshold,
            left=left_child,
            right=right_child
        )

    def predict(self, X):
        """Predict class labels for new samples."""
        if self.root is None:
            raise RuntimeError("Train the tree before predicting.")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")

        predictions = []

        for row in X:
            node = self.root

            while not node.is_leaf():
                if row[node.feature_index] <= node.threshold:
                    node = node.left
                else:
                    node = node.right

            predictions.append(node.value)

        return np.asarray(predictions)

    def partial_fit(self, X_chunk, y_chunk):
        """Update the tree using a new chunk of data."""
        X = np.asarray(X_chunk, dtype=float)
        y = np.asarray(y_chunk)

        if (
            X.ndim != 2
            or y.ndim != 1
            or len(X) != len(y)
            or len(y) == 0
        ):
            raise ValueError(
                "X_chunk and y_chunk need matching, nonempty rows."
            )

        if X.shape[1] == 0 or not np.all(np.isfinite(X)):
            raise ValueError("Features must be finite and nonempty.")

        if not hasattr(self, "_leaf_state"):
            self._leaf_state = {}
            self._stream_features = X.shape[1]

        if X.shape[1] != self._stream_features:
            raise ValueError("Feature count changed between chunks.")

        if self.root is None:
            self.root = TreeNode(value=y[0])

        self._update_stream_node(
            self.root,
            X,
            y,
            depth=0
        )

        return self

    def _update_stream_node(self, node, X, y, depth):
        """Update leaf statistics and grow new decision splits."""
        if len(y) == 0:
            return

        if not node.is_leaf():
            left = X[:, node.feature_index] <= node.threshold

            self._update_stream_node(
                node.left,
                X[left],
                y[left],
                depth + 1
            )

            self._update_stream_node(
                node.right,
                X[~left],
                y[~left],
                depth + 1
            )

            return

        state = self._leaf_state.setdefault(
            id(node),
            {
                "counts": {},
                "seen": 0,
                "X": np.empty((0, X.shape[1])),
                "y": np.empty(0, dtype=y.dtype),
            }
        )

        classes, counts = np.unique(y, return_counts=True)

        for label, count in zip(classes, counts):
            state["counts"][label] = (
                state["counts"].get(label, 0) + int(count)
            )

        # Update prediction from all observed class counts.
        node.value = min(
            state["counts"],
            key=lambda label: (-state["counts"][label], label)
        )

        state["seen"] += len(y)

        # Retain at most 256 recent samples per leaf.
        state["X"] = np.concatenate(
            (state["X"], X)
        )[-256:]

        state["y"] = np.concatenate(
            (state["y"], y)
        )[-256:]

        if (
            state["seen"] < self.min_samples_split
            or (
                self.max_depth is not None
                and depth >= self.max_depth
            )
        ):
            return

        # Find the split after the stopping conditions.
        split = find_best_split(
            state["X"],
            state["y"],
            criterion=self.criterion,
            max_features=self.max_features,
            rng=self._rng
        )

        if split is None:
            return

        feature_index, threshold, _ = split
        left_mask = (
            state["X"][:, feature_index] <= threshold
        )

        node.feature_index = feature_index
        node.threshold = threshold
        node.left = TreeNode()
        node.right = TreeNode()

        for child, mask in (
            (node.left, left_mask),
            (node.right, ~left_mask)
        ):
            child_y = state["y"][mask]

            child_classes, child_counts = np.unique(
                child_y,
                return_counts=True
            )

            child.value = child_classes[
                np.argmax(child_counts)
            ]

            self._leaf_state[id(child)] = {
                "counts": dict(
                    zip(
                        child_classes,
                        map(int, child_counts)
                    )
                ),
                "seen": len(child_y),
                "X": state["X"][mask].copy(),
                "y": child_y.copy(),
            }

        # The leaf has now become a decision split.
        node.value = None
        del self._leaf_state[id(node)]
