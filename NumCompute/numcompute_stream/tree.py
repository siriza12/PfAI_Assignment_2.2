
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


def find_best_split(X, y, criterion="gini"):
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

    for feature_index in range(X.shape[1]):
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

    def __init__(self, max_depth=5, min_samples_split=2, criterion="gini"):
        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth cannot be negative.")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")
        if criterion not in ("gini", "entropy"):
            raise ValueError("Criterion must be 'gini' or 'entropy'.")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
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
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return TreeNode(value=majority_class)

        split = find_best_split(X, y, criterion=self.criterion)

        if split is None:
            return TreeNode(value=majority_class)

        feature_index, threshold, _ = split
        left_mask = X[:, feature_index] <= threshold

        left_child = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._grow_tree(X[~left_mask], y[~left_mask], depth + 1)

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
