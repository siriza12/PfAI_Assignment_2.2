
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
