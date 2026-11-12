
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
