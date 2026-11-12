
import pytest

from numcompute_stream.tree import gini_impurity


def test_gini_balanced_classes():
    assert gini_impurity([0, 0, 1, 1]) == pytest.approx(0.5)


def test_gini_single_class():
    assert gini_impurity([1, 1, 1]) == pytest.approx(0.0)


def test_gini_empty():
    assert gini_impurity([]) == pytest.approx(0.0)


from numcompute_stream.tree import entropy_impurity, TreeNode


def test_entropy_balanced_classes():
    assert entropy_impurity([0, 0, 1, 1]) == pytest.approx(1.0)


def test_entropy_single_class():
    assert entropy_impurity([1, 1, 1]) == pytest.approx(0.0)


def test_entropy_empty():
    assert entropy_impurity([]) == pytest.approx(0.0)


def test_tree_leaf_node():
    node = TreeNode(value=1)

    assert node.is_leaf()
    assert node.value == 1


def test_tree_split_node():
    left = TreeNode(value=0)
    right = TreeNode(value=1)

    node = TreeNode(
        feature_index=0,
        threshold=2.5,
        left=left,
        right=right
    )

    assert not node.is_leaf()
    assert node.left.value == 0
    assert node.right.value == 1


from numcompute_stream.tree import find_best_split


def test_best_split_gini():
    X = [[1], [2], [3], [4]]
    y = [0, 0, 1, 1]

    result = find_best_split(X, y, criterion="gini")

    assert result[0] == 0
    assert result[1] == pytest.approx(2.5)
    assert result[2] == pytest.approx(0.5)


def test_best_split_entropy():
    X = [[1], [2], [3], [4]]
    y = [0, 0, 1, 1]

    result = find_best_split(X, y, criterion="entropy")

    assert result[1] == pytest.approx(2.5)
    assert result[2] == pytest.approx(1.0)


def test_best_split_single_class():
    X = [[1], [2], [3]]
    y = [1, 1, 1]

    assert find_best_split(X, y) is None


def test_best_split_invalid_criterion():
    with pytest.raises(ValueError):
        find_best_split([[1], [2]], [0, 1], criterion="invalid")


from numcompute_stream.tree import DecisionTreeClassifier


def test_tree_fit_and_predict():
    X = [[1], [2], [3], [4]]
    y = [0, 0, 1, 1]

    model = DecisionTreeClassifier()
    model.fit(X, y)

    assert model.predict([[1], [4]]).tolist() == [0, 1]


def test_tree_entropy_training():
    X = [[1], [2], [3], [4]]
    y = [0, 0, 1, 1]

    model = DecisionTreeClassifier(criterion="entropy")
    model.fit(X, y)

    assert model.predict([[1], [4]]).tolist() == [0, 1]


def test_tree_max_depth_zero():
    X = [[1], [2], [3], [4]]
    y = [0, 0, 0, 1]

    model = DecisionTreeClassifier(max_depth=0)
    model.fit(X, y)

    assert model.predict([[1], [4]]).tolist() == [0, 0]


def test_tree_min_samples_split():
    X = [[1], [2], [3], [4]]
    y = [0, 0, 0, 1]

    model = DecisionTreeClassifier(min_samples_split=5)
    model.fit(X, y)

    assert model.predict([[1], [4]]).tolist() == [0, 0]


def test_tree_predict_before_training():
    model = DecisionTreeClassifier()

    with pytest.raises(RuntimeError):
        model.predict([[1]])
