
import pytest

from numcompute_stream.tree import gini_impurity


def test_gini_balanced_classes():
    assert gini_impurity([0, 0, 1, 1]) == pytest.approx(0.5)


def test_gini_single_class():
    assert gini_impurity([1, 1, 1]) == pytest.approx(0.0)


def test_gini_empty():
    assert gini_impurity([]) == pytest.approx(0.0)


from numcompute_stream.tree import entropy_impurity


def test_entropy_balanced_classes():
    assert entropy_impurity([0, 0, 1, 1]) == pytest.approx(1.0)


def test_entropy_single_class():
    assert entropy_impurity([1, 1, 1]) == pytest.approx(0.0)


def test_entropy_empty():
    assert entropy_impurity([]) == pytest.approx(0.0)
