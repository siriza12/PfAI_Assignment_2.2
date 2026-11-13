
import pytest

from benchmark.stream_model_comparison import compare_models


def test_comparison_returns_both_models():
    results = compare_models(n_samples=40)

    assert "single_tree" in results
    assert "ensemble" in results


def test_comparison_returns_valid_metrics():
    results = compare_models(n_samples=40)

    for model_results in results.values():
        assert model_results["train_seconds"] >= 0
        assert model_results["predict_seconds"] >= 0
        assert 0 <= model_results["accuracy"] <= 1


def test_comparison_rejects_invalid_dataset_size():
    with pytest.raises(ValueError):
        compare_models(n_samples=10)
