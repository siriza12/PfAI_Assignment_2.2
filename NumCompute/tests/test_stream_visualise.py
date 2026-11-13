
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from numcompute_stream.visualise import plot_accuracy


def test_plot_accuracy():
    history = [
        {
            "phase": "evaluation",
            "chunk": 1,
            "accuracy": 0.7,
            "cumulative_accuracy": 0.7,
        },
        {
            "phase": "evaluation",
            "chunk": 2,
            "accuracy": 0.9,
            "cumulative_accuracy": 0.8,
        },
    ]

    fig, ax = plot_accuracy(history)

    assert len(ax.lines) == 2
    assert len(ax.lines[0].get_xdata()) == 2

    plt.close(fig)


def test_plot_accuracy_requires_evaluation():
    with pytest.raises(ValueError):
        plot_accuracy([])


from numcompute_stream.visualise import plot_memory_usage

def test_plot_memory_usage():
    history = [
        {
            "phase": "train",
            "chunk": 1,
            "peak_traced_memory_bytes": 1048576,
        },
        {
            "phase": "train",
            "chunk": 2,
            "peak_traced_memory_bytes": 2097152,
        },
    ]

    fig, ax = plot_memory_usage(history)

    assert len(ax.lines) == 1
    assert list(ax.lines[0].get_ydata()) == [1.0, 2.0]

    plt.close(fig)

def test_plot_memory_usage_requires_training():
    with pytest.raises(ValueError):
        plot_memory_usage([])


from numcompute_stream.visualise import plot_model_comparison


def test_plot_model_comparison():
    tree_history = [
        {"phase": "evaluation", "chunk": 1, "accuracy": 0.70},
        {"phase": "evaluation", "chunk": 2, "accuracy": 0.75},
    ]

    ensemble_history = [
        {"phase": "evaluation", "chunk": 1, "accuracy": 0.80},
        {"phase": "evaluation", "chunk": 2, "accuracy": 0.85},
    ]

    fig, ax = plot_model_comparison(
        tree_history, ensemble_history
    )

    assert len(ax.lines) == 2
    assert ax.get_ylabel() == "Evaluation accuracy"

    plt.close(fig)


def test_plot_model_comparison_requires_data():
    tree_history = [
        {"phase": "evaluation", "chunk": 1, "accuracy": 0.8}
    ]

    with pytest.raises(ValueError):
        plot_model_comparison(tree_history, [])
