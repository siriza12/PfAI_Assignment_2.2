
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


from numcompute_stream.visualise import plot_metric_over_time


def test_plot_metric_over_time():
    fig, ax = plot_metric_over_time(
        [0.60, 0.75, 0.90],
        title="Accuracy Over Time",
        ylabel="Accuracy"
    )

    assert ax.get_title() == "Accuracy Over Time"
    assert ax.get_ylabel() == "Accuracy"
    assert len(ax.lines[0].get_ydata()) == 3

    plt.close(fig)


def test_plot_metric_over_time_rejects_empty():
    with pytest.raises(ValueError):
        plot_metric_over_time(
            [],
            title="Accuracy",
            ylabel="Score"
        )



from numcompute_stream.visualise import plot_predictions_vs_truth


def test_plot_predictions_vs_truth():
    fig, ax = plot_predictions_vs_truth(
        [0, 1, 0, 1],
        [0, 1, 1, 1]
    )

    assert ax.get_title() == "Predictions vs Ground Truth"
    assert len(ax.collections) == 2

    plt.close(fig)


def test_plot_predictions_vs_truth_rejects_mismatch():
    with pytest.raises(ValueError):
        plot_predictions_vs_truth([0, 1], [0])
