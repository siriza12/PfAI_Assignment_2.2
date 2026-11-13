
"""Visualisations for streaming model performance."""

import matplotlib.pyplot as plt


def plot_accuracy(history):
    """Plot evaluation accuracy across processed chunks."""
    records = [
        record for record in history
        if record["phase"] == "evaluation"
    ]

    if not records:
        raise ValueError("No evaluation records available.")

    chunks = [record["chunk"] for record in records]
    accuracy = [record["accuracy"] for record in records]
    cumulative = [
        record["cumulative_accuracy"] for record in records
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(chunks, accuracy, marker="o", label="Chunk accuracy")
    ax.plot(chunks, cumulative, label="Cumulative accuracy")

    ax.set_xlabel("Training chunks processed")
    ax.set_ylabel("Accuracy")
    ax.set_title("Streaming Classification Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, ax


def plot_memory_usage(history):
    """Plot peak traced memory usage across streaming chunks."""
    records = [
        record for record in history
        if record["phase"] == "train"
    ]

    if not records:
        raise ValueError("No training records available.")

    chunks = [record["chunk"] for record in records]
    peak_memory_mb = [
        record["peak_traced_memory_bytes"] / (1024 ** 2)
        for record in records
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        chunks,
        peak_memory_mb,
        marker="o",
        label="Peak traced memory"
    )

    ax.set_xlabel("Training chunks processed")
    ax.set_ylabel("Peak traced memory (MB)")
    ax.set_title("Streaming Model Memory Usage")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, ax


def plot_model_comparison(tree_history, ensemble_history):
    """Compare evaluation accuracy for a tree and an ensemble."""

    def extract_accuracy(history):
        records = [
            record for record in history
            if record["phase"] == "evaluation"
        ]

        if not records:
            raise ValueError(
                "Both models need evaluation history."
            )

        chunks = [record["chunk"] for record in records]
        accuracy = [record["accuracy"] for record in records]

        return chunks, accuracy

    tree_chunks, tree_accuracy = extract_accuracy(tree_history)
    ensemble_chunks, ensemble_accuracy = extract_accuracy(
        ensemble_history
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        tree_chunks,
        tree_accuracy,
        marker="o",
        label="Single Decision Tree"
    )

    ax.plot(
        ensemble_chunks,
        ensemble_accuracy,
        marker="s",
        label="Bagging Ensemble"
    )

    ax.set_xlabel("Training chunks processed")
    ax.set_ylabel("Evaluation accuracy")
    ax.set_title("Single Tree vs Ensemble Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, ax


def plot_metric_over_time(metric_values, title, ylabel):
    """Plot a numerical metric across successive streaming chunks."""
    import numpy as np

    values = np.asarray(metric_values, dtype=float)

    if values.ndim != 1 or values.size == 0:
        raise ValueError("metric_values must be a nonempty 1D sequence.")

    if not np.all(np.isfinite(values)):
        raise ValueError("Metric values must be finite.")

    chunks = np.arange(1, len(values) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(chunks, values, marker="o", label=ylabel)

    ax.set_xlabel("Chunk number")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    return fig, ax


def plot_predictions_vs_truth(y_true, y_pred):
    """Compare predicted class labels with actual labels."""
    import numpy as np

    actual = np.asarray(y_true)
    predicted = np.asarray(y_pred)

    if (
        actual.ndim != 1
        or predicted.ndim != 1
        or len(actual) != len(predicted)
        or len(actual) == 0
    ):
        raise ValueError("Labels must be matching nonempty 1D arrays.")

    # Convert labels into positions for plotting.
    labels = np.unique(np.concatenate((actual, predicted)))
    actual_positions = np.searchsorted(labels, actual)
    predicted_positions = np.searchsorted(labels, predicted)

    fig, ax = plt.subplots(figsize=(9, 5))
    samples = np.arange(len(actual))

    ax.scatter(
        samples,
        actual_positions,
        label="Ground Truth",
        marker="o",
        alpha=0.7
    )

    ax.scatter(
        samples,
        predicted_positions,
        label="Predictions",
        marker="x",
        alpha=0.7
    )

    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Class Label")
    ax.set_title("Predictions vs Ground Truth")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([str(label) for label in labels])
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, ax


def compare_models(metric1, metric2, labels):
    """Compare two models using metric values across chunks."""
    import numpy as np

    first = np.asarray(metric1, dtype=float)
    second = np.asarray(metric2, dtype=float)

    if first.ndim != 1 or second.ndim != 1:
        raise ValueError("Model metrics must be one-dimensional.")

    if len(first) == 0 or len(first) != len(second):
        raise ValueError("Both models need equal, nonempty metrics.")

    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("Metric values must be finite.")

    if len(labels) != 2:
        raise ValueError("Provide exactly two model labels.")

    chunks = np.arange(1, len(first) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(chunks, first, marker="o", label=labels[0])
    ax.plot(chunks, second, marker="s", label=labels[1])

    ax.set_xlabel("Chunk Number")
    ax.set_ylabel("Metric Value")
    ax.set_title("Streaming Model Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, ax



def save_plot(fig, file_path, dpi=150):
    """Save a Matplotlib figure to PNG, PDF or SVG."""
    from pathlib import Path

    if not hasattr(fig, "savefig"):
        raise TypeError("Expected a Matplotlib figure.")

    path = Path(file_path)

    if path.suffix.lower() not in (".png", ".pdf", ".svg"):
        raise ValueError("Supported formats are PNG, PDF and SVG.")

    if isinstance(dpi, bool) or not isinstance(dpi, (int, float)) or dpi <= 0:
        raise ValueError("dpi must be a positive number.")

    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight"
    )

    return path
