
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
