
"""Test the plotting function required by Assignment 2.2."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from numcompute_stream.visualise import (
    plot_predictions_vs_ground_truth,
)


def test_plot_predictions_vs_ground_truth():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])

    fig, ax = plot_predictions_vs_ground_truth(y_true, y_pred)

    assert fig is not None
    assert ax is not None
    assert len(fig.axes) > 0

    plt.close(fig)
