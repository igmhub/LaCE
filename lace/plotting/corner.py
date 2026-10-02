"""Corner plots for parameter samples."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_parameter_corner(
    samples: np.ndarray | Sequence[np.ndarray],
    *,
    labels: Sequence[str],
    dataset_labels: Sequence[str] | None = None,
    colors: Sequence[str] | None = None,
    truth_values: Sequence[float] | None = None,
    save_path: str | Path | None = None,
):
    """Create a corner plot for one or more posterior sample arrays.

    Parameters
    ----------
    samples : ndarray or sequence of ndarray
        One or more arrays with shape ``(n_samples, n_parameters)``.
    labels : sequence of str
        Parameter labels, one per sample column.
    dataset_labels, colors : sequence of str, optional
        Legend labels and plotting colors for the sample sets.
    truth_values : sequence of float, optional
        Reference values drawn on diagonal panels.
    save_path : path-like, optional
        Figure output path.

    Returns
    -------
    figure, axes : tuple
        Corner figure and ``(n_parameters, n_parameters)`` axes grid.

    Raises
    ------
    ValueError
        If sample shapes or color counts do not match supplied labels.
    """
    import corner

    arrays = [np.asarray(samples)] if isinstance(samples, np.ndarray) else [np.asarray(item) for item in samples]
    if any(array.ndim != 2 or array.shape[1] != len(labels) for array in arrays):
        raise ValueError("Each sample array must have shape (n_samples, len(labels))")
    colors = list(colors or [f"C{index}" for index in range(len(arrays))])
    if len(colors) != len(arrays):
        raise ValueError("colors must match the number of sample arrays")
    figure = corner.corner(arrays[0], labels=labels, color=colors[0], plot_density=False, plot_datapoints=False, fill_contours=False, levels=(0.68, 0.95), smooth=1.0, hist_kwargs={"density": True})
    for array, color in zip(arrays[1:], colors[1:], strict=True):
        corner.corner(array, fig=figure, color=color, plot_density=False, plot_datapoints=False, fill_contours=False, levels=(0.68, 0.95), smooth=1.0, hist_kwargs={"density": True})
    axes = np.asarray(figure.axes).reshape(len(labels), len(labels))
    if truth_values is not None:
        for axis, value in zip(np.diag(axes), truth_values, strict=True):
            axis.axvline(value, color="black")
    if dataset_labels is not None:
        handles = [plt.Line2D([], [], color=color, label=label) for color, label in zip(colors, dataset_labels, strict=True)]
        figure.legend(handles=handles, loc="upper right")
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, bbox_inches="tight")
    return figure, axes
