"""Common cosmology-comparison figures."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _save(figure, save_path):
    """Save a figure only when a destination path is requested.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure to serialize.
    save_path : path-like or None
        Output path; parent directories are created when provided.
    """
    if save_path is not None:
        path = Path(save_path); path.parent.mkdir(parents=True, exist_ok=True); figure.savefig(path, bbox_inches="tight")


def plot_ratio_curves(k_Mpc: np.ndarray, ratios: Mapping[str, np.ndarray], *, ylabel: str, ax=None, save_path=None):
    """Plot named dimensionless ratio curves against comoving wavenumber.

    Parameters
    ----------
    k_Mpc : ndarray
        Shared comoving wavenumber grid in 1/Mpc.
    ratios : mapping of str to ndarray
        Dimensionless curves aligned with ``k_Mpc``.
    ylabel : str
        Vertical-axis label.
    ax : matplotlib.axes.Axes, optional
        Existing axes to populate.
    save_path : path-like, optional
        Figure output path.

    Returns
    -------
    figure, ax : tuple
        Matplotlib figure and populated axes.
    """
    if ax is None: figure, ax = plt.subplots(figsize=(7, 4))
    else: figure = ax.figure
    for label, ratio in ratios.items(): ax.plot(k_Mpc, ratio, label=label)
    ax.axhline(0, color="black", lw=0.8); ax.set(xscale="log", xlabel=r"$k\ [\mathrm{Mpc}^{-1}]$", ylabel=ylabel); ax.legend(); figure.tight_layout(); _save(figure, save_path)
    return figure, ax


def plot_expansion_history(redshift: np.ndarray, hubble: Sequence[np.ndarray], labels: Sequence[str], *, reference_index: int = 0, ratio: bool = True, ax=None, save_path=None):
    """Plot Hubble histories or ratios to one reference cosmology.

    Parameters
    ----------
    redshift : ndarray
        Redshift grid.
    hubble : sequence of ndarray
        Hubble-rate curves in km/s/Mpc, each aligned with ``redshift``.
    labels : sequence of str
        Curve labels.
    reference_index : int, default=0
        Reference curve used when ``ratio`` is true.
    ratio : bool, default=True
        Plot dimensionless Hubble ratios instead of absolute rates.
    ax : matplotlib.axes.Axes, optional
        Existing axes to populate.
    save_path : path-like, optional
        Figure output path.

    Returns
    -------
    figure, ax : tuple
        Matplotlib figure and populated axes.
    """
    values = np.asarray(hubble); values = values / values[reference_index] if ratio else values
    if ax is None: figure, ax = plt.subplots()
    else: figure = ax.figure
    for label, value in zip(labels, values, strict=True): ax.plot(redshift, value, label=label)
    ax.set(xlabel=r"$z$", ylabel=r"$H(z)/H_0(z)$" if ratio else r"$H(z)$ [km/s/Mpc]"); ax.legend(); figure.tight_layout(); _save(figure, save_path)
    return figure, ax
