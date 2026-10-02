"""Diagnostic plots for LaCE emulators."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from lace.utils.poly_p1d import PolyP1D


def _save(figure: plt.Figure, save_path: str | Path | None) -> None:
    """Save an emulator diagnostic figure when an output path is supplied.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure to serialize.
    save_path : path-like or None
        Destination path; parent directories are created as needed.
    """
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, bbox_inches="tight")


def plot_emulator_predictions(
    k_Mpc: np.ndarray,
    p1d_Mpc: np.ndarray,
    *,
    labels: Sequence[str] | None = None,
    ax: plt.Axes | None = None,
    divide_by_pi: bool = False,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot one or more emulator P1D predictions in comoving units.

    Parameters
    ----------
    k_Mpc : ndarray
        Comoving wavenumber grid in 1/Mpc.
    p1d_Mpc : ndarray
        One ``(nk,)`` spectrum or ``(nspectrum, nk)`` P1D array in Mpc.
    labels : sequence of str, optional
        Labels aligned with the plotted spectra.
    ax : matplotlib.axes.Axes, optional
        Existing axes to populate.
    divide_by_pi : bool, default=False
        Display ``k P1D / pi`` rather than ``k P1D``.
    save_path : path-like, optional
        Figure output path.

    Returns
    -------
    figure, ax : tuple
        Matplotlib figure and populated axes.

    Raises
    ------
    ValueError
        If spectra do not share the supplied wavenumber length.
    """
    spectra = np.atleast_2d(np.asarray(p1d_Mpc, dtype=float))
    k_Mpc = np.asarray(k_Mpc, dtype=float)
    if spectra.shape[1] != k_Mpc.size:
        raise ValueError("p1d_Mpc must have one value per k_Mpc")
    if ax is None:
        figure, ax = plt.subplots()
    else:
        figure = ax.figure
    normalization = np.pi if divide_by_pi else 1.0
    for index, spectrum in enumerate(spectra):
        label = None if labels is None else labels[index]
        ax.plot(k_Mpc, k_Mpc * spectrum / normalization, label=label)
    ax.set(xscale="log", yscale="log", xlabel=r"$k_\parallel$ [1/Mpc]")
    ax.set_ylabel(r"$\pi^{-1} k_\parallel P_{\rm 1D}$" if divide_by_pi else r"$k_\parallel P_{\rm 1D}$")
    if labels is not None:
        ax.legend(ncol=2)
    figure.tight_layout()
    _save(figure, save_path)
    return figure, ax


def plot_p1d_vs_emulator(
    testing_data: Sequence[dict[str, Any]], emulator: Any, *, save_path: str | Path | None = None
) -> tuple[plt.Figure, np.ndarray]:
    """Compare archive P1D spectra with emulator predictions and residuals.

    Parameters
    ----------
    testing_data : sequence of mapping
        Archive rows containing ``z``, ``k_Mpc``, ``p1d_Mpc``, and emulator
        inputs.
    emulator : object
        Object exposing ``emulate_p1d_Mpc``.
    save_path : path-like, optional
        Figure output path.

    Returns
    -------
    figure, axes : tuple
        Figure and two-panel axes for P1D and fractional residuals.
    """
    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]})
    eligible = [entry for entry in testing_data if entry.get("z", np.inf) < 4.8 and "kF_Mpc" in entry]
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(eligible)))
    for entry, color in zip(eligible, colors, strict=True):
        k = np.asarray(entry["k_Mpc"])
        power = np.asarray(entry["p1d_Mpc"])
        mask = (k > 0) & (k < 4)
        k = k[mask]
        true_power = PolyP1D(k, power[mask], kmin_Mpc=1e-3, kmax_Mpc=4, deg=5).P_Mpc(k)
        predicted_power = np.asarray(emulator.emulate_p1d_Mpc(entry, k)).reshape(-1, k.size)[0]
        label = f"$z={entry['z']:.1f}$"
        axes[0].scatter(k, k * predicted_power, color=color, marker="^", label=label)
        axes[0].plot(k, k * true_power, color=color)
        axes[1].plot(k, (true_power - predicted_power) / true_power, color=color)
    axes[0].set_ylabel(r"$k P_{\rm 1D}$")
    axes[1].set(xlabel=r"$k$ [1/Mpc]", ylabel="Relative error")
    axes[0].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    figure.tight_layout()
    _save(figure, save_path)
    return figure, axes
