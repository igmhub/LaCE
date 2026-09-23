"""Common cosmology-comparison figures."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _save(figure, save_path):
    if save_path is not None:
        path = Path(save_path); path.parent.mkdir(parents=True, exist_ok=True); figure.savefig(path, bbox_inches="tight")


def plot_ratio_curves(k_Mpc: np.ndarray, ratios: Mapping[str, np.ndarray], *, ylabel: str, ax=None, save_path=None):
    """Plot named dimensionless ratio curves against comoving wavenumber."""
    if ax is None: figure, ax = plt.subplots(figsize=(7, 4))
    else: figure = ax.figure
    for label, ratio in ratios.items(): ax.plot(k_Mpc, ratio, label=label)
    ax.axhline(0, color="black", lw=0.8); ax.set(xscale="log", xlabel=r"$k\ [\mathrm{Mpc}^{-1}]$", ylabel=ylabel); ax.legend(); figure.tight_layout(); _save(figure, save_path)
    return figure, ax


def plot_expansion_history(redshift: np.ndarray, hubble: Sequence[np.ndarray], labels: Sequence[str], *, reference_index: int = 0, ratio: bool = True, ax=None, save_path=None):
    """Plot H(z), or H(z) relative to a selected reference cosmology."""
    values = np.asarray(hubble); values = values / values[reference_index] if ratio else values
    if ax is None: figure, ax = plt.subplots()
    else: figure = ax.figure
    for label, value in zip(labels, values, strict=True): ax.plot(redshift, value, label=label)
    ax.set(xlabel=r"$z$", ylabel=r"$H(z)/H_0(z)$" if ratio else r"$H(z)$ [km/s/Mpc]"); ax.legend(); figure.tight_layout(); _save(figure, save_path)
    return figure, ax
