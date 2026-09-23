"""Plots for leave-one-out emulator covariance calculations."""

import matplotlib.pyplot as plt
import numpy as np


def plot_l1o_correlation(cov_zk, *, ax=None):
    """Plot the redshift-wavenumber correlation matrix."""
    if ax is None:
        _, ax = plt.subplots()

    scale = np.sqrt(np.diag(cov_zk))
    corr = cov_zk / np.outer(scale, scale)
    image = ax.imshow(corr, origin="lower", aspect="auto")
    ax.figure.colorbar(image, ax=ax, label="Correlation")
    ax.set_xlabel(r"$(z, k)$ bin")
    ax.set_ylabel(r"$(z, k)$ bin")
    return ax


def plot_l1o_errors(zz, k_Mpc, rel_diff, cov_zk, *, ax=None):
    """Plot L1O standard deviations and absolute mean biases by redshift."""
    if ax is None:
        _, ax = plt.subplots()

    standard_deviation = np.sqrt(np.diag(cov_zk)).reshape(len(zz), len(k_Mpc))
    bias = np.abs(np.mean(rel_diff, axis=0))
    for iz, z in enumerate(zz):
        line = ax.plot(k_Mpc, standard_deviation[iz], label=rf"$z={z:.2f}$")[0]
        ax.plot(k_Mpc, bias[iz], linestyle="--", color=line.get_color())

    ax.set_xscale("log")
    ax.set_xlabel(r"$k\,[\mathrm{Mpc}^{-1}]$")
    ax.set_ylabel("Relative error")
    ax.legend()
    return ax
