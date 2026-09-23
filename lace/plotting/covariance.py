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


def plot_l1o_bias(zz, k_Mpc, rel_diff, cov_zk, *, ax=None):
    """Plot the mean L1O residual normalized by its covariance error."""
    if ax is None:
        _, ax = plt.subplots()

    standard_deviation = np.sqrt(np.diag(cov_zk)).reshape(
        len(zz), len(k_Mpc)
    )
    mean_bias = np.mean(rel_diff, axis=0)
    normalized_bias = np.divide(
        mean_bias,
        standard_deviation,
        out=np.zeros_like(mean_bias),
        where=standard_deviation > 0,
    )
    for iz, z in enumerate(zz):
        ax.plot(k_Mpc, normalized_bias[iz], label=rf"$z={z:.2f}$")

    ax.axhline(0, color="black", linestyle=":")
    ax.set_xscale("log")
    ax.set_xlabel(r"$k\,[\mathrm{Mpc}^{-1}]$")
    ax.set_ylabel(r"Mean bias $/\,\sigma_{\rm L1O}$")
    ax.legend()
    return ax


def plot_l1o_covariance_robustness(
    zz, k_Mpc, rel_diff, cov_zk, *, ax=None
):
    """Plot sensitivity of covariance errors to removing one simulation.

    For every simulation, the diagonal covariance error is recomputed after
    removing that simulation. The plotted value is the standard deviation of
    its fractional change relative to the full-sample covariance error.
    """
    if ax is None:
        _, ax = plt.subplots()

    residuals = np.asarray(rel_diff).reshape(rel_diff.shape[0], -1)
    if residuals.shape[0] < 3:
        raise ValueError("At least three simulations are required.")

    full_error = np.sqrt(np.diag(cov_zk))
    leave_one_out_errors = np.empty((residuals.shape[0], residuals.shape[1]))
    for simulation_index in range(residuals.shape[0]):
        keep = np.arange(residuals.shape[0]) != simulation_index
        leave_one_out_errors[simulation_index] = np.std(
            residuals[keep], axis=0, ddof=1
        )

    fractional_change = np.divide(
        leave_one_out_errors,
        full_error[None, :],
        out=np.ones_like(leave_one_out_errors),
        where=full_error[None, :] > 0,
    ) - 1
    fractional_scatter = np.std(fractional_change, axis=0).reshape(
        len(zz), len(k_Mpc)
    )

    for iz, z in enumerate(zz):
        ax.plot(k_Mpc, fractional_scatter[iz], label=rf"$z={z:.2f}$")

    ax.set_xscale("log")
    ax.set_xlabel(r"$k\,[\mathrm{Mpc}^{-1}]$")
    ax.set_ylabel(r"Std. dev. of $\sigma_{-i}/\sigma_{\rm full}-1$")
    ax.legend()
    return ax
