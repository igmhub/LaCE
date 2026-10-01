# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: lace
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Cosmic-variance diagnostic from paired MPG simulations
#
# This notebook estimates the scatter of the one-dimensional flux power across
# the three line-of-sight axes of the `mpg_central` and `mpg_seed` simulations.
# Each axis is first averaged over the paired fixed-and-paired phases, then all
# six axis estimates are compared to their joint mean.
#
# This is a diagnostic of residual cosmic variance after paired-phase
# averaging. It is not a full covariance estimate: the three axes of a box and
# the central/seed boxes are not six statistically independent universes.
# Wavenumbers are in Mpc$^{-1}$ and P1D is in Mpc.

# %%
# %load_ext autoreload
# %autoreload 2

import matplotlib.pyplot as plt
import numpy as np

from lace.archive.gadget_archive import GadgetArchive

# %% [markdown]
# ## Load the standard LaCE Gadget archive
#
# `GadgetArchive` is the normal LaCE entry point: it resolves the configured
# LaCE data directory and reads the Cabayol23 post-processing. No ForestFlow
# archive wrapper or package-relative path is needed.

# %%
archive = GadgetArchive(postproc="Cabayol23_fixp3d")
print(f"Loaded {len(archive.data)} Cabayol23 archive entries")

# %% [markdown]
# ## Build paired-phase P1D estimates
#
# Flux power is combined in flux units. For each axis, the two phase spectra
# are weighted by their squared mean fluxes before converting back using the
# mean of the two mean fluxes. This is the same estimator used historically in
# this diagnostic and avoids mixing spectra with slightly different mean flux.

# %%
def select_entries(simulation_label, z=3.0, val_scaling=1.0):
    """Select the two phases and three axes of one unrescaled snapshot."""
    entries = [
        entry
        for entry in archive.data
        if entry["sim_label"] == simulation_label
        and np.isclose(entry["z"], z)
        and np.isclose(entry["val_scaling"], val_scaling)
    ]
    if len(entries) != 6:
        raise ValueError(
            f"Expected six phase/axis entries for {simulation_label} at z={z}; "
            f"found {len(entries)}"
        )
    return entries


def paired_axis_p1d(entries):
    """Return one paired-phase P1D estimate per axis and their flux-weighted mean."""
    grouped = {}
    for entry in entries:
        grouped.setdefault(entry["ind_axis"], []).append(entry)

    axis_p1d = []
    axis_mean_flux = []
    k_Mpc = None
    for axis in sorted(grouped):
        phase_entries = sorted(grouped[axis], key=lambda entry: entry["ind_phase"])
        if len(phase_entries) != 2:
            raise ValueError(f"Axis {axis} does not contain both paired phases")
        first, second = phase_entries
        k_axis = np.asarray(first["k_Mpc"])
        if not np.array_equal(k_axis, np.asarray(second["k_Mpc"])):
            raise ValueError("The paired phases use different k grids")
        if k_Mpc is None:
            k_Mpc = k_axis
        elif not np.array_equal(k_Mpc, k_axis):
            raise ValueError("The three axes use different k grids")

        mean_flux = 0.5 * (first["mF"] + second["mF"])
        p1d = (
            first["mF"] ** 2 * np.asarray(first["p1d_Mpc"])
            + second["mF"] ** 2 * np.asarray(second["p1d_Mpc"])
        ) / (2.0 * mean_flux**2)
        axis_mean_flux.append(mean_flux)
        axis_p1d.append(p1d)

    axis_p1d = np.asarray(axis_p1d)
    axis_mean_flux = np.asarray(axis_mean_flux)
    total_mean_flux = np.mean(axis_mean_flux)
    total_p1d = np.mean(axis_mean_flux[:, None] ** 2 * axis_p1d, axis=0)
    total_p1d /= total_mean_flux**2
    return k_Mpc, axis_p1d, total_mean_flux, total_p1d

# %% [markdown]
# ## Estimate the scatter
#
# The combined spectrum is the flux-weighted mean over both simulations and
# axes. The standard deviation uses `ddof=1`, i.e. the sample scatter of the
# six paired-phase axis estimates around that combined mean.

# %%
z = 3.0
central_k_Mpc, central_axis_p1d, central_mean_flux, central_p1d = paired_axis_p1d(
    select_entries("mpg_central", z=z)
)
seed_k_Mpc, seed_axis_p1d, seed_mean_flux, seed_p1d = paired_axis_p1d(
    select_entries("mpg_seed", z=z)
)
if not np.array_equal(central_k_Mpc, seed_k_Mpc):
    raise ValueError("Central and seed simulations use different k grids")

k_Mpc = central_k_Mpc
all_axis_p1d = np.concatenate((central_axis_p1d, seed_axis_p1d))
combined_mean_flux = 0.5 * (central_mean_flux + seed_mean_flux)
combined_p1d = 0.5 * (
    central_mean_flux**2 * central_p1d + seed_mean_flux**2 * seed_p1d
) / combined_mean_flux**2
cosmic_variance_p1d = np.std(all_axis_p1d - combined_p1d, axis=0, ddof=1)
relative_cosmic_variance = cosmic_variance_p1d / combined_p1d

print(f"Six paired-phase axis estimates at z={z:.1f}")

# %%
positive_k = k_Mpc > 0
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for index, p1d in enumerate(central_axis_p1d):
    axes[0].plot(k_Mpc[positive_k], p1d[positive_k] / combined_p1d[positive_k], alpha=0.55, label=f"central axis {index}")
for index, p1d in enumerate(seed_axis_p1d):
    axes[0].plot(k_Mpc[positive_k], p1d[positive_k] / combined_p1d[positive_k], alpha=0.55, linestyle="--", label=f"seed axis {index}")
axes[0].fill_between(k_Mpc[positive_k], 1 - relative_cosmic_variance[positive_k], 1 + relative_cosmic_variance[positive_k], color="0.5", alpha=0.2, label=r"$1\sigma$ scatter")
axes[0].axhline(1.0, color="black", linewidth=1)
axes[0].set(xlim=(0.0, 5.0), xlabel=r"$k_\parallel$ [Mpc$^{-1}$]", ylabel=r"$P_{1D}/\langle P_{1D}\rangle$")
axes[0].legend(fontsize=8, ncol=2)

axes[1].plot(k_Mpc[positive_k], relative_cosmic_variance[positive_k])
axes[1].set(xlim=(0.0, 5.0), xlabel=r"$k_\parallel$ [Mpc$^{-1}$]", ylabel="relative cosmic-variance scatter")
fig.tight_layout()

# %%
