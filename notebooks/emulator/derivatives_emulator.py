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
# # Response of the P1D emulator to its input parameters
#
# This notebook varies one emulator parameter at a time around the central
# simulation at $z=3$. It compares each prediction with the P1D at the central
# point, while holding all other inputs fixed. Both the MPG and Nyx central
# emulators are supported without loading their simulation archives.

# %%
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2
import matplotlib.pyplot as plt
import numpy as np

from lace.emulator.emulator_manager import set_emulator
from lace.emulator.central_parameters import get_central_parameters_z3

# %% [markdown]
# ## Select the emulator and central point
#
# Change only `emulator_label` to switch simulation suites. The corresponding
# central-simulation parameters at $z=3$ are stored in
# `lace.emulator.central_parameters`, so rerunning this notebook does not load
# an archive. The linear-power and P1D input parameters are dimensionless except
# for `sigT_Mpc` (Mpc) and `kF_Mpc` (1/Mpc).

# %%
emulator_label = "CH24_mpgcen_gpr"
# emulator_label = "CH24_nyxcen_gpr"

fiducial_parameters = get_central_parameters_z3(emulator_label)
emulator = set_emulator(emulator_label)

# %% [markdown]
# ## Define the wavenumbers and parameter variations
#
# We evaluate 100 logarithmically spaced comoving wavenumbers from
# $0.05$ to $5\,\mathrm{Mpc}^{-1}$. The step sizes below give a visible,
# symmetric response around the central point and can be edited independently.
# Nyx additionally varies `alpha_p`, which is an input of its emulator but not
# of the MPG central emulator.

# %%
len_max = 100
k_Mpc = np.logspace(np.log10(0.05), np.log10(5.0), len_max)

parameter_steps = {
    "Delta2_p": 0.05,
    "n_p": 0.05,
    "mF": 0.05,
    "gamma": 0.10,
    "sigT_Mpc": 0.02,
    "kF_Mpc": 2.0,
}
if emulator_label == "CH24_nyxcen_gpr":
    parameter_steps["alpha_p"] = 0.02

# %% [markdown]
# ## Compute one-at-a-time P1D responses
#
# For each parameter we evaluate a lower and an upper value while every other
# input remains at the central value. We store the fractional difference
# $P_\mathrm{1D}/P_\mathrm{1D}^{\mathrm{central}}-1$, which makes the scale
# and sign of each response easy to compare.

# %%
fiducial_p1d = np.asarray(
    emulator.emulate_p1d_Mpc(fiducial_parameters, k_Mpc)
).squeeze()

responses = {}
for parameter_name, step in parameter_steps.items():
    varied_values = np.array(
        [
            fiducial_parameters[parameter_name] - step,
            fiducial_parameters[parameter_name] + step,
        ]
    )
    relative_differences = []
    for varied_value in varied_values:
        varied_parameters = fiducial_parameters.copy()
        varied_parameters[parameter_name] = varied_value
        varied_p1d = np.asarray(
            emulator.emulate_p1d_Mpc(varied_parameters, k_Mpc)
        ).squeeze()
        relative_differences.append(varied_p1d / fiducial_p1d - 1.0)
    responses[parameter_name] = {
        "values": varied_values,
        "relative_difference": np.asarray(relative_differences),
    }

# %% [markdown]
# ## Plot the parameter responses
#
# Each panel varies only the parameter named in its title. The dotted line is
# the unchanged central prediction. Curve labels give the absolute parameter
# values used, so the MPG and Nyx versions can be compared without assuming
# identical central points.

# %%
parameter_labels = {
    "Delta2_p": r"$\Delta_p^2$",
    "n_p": r"$n_p$",
    "alpha_p": r"$\alpha_p$",
    "mF": r"$\bar{F}$",
    "gamma": r"$\gamma$",
    "sigT_Mpc": r"$\sigma_T\,[\mathrm{Mpc}]$",
    "kF_Mpc": r"$k_F\,[\mathrm{Mpc}^{-1}]$",
}

number_of_parameters = len(responses)
number_of_columns = 3
number_of_rows = int(np.ceil(number_of_parameters / number_of_columns))
figure, axes = plt.subplots(
    number_of_rows,
    number_of_columns,
    figsize=(15, 4 * number_of_rows),
    sharex=True,
    squeeze=False,
)
for axis, (parameter_name, response) in zip(
    axes.flat[:number_of_parameters], responses.items(), strict=True
):
    for varied_value, relative_difference in zip(
        response["values"], response["relative_difference"], strict=True
    ):
        axis.plot(
            k_Mpc,
            relative_difference,
            label=f"{parameter_labels[parameter_name]} = {varied_value:.4g}",
        )
    axis.axhline(0.0, color="black", linestyle=":")
    axis.set_xscale("log")
    axis.set_title(parameter_labels[parameter_name])
    axis.legend(fontsize=9)

for axis in axes.flat[number_of_parameters:]:
    axis.set_visible(False)

for axis in axes[-1]:
    axis.set_xlabel(r"$k_\parallel\,[\mathrm{Mpc}^{-1}]$")
for axis in axes[:, 0]:
    axis.set_ylabel(r"$P_\mathrm{1D}/P_\mathrm{1D}^{\mathrm{central}}-1$")

figure.suptitle(f"P1D parameter responses: {emulator_label}")
figure.tight_layout()

# %%
