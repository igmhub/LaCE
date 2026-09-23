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
# # Precision emulators
#
# Notebook to validate the precision of the different LaCE emulators.

# %%
# %load_ext autoreload
# %autoreload 2
from lace.archive import gadget_archive, nyx_archive
from lace.emulator.gp_emulator_multi import GPEmulator
from lace.plotting import EmulatorPrecisionPlotter, complete_igm_parameters

from matplotlib import rcParams

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

# %% [markdown]
# ## Load emulator and data
#
# Select the emulator once here. Its label determines the matching simulation
# suite and therefore the central/seed samples and training-data convention.

# %%
# emulator_label = "CH24_mpgcen_gpr"
emulator_label = "CH24_nyxcen_gpr"
emu_params = ["Delta2_p", "n_p", "mF", "sigT_Mpc", "gamma", "kF_Mpc"]

if emulator_label == "CH24_mpgcen_gpr":
    archive = gadget_archive.GadgetArchive()
    central = archive.get_testing_data("mpg_central")
    seed = archive.get_testing_data("mpg_seed")
    training_data = archive.get_training_data(emu_params=emu_params, average="both")
elif emulator_label == "CH24_nyxcen_gpr":
    archive = nyx_archive.NyxArchive()
    central = archive.get_testing_data("nyx_central")
    seed = archive.get_testing_data("nyx_seed")
    training_data = archive.get_training_data(emu_params=emu_params)
else:
    raise ValueError("Precision_emulators supports CH24_mpgcen_gpr or CH24_nyxcen_gpr.")

emulator = GPEmulator(emulator_label=emulator_label)

# %% [markdown]
# ## Testing-simulation precision
#
# The seed simulation is not used to train the central emulator. Each curve
# shows $P_\mathrm{1D}^\mathrm{smooth}/P_\mathrm{1D}^\mathrm{emu}-1$ at one
# redshift. We explicitly complete any missing seed IGM quantities from the
# matched central sample before plotting, and print every replacement.

# %%
precision_plotter = EmulatorPrecisionPlotter(
    emulator, archive=archive, emulator_label=emulator_label
)
seed_for_validation = complete_igm_parameters(seed, central)
_, _, testing_plot_data = precision_plotter.plot_testing_precision(
    seed_for_validation,
    kmax_Mpc=4.0,
    save_zenodo=False,
    # zenodo_filename="fig_4a.npy",
)

# %% [markdown]
# ## Accuracy of the smoothing model
#
# The bands show percentiles of
# $P_\mathrm{1D}^\mathrm{sim}/P_\mathrm{1D}^\mathrm{smooth}-1$ across the
# training samples, isolating the polynomial-smoothing approximation.

# %%
_, _, smoothing_plot_data = precision_plotter.plot_smoothing_precision(
    training_data,
    kmax_Mpc=4.0,
    save_zenodo=False,
    # zenodo_filename="fig_B2a.npy",
)

# %% [markdown]
# ## Leave-one-out precision
#
# One emulator is trained without each simulation and compared with that
# simulation. The bands show $P_\mathrm{1D}^\mathrm{emu}/P_\mathrm{1D}^
# \mathrm{smooth}-1$. The Nyx calculation keeps its historical stopping point.

# %%
leave_one_out_stop = "nyx_14" if emulator_label.startswith("CH24_nyx") else None

_, _, leave_one_out_plot_data = precision_plotter.plot_leave_one_out_precision(
    testing_prefix="nyx" if emulator_label.startswith("CH24_nyx") else "mpg",
    stop_simulation=leave_one_out_stop,
    save_zenodo=False,
    # zenodo_filename="fig_B2b.npy",
)

# %%
