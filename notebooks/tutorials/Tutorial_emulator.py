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
# # Tutorial of LaCE emulators
#
# We have multiple emulators with different architectures and trained on different simulations. The preferred ones are:
#
# - Most calculations: CH24_mpgcen_gpr trained on the MP-Gadget simulations. Default emulator in P1D analysis of DESI DR1 data (Chaves-Montero+2026)
#
# - Small scales: CH24_nyxcen_gpr trained on the Lyssa simulations. Alternative emulator in P1D analysis of DESI DR1 data (Chaves-Montero+2026)
#
# We detail how to use the CH24_mpgcen_gpr emulator below, but both work in the same way

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
from lace.emulator.emulator_manager import set_emulator
from lace.archive import gadget_archive
from lace.plotting import plot_emulator_predictions, plot_parameter_corner

# %% [markdown]
# #### Load mpg simulations (Pedersen+21; Cabayol-Garcia+23)

# %%
archive = gadget_archive.GadgetArchive(postproc="Cabayol23")

# %% [markdown]
# #### Load Emulator

# %%
emulator = set_emulator("CH24_mpgcen_gpr")

# %% [markdown]
# ### Evaluate the emulator

# %% [markdown]
#  #### For the parameters of the snapshots of the mpg-central simulation

# %%
# get data
testing_data = archive.get_testing_data("mpg_central")

# set k values and get p1d
k_Mpc = np.geomspace(0.1, 4, 100)
p1d_all = []
for snap in testing_data:
    # evaluate the emulator
    p1d = emulator.emulate_p1d_Mpc(snap, k_Mpc)
    p1d_all.append(p1d[0])

# %%
plot_emulator_predictions(
    k_Mpc,
    np.asarray(p1d_all),
    labels=[f"z={entry['z']:.2f}" for entry in testing_data],
    divide_by_pi=True,
)

# %% [markdown]
# ### For some random parameters
#
# For a better performance, check that these are within the range of values used to train the emulator

# %%
k_Mpc = np.geomspace(0.1, 4, 100)
# we only change Delta2_p, the rest of the parameters are fixed to the central values of the training set
input_params = {
    'Delta2_p': [0.30, 0.35, 0.40],
    'n_p': [-2.3, -2.3, -2.3],
    'alpha_p': [-0.215, -0.215, -0.215],
    'mF': [0.66, 0.66, 0.66],
    'gamma': [1.5, 1.5, 1.5],
    'sigT_Mpc': [0.128, 0.128, 0.128],
    'kF_Mpc': [10.5, 10.5, 10.5]
}
p1d = emulator.emulate_p1d_Mpc(input_params, k_Mpc)

plot_emulator_predictions(k_Mpc, p1d, divide_by_pi=True)

# %% [markdown]
# To check the range of values of the parameters in the training set, we can do:

# %%
emu_params = ['Delta2_p', 'n_p', 'mF', 'sigT_Mpc', 'gamma', 'kF_Mpc']
training_data = archive.get_training_data(emu_params, average="both")

# %% [markdown]
# Plot with all simulations in the training set

# %%
# Build numpy array with shape (n_samples, n_params)
samples = np.array([
    [sample[p] for p in emu_params]
    for sample in training_data
])

plot_parameter_corner(samples, labels=emu_params)

# %%
