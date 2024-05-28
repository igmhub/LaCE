# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: emulators2
#     language: python
#     name: emulators2
# ---

# %% [markdown]
# # THE LaCE EMULATORS WITH GADGET AND NYX SIMULATIONS

# %% [markdown]
# A new version of the LaCE emulators including Nyx simulations is now available in igmhub/LaCE: https://github.com/igmhub/LaCE
#

# %%
import os, sys
import numpy as np
from matplotlib import pyplot as plt

# %%
# load LaCE-related modules
from lace.archive import gadget_archive
from lace.archive import nyx_archive
from lace.emulator.nn_emulator import NNEmulator
from lace.emulator.gp_emulator import GPEmulator
from lace.utils import poly_p1d

# %% [markdown]
# ## GENERAL USAGE

# %% [markdown]
# There are two emulator objects:
# - GPEmulator(**args)
# - NNEmulator(**args)
#
# And for both, we can call them specifying:
# - A custom archive that we have externally generated
# - A training set label (training_set), pointing the emulator to a pre-defined training set.
#
# There is also the option of defining a pre-defined emulator with the same configuration as the emulator used in a given publication

# %% [markdown]
# ## EXAMPLES

# %% [markdown]
# ### A. HOW TO CREATE AN ARCHIVE

# %%
# Gadget archive with the post-processing using in Pedersen21
mpg_arch_P21 = gadget_archive.GadgetArchive(postproc="Pedersen21")

# %% [markdown]
# In the post-processing in Pedersen21, the P1D is measured along one axis, while in Cabayol23, we measure the P1D along the three axes.

# %%
# Gadget archive with the post-processing using in Cabayol23
mpg_arch_C23 = gadget_archive.GadgetArchive(postproc="Cabayol23")

# %%
# Nyx archive provided by Solene Chabanier (this function takes a while, so you can limit the z range)
nyx_arch = nyx_archive.NyxArchive(verbose=False)

# %% [markdown]
# ### B. HOW TO CREATE AN EMULATOR

# %% [markdown]
# When calling an emulator, we can choose the settings, or ask for a pre-defined emulator's configuration.
# There are three possibilities:
#
#     - Pedersen21: Configuration used in Pedersen21. kmax=3, kbin emulator (available for GP).
#     - Pedersen23: Configuration used in Pedersen23. kmax=3, polyfit emulator (available for GP).
#     - Cabayol23: Configuration used in Cabayol-Garcia 2023. kmax=4, polyfit emulator (available for NN).

# %% [markdown]
# This creates a GP emulator with the default configuration from Pedersen21

# %%
gp_emu_P21 = GPEmulator(archive=mpg_arch_P21,emulator_label='Pedersen21')

# %% [markdown]
# This creates a GP emulator with the default configuration from Pedersen23 (same archive, different settings)

# %%
gp_emu_P23 = GPEmulator(archive=mpg_arch_P21,emulator_label='Pedersen23')

# %% [markdown]
# This creates a NN emulator with the default configuration of Cabayol23

# %% jupyter={"outputs_hidden": true}
nn_emu_C23 = NNEmulator(archive=mpg_arch_C23, emulator_label='Cabayol23')

# %% [markdown]
# This creates a NN emulator with settings similar to those used in Cabayol23

# %%
nn_emu_C23plus = NNEmulator(archive=mpg_arch_C23, emulator_label='Cabayol23+')

# %% [markdown]
# This creates a NN emulator with few settings improved to those used in Cabayol23

# %%
fast_training=False
if fast_training:
    print('Using a sub-optimal training for the Nyx emulator, the results will not be as good as possible!')
    # the emulator performance will not be super good here
    nn_emu_nyx = NNEmulator(archive=nyx_arch)
else:
    # this might take a while to train...
    nn_emu_nyx = NNEmulator(archive=nyx_arch, emulator_label='Nyx_v1')

# %% [markdown]
# ### C. POINTING ALSO TO A PRE-DEFINED TRAINING SET

# %% [markdown]
# You can also specify the training set instead of passing an archive. If you want to test this, set load_training=True (it will take some time to run)

# %%
load_training=False

# %%
if load_training:
    gp_emu_P21 = GPEmulator(training_set='Pedersen21', emulator_label='Pedersen21')

# %%
if load_training:
    gp_emu_P23 = GPEmulator(training_set='Pedersen21', emulator_label='Pedersen23')

# %%
if load_training:
    nn_emu_C23 = NNEmulator(training_set='Cabayol23', emulator_label='Cabayol23')

# %%
if load_training:
    nn_emu_nyx = NNEmulator(training_set='Nyx23', emulator_label='Nyx_v0')

# %% [markdown]
# The latest Nyx emulator version is 'Nyx_v1', which produces much better results than the previous 'Nyx_v0'

# %%
if load_training:
    nn_emu_nyx = NNEmulator(training_set='Nyx23', emulator_label='Nyx_v1')

# %%

# %% [markdown]
# ### TESTING THE EMULATOR

# %% [markdown]
# One can also ask the archives for a testing set (a simulation not included in the emulator training)

# %%
mpg_test_P21 = mpg_arch_P21.get_testing_data(sim_label='mpg_seed')

# %%
mpg_test_C23 = mpg_arch_C23.get_testing_data(sim_label='mpg_seed')

# %%
nyx_test = nyx_arch.get_testing_data(sim_label='nyx_3')#, emu_params = nn_emu_nyx.emu_params)


# %%
def emulator_vs_true(emulator, test_data, iz=0, plot_ratio=True, smooth_test=True):
    
    # get true P1D in test data (over a certain k range)
    k_Mpc = test_data[iz]['k_Mpc']
    test_p1d = test_data[iz]['p1d_Mpc']
    
    # k range used in plot
    mask = (k_Mpc>0) & (k_Mpc<emulator.kmax_Mpc)
    k_Mpc = k_Mpc[mask]
    test_p1d = test_p1d[mask]

    # if smooth_test, use polynomial fit to the test data as truth (as assumed by emulator)
    fit_p1d = poly_p1d.PolyP1D(k_Mpc,test_p1d,deg=emulator.ndeg)
    test_p1d = fit_p1d.P_Mpc(k_Mpc)

    # get emulator parameter values of the test data
    model={}
    for param in emulator.emu_params:
        try:
            model[param]=test_data[iz][param]
        except:
            model[param]=test_data[iz]['cosmo_params'][param]
        print(param,model[param])
    
    # make emulator prediction
    emu_p1d = emulator.emulate_p1d_Mpc(model,k_Mpc)

    if plot_ratio:
        plt.plot(k_Mpc,emu_p1d/test_p1d-1.0)
        plt.plot(k_Mpc,k_Mpc*0.0,ls=':',color='gray')
        plt.ylabel(r'P1D residuals', fontsize = 14)
    else:
        plt.plot(k_Mpc,emu_p1d,label='emulated')
        plt.plot(k_Mpc,test_p1d,label='true')
        plt.legend(fontsize = 14)
        plt.ylabel(r'P1D [Mpc]', fontsize = 14)
    plt.xscale('log')
    plt.xlabel(r'$k$ [1/Mpc]', fontsize = 14)


# %%
emulator_vs_true(gp_emu_P23,mpg_test_P21,iz=5,plot_ratio=True)

# %%
emulator_vs_true(nn_emu_C23plus,mpg_test_C23,iz=5,plot_ratio=True)

# %%
emulator_vs_true(nn_emu_nyx,nyx_test,iz=3,plot_ratio=True)

# %% [markdown]
# ## D. EXTENDED EMULATOR

# %%
nn_emu_C23 = NNEmulator(archive=mpg_arch_C23, kmax_Mpc=8, ndeg=7)

# %%
nn_emu_C23 = NNEmulator(archive=mpg_arch_C23, emulator_label='Cabayol23_extended')

# %%
emulator_vs_true(nn_emu_C23,mpg_test_C23,iz=5,plot_ratio=True)

# %%
nn_emu_C23.ndeg

# %%
