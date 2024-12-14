# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: lace
#     language: python
#     name: python3
# ---

# %% [markdown]
# # LaCE EMULATOR TUTORIAL

# %% [markdown]
# #### IMPORTS

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import numpy as np
from matplotlib import pyplot as plt
from lace.emulator.emulator_manager import set_emulator    


# %%
import os, sys
import numpy as np
from matplotlib import pyplot as plt
# our modules
from lace.archive import gadget_archive
from lace.archive import nyx_archive
from lace.emulator.nn_emulator import NNEmulator
from lace.emulator.gp_emulator import GPEmulator
from lace.utils import poly_p1d
from lace.emulator.emulator_manager import set_emulator, emulators_supported

# %% [markdown]
# ## LOAD ARCHIVE

# %%
# you could specify here the path to the Nyx files, or set a NYX_PATH variable
archive_nyx = nyx_archive.NyxArchive(verbose=True, 
                                 nyx_version="Jul2024")

# %%
archive_mpg = gadget_archive.GadgetArchive(postproc="Cabayol23")

# %%
testing_central_nyx = archive_nyx.get_testing_data('nyx_central')
testing_central_mpg = archive_mpg.get_testing_data('mpg_central')


# %% [markdown]
# ## LOAD EXISTING EMULATORS

# %%
emulators_supported()

# %% [markdown]
# We have developed emulators using different architectures and training sets. The preferred emulators are:
# - Cabayol23+ for mpg simulators (Cabayol-Garcia+23, https://ui.adsabs.harvard.edu/abs/2023MNRAS.525.3499C/abstract)
# - Nyx_alphap for Nyx simulations
#
# Loading the emulators is fairly simple:

# %% [markdown]
# We can load any of the supported emulators providning an archive. The emulator and the archive must be consistent. The mapping between archives and emulators can be found in the [documentation](https://igmhub.github.io/LaCE/)

# %%
emulator_nyx = set_emulator(
        emulator_label="Nyx_alphap",
        archive=archive_nyx,
    )

# %%
emulator_gadget = set_emulator(
        emulator_label="Cabayol23+",
        archive=archive_mpg,
    )

# %% [markdown]
# The set_emulator function returns an emulator object loading a pre-defined model that we point to. If you want to load a different model, you can do so by providing the path to the model.

# %%
# %%time
emulator = NNEmulator(
    training_set='Cabayol23',
    emulator_label='Cabayol23+',
    model_path='NNmodels/Cabayol23+/Cabayol23+.pt', 
    train=False
)


# %% [markdown]
# ## EVALUATING THE EMULATOR

# %%
k_Mpc = np.geomspace(0.1, 3, 100)

# %% [markdown]
# ### FOR A PROVIDED SET OF PARAMETERS

# %%

input_params = {
    'Delta2_p': [0.35, 0.4],
    'n_p': [-2.3, -2.3],
    'mF': [0.66, 0.66],
    'gamma': [1.5, 1.5],
    'sigT_Mpc': [0.128, 0.128],
    'kF_Mpc': [10.5, 10.5]
}
p1d = emulator_gadget.emulate_p1d_Mpc(input_params, k_Mpc)

for ii in range(p1d.shape[0]):
    plt.plot(k_Mpc, k_Mpc * p1d[ii]/np.pi)
plt.xlabel(r'$k_\parallel$ [1/Mpc]')
plt.ylabel(r'$\pi^{-1} \, k_\parallel \, P_\mathrm{1D}$')
plt.xscale('log')

# %% [markdown]
# ### MPG SIMULATIONS

# %%
p1d = emulator_gadget.emulate_p1d_Mpc(testing_central_mpg[0], k_Mpc)
plt.loglog(k_Mpc, k_Mpc * p1d/np.pi)
plt.xlabel(r'$k_\parallel$ [1/Mpc]')
plt.ylabel(r'$\pi^{-1} \, k_\parallel \, P_\mathrm{1D}$')
plt.xscale('log')

# %% [markdown]
# ### NYX SIMULATIONS

# %%
p1d = emulator_gadget.emulate_p1d_Mpc(testing_central_nyx[0], k_Mpc)
plt.loglog(k_Mpc, k_Mpc * p1d/np.pi)
plt.xlabel(r'$k_\parallel$ [1/Mpc]')
plt.ylabel(r'$\pi^{-1} \, k_\parallel \, P_\mathrm{1D}$')
plt.xscale('log')

# %% [markdown]
# ##### It's also possible to provide a different  k_Mpc for each call

# %%
k_Mpc = np.array([k_Mpc, k_Mpc])
p1d = emulator_gadget.emulate_p1d_Mpc(input_params, k_Mpc)

for ii in range(p1d.shape[0]):
    plt.plot(k_Mpc[ii], k_Mpc[ii] * p1d[ii]/np.pi)
plt.xlabel(r'$k_\parallel$ [1/Mpc]')
plt.ylabel(r'$\pi^{-1} \, k_\parallel \, P_\mathrm{1D}$')
plt.xscale('log')

# %% [markdown]
# ##  FOR DEVELOPERS: TRAINING NEW EMULATORS 

# %% [markdown]
# ### EMULATOR LABELS: When training a new emulator, we can provide a set of hyperparameters or an emulator_label. Emulator labels identify the hyperparameters to be used. One can find the hyperparameters and the possible emulator labels in the [documentation](https://igmhub.github.io/LaCE/)

# %%
# This will create a new emulator with the hyperparameters defined in the emulator_label
emulator_C23 = NNEmulator(archive=archive_mpg, 
                      emulator_label='Cabayol23+')

# %%
# This will create a new emulator with the hyperparameters defined in the emulator_label and save it in the path provided
emulator_C23 = NNEmulator(archive=archive_mpg, 
                      emulator_label='Cabayol23+',
                      model_path='NNmodels/Cabayol23+/tests/Cabayol23+.pt')

# %% [markdown]
# ### TRAINING SETS: When training a new emulator, we can provide a training_set label or an archive. Training sets are pre-defined sets of simulations that have been used to train the emulators. One can find the training sets and the possible training_set labels in the [documentation](https://igmhub.github.io/LaCE/)

# %%
# This will create a new emulator with the hyperparameters defined in the emulator_label and save it in the path provided
emulator_C23 = NNEmulator(training_set='Cabayol23', 
                      emulator_label='Cabayol23+',
                      model_path='NNmodels/Cabayol23+/tests/Cabayol23+.pt')

# %% [markdown]
# ### TRAINING NOT PRE-DEFINED EMULATORS

# %%
emulator_custom = NNEmulator(training_set='Cabayol23', 
                             ndeg=4,
                             kmax_Mpc=4,
                             emu_params=['Delta2_p', 'n_p','mF', 'sigT_Mpc', 'gamma', 'kF_Mpc'],
                             nepochs=1,
                             lr0=1e-3,
                             batch_size=100,
                             )

# %% [markdown]
# ##  GAUSSIAN PROCESS EMULATOR

# %% [markdown]
# #### The Gaussian process emulator uses the following default parameters:

# %% [markdown]
# - paramList=['Delta2_p', 'n_p','mF', 'sigT_Mpc', 'gamma', 'kF_Mpc']
# - kmax_Mpc=10
# - ndeg=4

# %% [markdown]
# ## TRAINING GAUSSIAN PROCESS EMULATORS

# %%
# specifying the training_set
emulator = GPEmulator(training_set='Pedersen21')

# %%
# specifying the archive
archive_p12 = gadget_archive.GadgetArchive(postproc="Pedersen21")
emulator = GPEmulator(archive=archive_p12)

# %% [markdown]
# ## LEAVE ONE OUT TESTS

# %% [markdown]
# #### To perform a leave one out test, we can provide the drop_sim parameter. This parameter is the name of the simulation to be left out.

# %%
# training a new emulator
emulator = NNEmulator(
    training_set='Cabayol23',
    emulator_label='Cabayol23+',
    drop_sim='nyx_0',
)

# %%
# loading a pre-trained emulator
emulator = set_emulator(emulator_label='Cabayol23+', 
                        archive=archive_mpg, 
                        drop_sim='mpg_0')
