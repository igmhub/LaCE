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
# # Tutorial explaining how to use emulators

# %% [markdown]
# ## For normal users

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import numpy as np
from matplotlib import pyplot as plt
from lace.emulator.emulator_manager import set_emulator

# %% [markdown]
# We have developed emulators using different architectures and training sets. The preferred emulators are:
# - Cabayol23+ for mpg simulators (Cabayol-Garcia+23, https://ui.adsabs.harvard.edu/abs/2023MNRAS.525.3499C/abstract)
# - Nyx_alphap for Nyx simulations
#
# Loading the emulators is fairly simple:

# %% [markdown]
# #### mpg simulations

# %%
# %%time
emulator_C23 = set_emulator(emulator_label="Cabayol23+")

# %% [markdown]
# To evaluate it, provide input cosmological and IGM parameters and a series of kpar values

# %%
k_Mpc = np.geomspace(0.1, 3, 100)
input_params = {
    'Delta2_p': [0.35, 0.4],
    'n_p': [-2.3, -2.3],
    'mF': [0.66, 0.66],
    'gamma': [1.5, 1.5],
    'sigT_Mpc': [0.128, 0.128],
    'kF_Mpc': [10.5, 10.5]
}
p1d = emulator_C23.emulate_p1d_Mpc(input_params, k_Mpc)

for ii in range(p1d.shape[0]):
    plt.plot(k_Mpc, k_Mpc * p1d[ii]/np.pi)
plt.xlabel(r'$k_\parallel$ [1/Mpc]')
plt.ylabel(r'$\pi^{-1} \, k_\parallel \, P_\mathrm{1D}$')
plt.xscale('log')

# %% [markdown]
# for a different k_Mpc for each call

# %%
k_Mpc = np.array([k_Mpc, k_Mpc])
p1d = emulator_C23.emulate_p1d_Mpc(input_params, k_Mpc)

for ii in range(p1d.shape[0]):
    plt.plot(k_Mpc[ii], k_Mpc[ii] * p1d[ii]/np.pi)
plt.xlabel(r'$k_\parallel$ [1/Mpc]')
plt.ylabel(r'$\pi^{-1} \, k_\parallel \, P_\mathrm{1D}$')
plt.xscale('log')

# %% [markdown]
# To get the amplitude and slope of the linear power spectrum from a target cosmology (input of emulator)

# %%
from lace.cosmo.camb_cosmo import get_cosmology
from lace.cosmo.fit_linP import get_linP_Mpc_zs

# %%
h = 0.6778216034931903
H0 = 100 * h
omch2 = 0.11762521740991255
ombh2 = 0.022239961079792658
ns = 0.9628958142411611
As = np.exp(3.0151299559643365)*1e-10

cosmo = get_cosmology(H0=H0, omch2=omch2, ombh2=ombh2, ns=ns, As=As)

zs = [3]
kp_Mpc = 0.7
get_linP_Mpc_zs(cosmo, zs, kp_Mpc)

# %% [markdown]
# To get the amplitude and slope of the linear power spectrum on kms units at z=3 for a target cosmology (output of cup1d)

# %%
from lace.cosmo.fit_linP import parameterize_cosmology_kms

# %%
h = 0.6778216034931903
H0 = 100 * h
omch2 = 0.11762521740991255
ombh2 = 0.022239961079792658
ns = 0.9628958142411611
As = np.exp(3.0151299559643365)*1e-10

cosmo = get_cosmology(H0=H0, omch2=omch2, ombh2=ombh2, ns=ns, As=As)

zstar = 3
kp_kms = 0.009
camb_results = None
parameterize_cosmology_kms(cosmo, camb_results, zstar, kp_kms)


# %% [markdown]
# #### nyx simulations

# %%
# %%time
# You need to specify the path to the Nyx files by setting a NYX_PATH variable
emulator_Nyx = set_emulator(emulator_label="Nyx_alphap")

# %%
k_Mpc = np.geomspace(0.1, 3, 100)
input_params = {
    'Delta2_p': 0.35,
    'n_p': -2.3,
    "alpha_p":-0.22,
    'mF': 0.66,
    'gamma': 1.5,
    'sigT_Mpc': 0.128,
    'kF_Mpc': 10.5
}
p1d = emulator_Nyx.emulate_p1d_Mpc(input_params, k_Mpc)

# %%
plt.plot(k_Mpc, k_Mpc * p1d/ np.pi)
plt.xlabel(r'$k_\parallel$ [1/Mpc]')
plt.ylabel(r'$\pi^{-1} \, k_\parallel \, P_\mathrm{1D}$')
plt.xscale('log')

# %% [markdown]
# ## A general user should stop here. For developers below

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
# ### List of supported emulators
#
# Total list of emulators:

# %%
emulators_supported()

# %% [markdown]
# ## CREATE TRAINING AND TESTING ARCHIVE (Gadget)

# %%
# list of emulator parameters used with Gadget sims
emu_params=['Delta2_p', 'n_p','mF', 'sigT_Mpc', 'gamma', 'kF_Mpc']

# %%
archive = gadget_archive.GadgetArchive(postproc="Cabayol23")
training_data=archive.get_training_data(emu_params=emu_params)
len(training_data)

# %%
testing_data = archive.get_testing_data(sim_label='mpg_central')
len(testing_data)

# %%
kMpc = testing_data[0]['k_Mpc']
kMpc = kMpc[(kMpc>0) & (kMpc<4)]

# %% [markdown]
# ## NEURAL NETWORK EMULATOR  

# %% [markdown]
# Some of the cells in this notebooks can be quite slow, so we only run them if thorough==True

# %%
thorough=False

# %% [markdown]
# ### Example 1: We can train a custom emulator... 

# %% [markdown]
# #### A. passing a custom archive:

# %%
emulator = NNEmulator(archive=archive, nepochs=1)

# %% [markdown]
# ### or a training_set label

# %%
emulator = NNEmulator(training_set='Cabayol23',nepochs=1)

# %% [markdown]
# #### If none or both are provided, the emulator fails. 

# %%
emulator = NNEmulator(nepochs=1)

# %% [markdown]
# ### Example 2: We can train a pre defined emulator... 

# %% [markdown]
# #### A. with a training_set label

# %%
emulator = NNEmulator(training_set='Cabayol23', emulator_label='Cabayol23+')

# %% [markdown]
# #### B. with an archive

# %%
emulator = NNEmulator(archive=archive, emulator_label='Cabayol23+', nepochs=1)

# %% [markdown]
# #### If none are provided, the training fails

# %%
emulator = NNEmulator(emulator_label='Cabayol23+', nepochs=1)

# %% [markdown]
# ### Example 3: Load a pre-trained emulator, providing the path of the saved network parameters

# %%
# %%time
emulator = NNEmulator(
    training_set='Cabayol23',
    emulator_label='Cabayol23+',
    model_path='NNmodels/Cabayol23+/Cabayol23+.pt', 
    train=False
)
# test emulator by making simple plot
p1d = emulator.emulate_p1d_Mpc(testing_data[0], kMpc)
plt.loglog(kMpc,p1d)

# %% [markdown]
# ##  GAUSSIAN PROCESS EMULATOR

# %% [markdown]
# #### The Gaussian process emulator uses the following default parameters:

# %% [markdown]
# - paramList=['Delta2_p', 'n_p','mF', 'sigT_Mpc', 'gamma', 'kF_Mpc']
# - kmax_Mpc=10
# - ndeg=4

# %% [markdown]
# ### Example 1: Train custom emulator 

# %% [markdown]
# #### with a defined training_set

# %%
emulator = GPEmulator(training_set='Pedersen21')

# %% [markdown]
# #### with a custom archive

# %%
emulator = GPEmulator(archive=archive)

# %%
# test emulator by making simple plot
p1d = emulator.emulate_p1d_Mpc(testing_data[0],kMpc)
plt.plot(kMpc,p1d)

# %% [markdown]
# ### Example 2: Pre-defined GP emulators:

# %%
emulator = GPEmulator(training_set='Pedersen21', emulator_label='Pedersen21')

# %%
emulator = GPEmulator(training_set='Pedersen21', emulator_label='Pedersen23')

# %%
emulator = GPEmulator(training_set='Cabayol23', emulator_label='k_bin_sm')

# %% [markdown]
# ## LaCE-Nyx emulator (NN)

# %%
# emulators will use different emulator parameters depending on the archive
nyx_emu_params=['Delta2_p','n_p','alpha_p','mF','sigT_Mpc','gamma','kF_Mpc']


# %%
# you could specify here the path to the Nyx files, or set a NYX_PATH variable
archive = nyx_archive.NyxArchive(verbose=True)

# %%
if thorough:
    emulator = NNEmulator(archive=archive, drop_sim = 'nyx_3', emulator_label='Nyx_alphap')

# %% [markdown]
# #### Pre-trained

# %% [markdown]
# #### L1O

# %%
emulator = NNEmulator(
    training_set='Nyx23_Oct2023',
    emulator_label='Nyx_alphap',
    emu_params=nyx_emu_params,
    model_path='NNmodels/Nyxap_Oct2023/Nyxap_drop_sim_nyx_0.pt', 
    drop_sim='nyx_0',
    train=False,
)

# %%
# test emulator by making simple plot
testing_data = archive.get_testing_data('nyx_0')

# %% [markdown]
# #### Full emu

# %%
# %%time
emulator = NNEmulator(
    training_set='Nyx23_Oct2023',
    emulator_label='Nyx_alphap',
    emu_params=nyx_emu_params,
    model_path='NNmodels/Nyxap_Oct2023/Nyx_alphap.pt',
    train=False,
)

# %%
# test emulator by making simple plot
testing_data = archive.get_testing_data('nyx_central')

p1ds_true = np.zeros(shape=(11,75))
p1ds = np.zeros(shape=(11,75))

for m in range(11):
    if('kF_Mpc' not in testing_data[m]):
        continue
    p1d_true = testing_data[m]['p1d_Mpc']
    kMpc = testing_data[m]['k_Mpc']
    kMpc_test = kMpc[(kMpc>0) & (kMpc<4)]
    p1d_true = p1d_true[(kMpc>0) & (kMpc<4)]
    
    fit_p1d = poly_p1d.PolyP1D(kMpc_test,p1d_true, kmin_Mpc=1e-3,kmax_Mpc=4,deg=5)
    p1d_true = fit_p1d.P_Mpc(kMpc_test)
    
    p1d = emulator.emulate_p1d_Mpc(testing_data[m], kMpc_test)
    
    
    p1ds_true[m] = p1d_true
    p1ds[m] = p1d
    
    plt.plot(kMpc_test,p1d, label = 'Emulated', color = 'navy')
    plt.plot(kMpc_test,p1d_true, label = 'True', color = 'crimson')

    plt.xlabel(r'$k$ [1/Mpc]')
    plt.ylabel(r'P1D')

plt.show()

# %%
p1ds_true = np.zeros(shape=(11,75))
p1ds = np.zeros(shape=(11,75))

for m in range(11):
    if('kF_Mpc' not in testing_data[m]):
        continue
    p1d_true = testing_data[m]['p1d_Mpc']
    kMpc = testing_data[m]['k_Mpc']
    kMpc_test = kMpc[(kMpc>0) & (kMpc<4)]
    p1d_true = p1d_true[(kMpc>0) & (kMpc<4)]
    
    fit_p1d = poly_p1d.PolyP1D(kMpc_test,p1d_true, kmin_Mpc=1e-3,kmax_Mpc=4,deg=5)
    p1d_true = fit_p1d.P_Mpc(kMpc_test)
    
    p1d = emulator.emulate_p1d_Mpc(testing_data[m],kMpc_test)
    
    
    p1ds_true[m] = p1d_true
    p1ds[m] = p1d
    
    plt.plot(kMpc_test, p1d/p1d_true, label = 'z=' + str(np.round(testing_data[m]["z"], 2)))

    
plt.plot(kMpc_test, np.ones_like(kMpc_test), 'k--')

percent_error = p1ds/p1ds_true
plt.errorbar(kMpc_test, 
             np.nanmean(percent_error, axis=0), 
             np.nanstd(percent_error, axis=0), 
             color='k', alpha=0.2)
plt.xlabel(r'$k$ [1/Mpc]')
plt.ylabel(r'P1D')
plt.xscale('log')

plt.legend()

# %%
