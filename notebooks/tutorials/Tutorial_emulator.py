# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Emulator tutorial
#
# This notebook explain how to:
# - Load an emulator
# - Use the emulator
# - Train a new emulator

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

# %%
# No need to load archive, just training set, to speed out process

# %%

from lace.archive import gadget_archive
from lace.archive import nyx_archive
from lace.emulator.nn_emulator import NNEmulator
from lace.emulator.gp_emulator import GPEmulator
from cup1d.likelihood.pipeline import set_archive

# %%

training_set = "Nyx23_Jul2024"
archive_nyx = set_archive(training_set)

# %%

emulator = GPEmulator(archive=archive_nyx, emulator_label="CH24_nyx_gp")


# %% [markdown]
# ## 2 min, need to save

# %%
archive = gadget_archive.GadgetArchive(postproc="Cabayol23")

# %%
# emulator = set_emulator(emulator_label="CH24")
emulator = GPEmulator(archive=archive, emulator_label="CH24_mpg_gp")

# %%
archive = gadget_archive.GadgetArchive(postproc="Cabayol23")
# emulator = GPEmulator(archive=archive, emulator_label="Pedersen23_ext")

# %% [markdown]
# SAVE_EMU!!! With everything, no need to read data

# %%
k_Mpc = np.geomspace(0.1, 3, 100)
input_params = {
    'Delta2_p': [0.30, 0.30],
    'n_p': [-2.301, -2.3],
    'alpha_p': [-0.215, -0.215],
    'mF': [0.66, 0.66],
    'gamma': [1.5, 1.5],
    'sigT_Mpc': [0.128, 0.128],
    'kF_Mpc': [10.5, 10.5]
}
p1d = emulator.emulate_p1d_Mpc(input_params, k_Mpc)
# p1d = emulator.emulate_p1d_Mpc(testing_data[0],kMpc)

for ii in range(p1d.shape[0]):
    plt.plot(k_Mpc, k_Mpc * p1d[ii]/np.pi)
    
plt.xlabel(r'$k_\parallel$ [1/Mpc]')
plt.ylabel(r'$\pi^{-1} \, k_\parallel \, P_\mathrm{1D}$')
plt.xscale('log')

# %%
plt.plot(k_Mpc, p1d[1]/p1d[0]-1)

# %%
archive = archive_nyx

# %%
emu_params = ['Delta2_p', 'n_p', 'mF', 'sigT_Mpc', 'gamma', 'kF_Mpc']
training_data = archive.get_training_data(emu_params, average="both")

# %%
training_data = archive.get_testing_data("nyx_central")

# %%
from lace.emulator.gp_emulator import func_poly
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from lace.utils import poly_p1d

# %%

# repo = os.path.dirname(lace.__path__[0])
# fname = os.path.join(repo, "data", "ff_mpgcen.npy")
# input_norm = np.load(fname, allow_pickle=True).item()
# norm_imF = interp1d(
#     input_norm["mF"], input_norm["p1d_Mpc_mF"], axis=0
# )

# %%
_k_Mpc = training_data[0]['k_Mpc']
ind = (_k_Mpc < 5) & (_k_Mpc > 0)
k_Mpc = _k_Mpc[ind]
x = k_Mpc / emulator.kmax_Mpc
nsam = len(training_data)
p1d_Mpc_sim = np.zeros((nsam, k_Mpc.shape[0]))
p1d_Mpc_sm = np.zeros((nsam, k_Mpc.shape[0]))
p1d_Mpc_emu = np.zeros((nsam, k_Mpc.shape[0]))

for ii in range(nsam):
# for ii in range(10):
    if "kF_Mpc" not in  training_data[ii]:
        continue
    p1d_Mpc_sim[ii] = training_data[ii]['p1d_Mpc'][ind]
    p1d_Mpc_emu[ii] = emulator.emulate_p1d_Mpc(
        training_data[ii], 
        k_Mpc
    )

    if emulator.emu_type == "gpolyfit":
        norm = np.interp(
            k_Mpc,
            emulator.input_norm["k_Mpc"],
            emulator.norm_imF(training_data[ii]["mF"]),
        )
        y2fit = np.log(training_data[ii]["p1d_Mpc"][ind] / norm)
        store_fit, _ = curve_fit(emulator.func_poly, x, y2fit)
        # print("a", store_fit)
    
        p1d_Mpc_sm[ii] = np.exp(emulator.func_poly(x, *store_fit)) * norm
    else:
        fit_p1d = poly_p1d.PolyP1D(
            training_data[ii]["k_Mpc"], 
            training_data[ii]["p1d_Mpc"], 
            kmin_Mpc=1.0e-3, 
            kmax_Mpc=emulator.kmax_Mpc, 
            deg=emulator.ndeg
        )

        p1d_Mpc_sm[ii] = np.exp(fit_p1d.lnP(np.log(k_Mpc)))

# %%

# emulator = GPEmulator(archive=archive, emulator_label="CH24_mpg_gp")

# %% [markdown]
# #### L1O test for emulator

# %%
np.isfinite(rat[:, 0])

# %%
# ii = 0
# plt.plot(k_Mpc, p1d_Mpc_sim[ii]/p1d_Mpc_sm[ii])
# plt.plot(k_Mpc, p1d_Mpc_emu[ii]/p1d_Mpc_sm[ii])
# plt.xlabel(r'$k_\parallel$ [1/Mpc]')
# plt.ylabel(r'$\pi^{-1} \, k_\parallel \, P_\mathrm{1D}$')
# plt.xscale('log')

# %%
rat = p1d_Mpc_emu/p1d_Mpc_sim-1
ind = np.isfinite(rat[:, 0])
# rat = p1d_Mpc_emu/p1d_Mpc_sm-1
# plt.plot(k_Mpc, np.percentile(rat, 5, axis=0), label="5")
plt.plot(k_Mpc, np.percentile(rat[ind], 16, axis=0), "C0", label="16")
plt.plot(k_Mpc, np.percentile(rat[ind], 50, axis=0), "C1", label="50")
plt.plot(k_Mpc, np.percentile(rat[ind], 84, axis=0), "C2", label="84")
# plt.plot(k_Mpc, np.percentile(rat, 95, axis=0), label="95")

ratsm = p1d_Mpc_emu/p1d_Mpc_sm-1
plt.plot(k_Mpc, np.percentile(ratsm[ind], 16, axis=0), "C0--", label="16")
plt.plot(k_Mpc, np.percentile(ratsm[ind], 50, axis=0), "C1--", label="50")
plt.plot(k_Mpc, np.percentile(ratsm[ind], 84, axis=0), "C2--", label="84")


plt.plot(k_Mpc, k_Mpc[:]*0, "k:")
plt.legend()
plt.xscale("log")
plt.ylim(-0.025, 0.025)
plt.xlabel("k [Mpc]")
plt.ylabel("emu/data-1")
# plt.savefig("precision_PE23_noscalig.png")

# %% [markdown]
# compare with smooth!!

# %%
# # %%time
# emulator_P23 = set_emulator(emulator_label="CH24")

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

# %%
# archive = gadget_archive.GadgetArchive(postproc="Pedersen21")
archive = gadget_archive.GadgetArchive(postproc="Cabayol23")

# %%
training_data = archive.get_training_data(emu_params=emu_params, average="both")

# %%
len(training_data)

# %%
# for ii in range(len(training_data)):
#     ind = (training_data[ii]["k_Mpc"] > 0) & (training_data[ii]["k_Mpc"] < 4)
#     print(ii, training_data[ii]["k_Mpc"][ind].shape[0])
#     print(training_data[ii]["k_Mpc"][ind][0], training_data[ii]["k_Mpc"][ind][-1])

# %%
emulator = NNEmulator(
    archive=archive, 
    emulator_label='CH24', 
    nepochs=500,
    gamma_optimizer=0.75
)

# %% [markdown]
# keeps getting better down to 500

# %%
plt.plot(emulator.loss_arr)
plt.yscale("log")

# %%

from scipy.optimize import curve_fit
def func_poly(x, a, b, c, d, e, f, g):
    return (
        a * x**0.5
        + b * x**0.75
        + c * x
        + d * x**2
        + e * x**3
        + f * x**4
        + g * x**5
    )


# %%
get_smooth = True

_k_Mpc = training_data[0]['k_Mpc']
ind = (_k_Mpc < 4) & (_k_Mpc > 0)
k_Mpc = _k_Mpc[ind]
k_fit = k_Mpc/emulator.kmax_Mpc
nsam = len(training_data)
p1d_Mpc_sim = np.zeros((nsam, k_Mpc.shape[0]))
p1d_Mpc_emu = np.zeros((nsam, k_Mpc.shape[0]))
if get_smooth:
    p1d_Mpc_sm = np.zeros((nsam, k_Mpc.shape[0]))
    norm_use = np.interp(
        k_Mpc, emulator.input_norm["kpar"], emulator.input_norm["p1d"]
    )

for ii in range(nsam):
    p1d_Mpc_sim[ii] = training_data[ii]['p1d_Mpc'][ind]
    p1d_Mpc_emu[ii] = emulator.emulate_p1d_Mpc(
        training_data[ii], 
        k_Mpc
    )
    if get_smooth:
        mF = training_data[ii]["mF"]
        norm = emulator.func_norm(np.log(mF)) * norm_use
        yfit = np.log(training_data[ii]["p1d_Mpc"][ind]/norm)
        popt, _ = curve_fit(func_poly, k_fit, yfit)
        p1d_Mpc_sm[ii] = norm * np.exp(func_poly(k_fit, *popt))

# %%
# rat = p1d_Mpc_emu/p1d_Mpc_sim-1
rat = p1d_Mpc_emu/p1d_Mpc_sm-1
plt.plot(k_Mpc, np.percentile(rat, 5, axis=0), label="5")
plt.plot(k_Mpc, np.percentile(rat, 16, axis=0), label="16")
plt.plot(k_Mpc, np.percentile(rat, 50, axis=0), label="50")
plt.plot(k_Mpc, np.percentile(rat, 84, axis=0), label="84")
plt.plot(k_Mpc, np.percentile(rat, 95, axis=0), label="95")
plt.plot(k_Mpc, k_Mpc[:]*0, "k:")
plt.plot(k_Mpc, k_Mpc[:]*0+0.01, "k:")
plt.plot(k_Mpc, k_Mpc[:]*0-0.01, "k:")
plt.legend()
plt.xscale("log")
plt.xlabel("k [Mpc]")
plt.ylabel("emu/data-1")

# %% [markdown]
# #### testing

# %%
testing_data = archive.get_testing_data("mpg_seed")
len(testing_data)

# %%
get_smooth = True

_k_Mpc = testing_data[0]['k_Mpc']
ind = (_k_Mpc < 4) & (_k_Mpc > 0)
k_Mpc = _k_Mpc[ind]
k_fit = k_Mpc/emulator.kmax_Mpc
nsam = len(testing_data)
p1d_Mpc_sim = np.zeros((nsam, k_Mpc.shape[0]))
p1d_Mpc_emu = np.zeros((nsam, k_Mpc.shape[0]))
if get_smooth:
    p1d_Mpc_sm = np.zeros((nsam, k_Mpc.shape[0]))
    norm_use = np.interp(
        k_Mpc, emulator.input_norm["kpar"], emulator.input_norm["p1d"]
    )

for ii in range(nsam):
    p1d_Mpc_sim[ii] = testing_data[ii]['p1d_Mpc'][ind]
    p1d_Mpc_emu[ii] = emulator.emulate_p1d_Mpc(
        testing_data[ii], 
        k_Mpc
    )
    if get_smooth:
        mF = testing_data[ii]["mF"]
        norm = emulator.func_norm(np.log(mF)) * norm_use
        yfit = np.log(testing_data[ii]["p1d_Mpc"][ind]/norm)
        popt, _ = curve_fit(func_poly, k_fit, yfit)
        p1d_Mpc_sm[ii] = norm * np.exp(func_poly(k_fit, *popt))

# %%

cmap=plt.get_cmap('tab20')

# %%
for ii in range(nsam):
    plt.plot(k_Mpc, p1d_Mpc_emu[ii]/p1d_Mpc_sm[ii]-1, label=str(testing_data[ii]["z"]), color=cmap(ii))
    # plt.plot(k_Mpc, p1d_Mpc_sim[ii]/p1d_Mpc_sm[ii]-1, color=cmap(ii))
plt.plot(k_Mpc, k_Mpc[:] * 0, "k:")
plt.plot(k_Mpc, k_Mpc[:] * 0 + 0.01, "k:")
plt.plot(k_Mpc, k_Mpc[:] * 0 - 0.01, "k:")
plt.legend()
plt.xscale("log")

# %%
for ii in range(nsam):
    plt.plot(k_Mpc, p1d_Mpc_emu[ii]/p1d_Mpc_sm[ii]-1, label=str(testing_data[ii]["z"]), color=cmap(ii))
    # plt.plot(k_Mpc, p1d_Mpc_sim[ii]/p1d_Mpc_sm[ii]-1, color=cmap(ii))
plt.plot(k_Mpc, k_Mpc[:] * 0, "k:")
plt.plot(k_Mpc, k_Mpc[:] * 0 + 0.01, "k:")
plt.plot(k_Mpc, k_Mpc[:] * 0 - 0.01, "k:")
plt.legend()
plt.xscale("log")

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
