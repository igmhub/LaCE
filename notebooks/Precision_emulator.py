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
# # Precision of LaCE

# %% [markdown]
# This notebook aims to check:
#
# - The precision of LaCE recovering Delta2_p as a function of the position in the convex hull. It does so in a realistic way, weighting by the eBOSS covariance matrix and the derivative of P1D relative to Delta2_p
#
# - The precision of LaCE for each simulation w/ and w/o l1O

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
import lace

from cup1d.data.data_Chabanier2019 import P1D_Chabanier2019

# %%
path_fig = os.path.dirname(lace.__path__[0]) + '/figures/'

# %% [markdown]
# #### Read eBOSS covariance matrix

# %%
data_boss = P1D_Chabanier2019()

norm_cov = np.zeros((len(data_boss.cov_Pk_kms), data_boss.cov_Pk_kms[0].shape[0]))
for ii in range(len(data_boss.cov_Pk_kms)):
    norm_cov[ii] = np.sqrt(np.diagonal(data_boss.cov_Pk_kms[ii])) * data_boss.k_kms / np.pi

# %% [markdown]
# #### Read sim data

# %%
# list of emulator parameters used with Gadget sims
emu_params=['Delta2_p', 'n_p','mF', 'sigT_Mpc', 'gamma', 'kF_Mpc']

file = '/home/jchaves/Proyectos/projects/lya/ForestFlow/data/std_pnd_mpg.npz'
err_pnd = np.load(file)
rel_err_p1d = err_pnd["std_p1d"]

archive = gadget_archive.GadgetArchive(postproc="Cabayol23")
training_data = archive.get_training_data(emu_params=emu_params)
mask = (training_data[0]['k_Mpc'] > 0) & (training_data[0]['k_Mpc'] < 4)
kMpc = training_data[0]['k_Mpc'][mask]
errp1d = err_pnd["std_p1d"][mask] * kMpc / np.pi

# %% [markdown]
# #### Compute Dp derivative

# %% [markdown]
# Get cosmo params from mpg_central at z=3

# %%
input = {
    'Delta2_p': 0.6179245757601503,
    'n_p': -2.348387407902965,
    'mF': 0.8711781695077168,
    'gamma': 1.7573934933568836,
    'sigT_Mpc': 0.1548976469710291,
    'kF_Mpc': 7.923157506298608,
}

testing_data = archive.get_testing_data(sim_label="mpg_central")
for ind_book in range(len(testing_data)):
    if(
        (testing_data[ind_book]['z'] == 3)
    ):
        _id = ind_book
Ap_cen = testing_data[_id][emu_params[0]]
np_cen = testing_data[_id][emu_params[1]]
for par in input:
    input[par] = testing_data[_id][par]
input

# %% [markdown]
# Range of Dp values in the sim

# %%
Dp_range = np.array([10., -10.])
for ii in range(len(archive.data)):
    dp = archive.data[ii]['Delta2_p']
    if(dp > Dp_range[1]):
        Dp_range[1] = dp
    if(dp < Dp_range[0]):
        Dp_range[0] = dp
Dp_range

# %%
nn = 100
dp_vals = np.linspace(Dp_range[0], Dp_range[1], nn)
orig_dp = emulator.emulate_p1d_Mpc(input, kMpc)
var_dp = np.zeros((nn, kMpc.shape[0]))
for ii in range(nn):
    input2 = input.copy()
    input2['Delta2_p'] = dp_vals[ii]
    var_dp[ii] = emulator.emulate_p1d_Mpc(input2, kMpc)

# %%
for ii in range(nn):
    plt.plot(kMpc, var_dp[ii]/orig_dp)

# %%
dp_range = np.max(var_dp/orig_dp, axis=0) - np.min(var_dp/orig_dp, axis=0)

# %%
dp_range

# %% [markdown]
# # L1O
#
# We compute chi2 and bias.

# %%
val_scaling = 1

full_emulator = NNEmulator(
    archive=archive,
    training_set='Cabayol23',
    emulator_label='Cabayol23',
    model_path='NNmodels/Cabayol23/Cabayol23.pt', 
    train=False
)
mask2 = kMpc < 10

# l10 (n, y), smooth (n, y), etc
reldiff_P1D = np.zeros((2, 2, len(archive.list_sim_cube), 11, kMpc.shape[0]))
wdiff_P1D = np.zeros((2, 2, len(archive.list_sim_cube), 11, kMpc.shape[0]))
chi2 = np.zeros((2, 2, len(archive.list_sim_cube)))
fob = np.zeros((2, 2, len(archive.list_sim_cube)))

arr_Ap = np.zeros(len(archive.list_sim_cube))
arr_np = np.zeros(len(archive.list_sim_cube))

ids_all = []

for isim, sim_label in enumerate(archive.list_sim_cube):

    ids = []
    zz = []
    for ind_book in range(len(training_data)):
        if(
            (training_data[ind_book]['sim_label'] == sim_label) & 
            (training_data[ind_book]['ind_phase'] == 'average') & 
            (training_data[ind_book]['ind_axis'] == 'average') &  
            (training_data[ind_book]['val_scaling'] == val_scaling)
        ):
            zz.append(training_data[ind_book]['z'])
            ids.append(ind_book)
            if(
                (training_data[ind_book]['z'] == 3)
            ):
                _id = ind_book
    nz = len(ids)

    ids_all.append(ids)

    emulator = NNEmulator(
        archive=archive,
        training_set='Cabayol23',
        emulator_label='Cabayol23',
        # model_path='NNmodels/Cabayol23/Cabayol23.pt', 
        model_path='NNmodels/Cabayol23/Cabayol23_drop_sim_'+sim_label+'.pt', 
        drop_sim=sim_label,
        train=False
    )

    p1d_emu = np.zeros((2, nz, kMpc[mask2].shape[0]))
    p1d_sim = np.zeros((nz, kMpc[mask2].shape[0]))
    p1d_sm = np.zeros((nz, kMpc[mask2].shape[0]))
    err_p1d = np.zeros((nz, kMpc[mask2].shape[0]))
    ndeg = p1d_emu.shape[1] + p1d_emu.shape[2] - len(emu_params)
    
    for ii, id in enumerate(ids):
        p1d_emu[0, ii, :] = full_emulator.emulate_p1d_Mpc(training_data[id], kMpc[mask2]) * kMpc[mask2] / np.pi
        p1d_emu[1, ii, :] = emulator.emulate_p1d_Mpc(training_data[id], kMpc[mask2]) * kMpc[mask2] / np.pi
        
        fit_p1d = poly_p1d.PolyP1D(
            kMpc, 
            training_data[id]['p1d_Mpc'][mask], 
            kmin_Mpc=1.e-5,
            # kmin_Mpc=kMpc.min(), 
            kmax_Mpc=emulator.kmax_Mpc, 
            deg=emulator.ndeg
        )
        p1d_sim[ii, :] = training_data[id]['p1d_Mpc'][mask][mask2] * kMpc[mask2] / np.pi
        p1d_sm[ii, :] = fit_p1d.P_Mpc(kMpc[mask2]) * kMpc[mask2] / np.pi

        k_kms = kMpc[mask2]/training_data[id]['dkms_dMpc']

        indz_cov = np.argmin(np.abs(data_boss.z - training_data[id]['z']))
        err_p1d[ii, :] = np.interp(k_kms, data_boss.k_kms, norm_cov[indz_cov], right=1000)/dp_range[mask2]

        for i0 in range(2):
            reldiff_P1D[i0, 0, isim, ii] = p1d_emu[i0, ii, :]/p1d_sim[ii, :] - 1
            reldiff_P1D[i0, 1, isim, ii] = p1d_emu[i0, ii, :]/p1d_sm[ii, :] - 1
            
            wdiff_P1D[i0, 0, isim, ii] = (p1d_emu[i0, ii, :]-p1d_sim[ii, :])/err_p1d[ii, :]*dp_range[mask2]
            wdiff_P1D[i0, 1, isim, ii] = (p1d_emu[i0, ii, :]-p1d_sm[ii, :])/err_p1d[ii, :]*dp_range[mask2]

    for i0 in range(2):
        chi2[i0, 0, isim] = np.sum((p1d_emu[i0]-p1d_sim)**2/err_p1d**2)/ndeg
        chi2[i0, 1, isim] = np.sum((p1d_emu[i0]-p1d_sm)**2/err_p1d**2)/ndeg
        fob[i0, 0, isim]  = np.sum((p1d_emu[i0]-p1d_sim)/err_p1d)/ndeg
        fob[i0, 1, isim]  = np.sum((p1d_emu[i0]-p1d_sm)/err_p1d)/ndeg
    
    arr_Ap[isim] = training_data[_id][emu_params[0]]
    arr_np[isim] = training_data[_id][emu_params[1]]

# %% [markdown]
# #### Convex hull

# %%
flags = ["$\\Delta^2_\\star$", "$n_\\star$"]
lab = np.array(
    [
    ["w/o sm, w/o l10", "w/ sm, w/o l10"], 
    ["w/o sm, w/ l10", "w/ sm, w/ l10"]
    ]
)


data = [chi2, fob]
labcols = ["chi2", "fob"]

for jj in range(2):
    
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12, 9))
    col = data[jj]
    aa = []
    for i0 in range(2):
        for i1 in range(2):
            _ = ax[i0, i1].scatter(arr_Ap, arr_np, c=col[i0, i1])
    
            for ii in range(len(archive.list_sim_cube)):
                ax[i0, i1].annotate(
                    archive.list_sim_cube[ii],
                    (arr_Ap[ii], arr_np[ii]),
                    textcoords="offset points",
                    xytext=(0, -15),
                    size=7,
                    ha="center",
                )
    
            plt.colorbar(_, ax=ax[i0, i1])
            ax[i0, i1].set_title(lab[i0, i1])
            
    plt.suptitle(labcols[jj])
            
    ax[0, 0].set_ylabel(flags[1])
    ax[1, 0].set_ylabel(flags[1])
    ax[1, 0].set_xlabel(flags[0])
    ax[1, 1].set_xlabel(flags[0])
    plt.tight_layout()
    folder = path_fig + '/precision/eBOSS/'
    plt.savefig(folder + labcols[jj] + '.png')
    

# %% [markdown]
# #### Diference between emulator and data

# %%
zz = np.array(zz)

# %%

lab = np.array(
    [
    ["w/o sm, w/o l10", "w/ sm, w/o l10"], 
    ["w/o sm, w/ l10", "w/ sm, w/ l10"]
    ]
)
for isim, sim_label in enumerate(archive.list_sim_cube):
# for isim in range(5, 6):
    
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12, 9))
    
    for i0 in range(2):
        for i1 in range(2):
            for ii in range(nz):
                ax[i0, i1].plot(kMpc, reldiff_P1D[i0, i1, isim, ii], label=str(zz[ii]))
        
            ax[i0, i1].axhline(color='k', ls='--')
            ax[i0, i1].axhline(0.01, color='k', ls='--')
            ax[i0, i1].axhline(-0.01, color='k', ls='--')
            ax[i0, i1].set_title(lab[i0, i1])
    ax[0, 0].legend(ncol=5)
    ax[0, 0].set_xscale('log')
    # ax[0, 0].set_ylim(-0.045, 0.045)
    ax[0, 0].set_ylabel("model/data-1")
    ax[1, 0].set_ylabel("model_l10/data-1")
    ax[1, 0].set_xlabel("kpar [1/Mpc]")
    ax[1, 1].set_xlabel("kpar [1/Mpc]")

    plt.tight_layout()
    folder = path_fig + '/precision/eBOSS/'
    plt.savefig(folder + 'reldiff_' + sim_label + '.png')

# %% [markdown]
# #### weights

# %%

lab = np.array(
    [
    ["w/o sm, w/o l10", "w/ sm, w/o l10"], 
    ["w/o sm, w/ l10", "w/ sm, w/ l10"]
    ]
)

for isim, sim_label in enumerate(archive.list_sim_cube):
# for isim in range(28, 29):
    
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12, 9))
    
    for i0 in range(2):
        for i1 in range(2):
            for ii in range(nz):
                ax[i0, i1].plot(kMpc, wdiff_P1D[i0, i1, isim, ii], label=str(zz[ii]))
        
    
            ax[i0, i1].axhline(color='k', ls='--')
            ax[i0, i1].axhline(0.5, color='k', ls='--')
            ax[i0, i1].axhline(-0.5, color='k', ls='--')
            ax[i0, i1].set_title(lab[i0, i1])
    ax[0, 0].legend(ncol=5)
    ax[0, 0].set_xscale('log')
    # ax[0, 0].set_ylim(-0.7, 1)
    ax[0, 0].set_ylabel("(model-data)/err_eBOSS")
    ax[1, 0].set_ylabel("(model_l10-data)/err_eBOSS")
    ax[1, 0].set_xlabel("kpar [1/Mpc]")
    ax[1, 1].set_xlabel("kpar [1/Mpc]")

    plt.tight_layout()
    folder = path_fig + '/precision/eBOSS/'
    plt.savefig(folder + 'wdiff_' + sim_label + '.png')

# %%
