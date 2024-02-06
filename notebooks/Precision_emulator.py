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
path_fig = '/home/jchaves/Proyectos/projects/lya/data/lace/precision/'

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
# Gadget archive with the post-processing using in Pedersen21
archive_p21 = gadget_archive.GadgetArchive(postproc="Pedersen21")
# Gadget archive with the post-processing using in Cabayol23
archive_c23 = gadget_archive.GadgetArchive(postproc="Cabayol23")

# %%
# list of emulator parameters used with Gadget sims
emu_params=['Delta2_p', 'n_p','mF', 'sigT_Mpc', 'gamma', 'kF_Mpc']

training_data_p21 = archive_p21.get_training_data(emu_params=emu_params)
training_data_c23 = archive_c23.get_training_data(emu_params=emu_params)

# %% [markdown]
# #### Get emulator

# %%
full_emulator_c23 = NNEmulator(
    archive=archive_c23,
    training_set='Cabayol23',
    emulator_label='Cabayol23',
    model_path='NNmodels/Cabayol23/Cabayol23.pt',
    train=False
)
full_emulator_p21 = GPEmulator(training_set='Pedersen21', emulator_label='Pedersen21')
full_emulator_p23 = GPEmulator(training_set='Pedersen21', emulator_label='Pedersen23')
full_emulator = [full_emulator_c23, full_emulator_p21, full_emulator_p23]

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

testing_data = archive_c23.get_testing_data(sim_label="mpg_central")
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
# Dp_range = np.array([10., -10.])
# for ii in range(len(archive.data)):
#     dp = archive.data[ii]['Delta2_p']
#     if(dp > Dp_range[1]):
#         Dp_range[1] = dp
#     if(dp < Dp_range[0]):
#         Dp_range[0] = dp
# Dp_range

fol = '/home/jchaves/Proyectos/projects/lya/data/cup1d/sampler/v1/Cabayol23_lres/emu_Cabayol23_cov_Chabanier2019_mocksim_mpg_5_cosmosim_mpg_5_igmsim_mpg_5_nigm_2_ydrop_ypoly/chain_1/'
file = fol + 'chain.npy'
res = np.load(file)
Dp_range = np.percentile(res[:, -2], [5, 95])
Dp_range

# %%
nn = 10
len_max = np.argwhere((training_data_c23[0]['k_Mpc'] > 0) & (training_data_c23[0]['k_Mpc'] < 4))[:,0].shape[0]
dp_vals = np.linspace(Dp_range[0], Dp_range[1], nn)
orig_dp = np.zeros((3, len_max))
var_dp = np.zeros((3, nn, len_max))
for jj in range(3):    
    if(jj == 0):
        training_data = training_data_c23
        kmax = 4
    else:
        training_data = training_data_p21
        kmax = 3
        
    mask = (training_data[0]['k_Mpc'] > 0) & (training_data[0]['k_Mpc'] < kmax)
    kMpc = training_data[0]['k_Mpc'][mask]
    
    orig_dp[jj, :kMpc.shape[0]] = full_emulator[jj].emulate_p1d_Mpc(input, kMpc)
    for ii in range(nn):
        input2 = input.copy()
        input2['Delta2_p'] = dp_vals[ii]
        var_dp[jj, ii, :kMpc.shape[0]] = full_emulator[jj].emulate_p1d_Mpc(input2, kMpc)

# %%
emu_labs = ['C23', 'P21', 'P23'] 
fig, ax = plt.subplots(3, sharex=True, sharey=True, figsize=(12,9))
for jj in range(3):
    for ii in range(nn):
        if(jj == 1):
            leg = r'$\Delta^2_p=$' + str(np.round(dp_vals[ii],2))
        else:
            leg = None    
        if(jj == 0):
            training_data = training_data_c23
            kmax = 4
        else:
            training_data = training_data_p21
            kmax = 3
            
        mask = (training_data[0]['k_Mpc'] > 0) & (training_data[0]['k_Mpc'] < kmax)
        kMpc = training_data[0]['k_Mpc'][mask]
            
        ax[jj].plot(kMpc, (var_dp[jj, ii, :kMpc.shape[0]]/orig_dp[jj, :kMpc.shape[0]]-1), label=leg)
    if(jj == 1):
        ax[jj].legend(ncol=3)
    ax[jj].axhline(color='k')
    ax[jj].set_title(emu_labs[jj])
    ax[jj].set_ylabel('P1D/P1Dcen-1')
ax[jj].set_xlabel(r'$k$[1/Mpc]')
plt.tight_layout()
plt.savefig(path_fig+'Ap_cosmo_dependence.png')

# %%
dp_range = np.zeros((3, orig_dp[0].shape[0]))
for ii in range(3):
    dp_range[ii] = np.max(var_dp[ii]/orig_dp[ii], axis=0) - np.min(var_dp[ii]/orig_dp[ii], axis=0)

# %% [markdown]
# # L1O
#
# We compute chi2 and bias.

# %% [markdown]
# #### Get IDs

# %%
val_scaling = 1


ids_all = [[],[],[]]

arr_Ap = np.zeros(len(archive_c23.list_sim_cube))
arr_np = np.zeros(len(archive_c23.list_sim_cube))

for iemu in range(3):
    if(iemu == 0):
        training_data = training_data_c23
    else:
        training_data = training_data_p21
        
    for isim, sim_label in enumerate(archive_c23.list_sim_cube):
    
        ids = []
        zz = []
        for ind_book in range(len(training_data)):
            if(
                (training_data[ind_book]['sim_label'] == sim_label) & 
                (training_data[ind_book]['ind_phase'] == 'average') & 
                (training_data[ind_book]['ind_axis'] == 'average') &  
                (training_data[ind_book]['val_scaling'] == val_scaling)
            ):                
                ids.append(ind_book)
                zz.append(training_data[ind_book]['z'])
                if(
                    (training_data[ind_book]['z'] == 3)
                ):
                    _id = ind_book

        arr_Ap[isim] = training_data[_id][emu_params[0]]
        arr_np[isim] = training_data[_id][emu_params[1]]
                    
    
        ids_all[iemu].append(ids)

# %%
# emu_type, l10 (n, y), smooth (n, y), etc
reldiff_P1D = np.zeros((3, 2, 2, len(archive_c23.list_sim_cube), 11, len_max))
wdiff_P1D = np.zeros((3, 2, 2, len(archive_c23.list_sim_cube), 11, len_max))
chi2 = np.zeros((3, 2, 2, len(archive_c23.list_sim_cube)))
fob = np.zeros((3, 2, 2, len(archive_c23.list_sim_cube)))


for isim, sim_label in enumerate(archive_c23.list_sim_cube):
    
    for iemu in range(3):
        if(iemu == 0):
            training_data = training_data_c23
            kmax = 4
        else:
            training_data = training_data_p21
            kmax = 3
        mask = np.argwhere((training_data[0]['k_Mpc'] > 0) & (training_data[0]['k_Mpc'] < kmax))[:,0]
        kMpc = training_data[0]['k_Mpc'][mask]

        if(iemu == 0):
            emulator = NNEmulator(
                archive=archive_c23,
                training_set='Cabayol23',
                emulator_label='Cabayol23',
                model_path='NNmodels/Cabayol23/Cabayol23_drop_sim_'+sim_label+'.pt', 
                drop_sim=sim_label,
                train=False
            )
            training_data = training_data_c23
        elif(iemu ==1):
            emulator = GPEmulator(training_set='Pedersen21', emulator_label='Pedersen21', drop_sim=sim_label)
        elif(iemu ==2):
            emulator = GPEmulator(training_set='Pedersen21', emulator_label='Pedersen23', drop_sim=sim_label)
    
        p1d_emu = np.zeros((2, 11, kMpc.shape[0]))
        p1d_sim = np.zeros((11, kMpc.shape[0]))
        p1d_sm = np.zeros((11, kMpc.shape[0]))
        err_p1d = np.zeros((11, kMpc.shape[0]))
        ndeg = p1d_emu.shape[1] + p1d_emu.shape[2] - len(emu_params)
        
        for ii, id in enumerate(ids_all[iemu][isim]):
            p1d_emu[0, ii, :] = full_emulator[iemu].emulate_p1d_Mpc(training_data[id], kMpc) * kMpc / np.pi
            p1d_emu[1, ii, :] = emulator.emulate_p1d_Mpc(training_data[id], kMpc) * kMpc / np.pi
            
            fit_p1d = poly_p1d.PolyP1D(
                kMpc, 
                training_data[id]['p1d_Mpc'][mask],
                kmin_Mpc=1.e-5,
                # kmin_Mpc=kMpc.min(), 
                kmax_Mpc=emulator.kmax_Mpc, 
                deg=emulator.ndeg
            )
            p1d_sim[ii, :] = training_data[id]['p1d_Mpc'][mask] * kMpc / np.pi
            p1d_sm[ii, :] = fit_p1d.P_Mpc(kMpc) * kMpc / np.pi
    
            k_kms = kMpc/training_data[id]['dkms_dMpc']
    
            indz_cov = np.argmin(np.abs(data_boss.z - training_data[id]['z']))
            err_p1d[ii, :] = np.interp(k_kms, data_boss.k_kms, norm_cov[indz_cov], right=1000)
    
            for i0 in range(2):
                reldiff_P1D[iemu, i0, 0, isim, ii, :kMpc.shape[0]] = p1d_emu[i0, ii, :]/p1d_sim[ii, :] - 1
                reldiff_P1D[iemu, i0, 1, isim, ii, :kMpc.shape[0]] = p1d_emu[i0, ii, :]/p1d_sm[ii, :] - 1
                
                wdiff_P1D[iemu, i0, 0, isim, ii, :kMpc.shape[0]] = (p1d_emu[i0, ii, :]-p1d_sim[ii, :])/err_p1d[ii, :]
                wdiff_P1D[iemu, i0, 1, isim, ii, :kMpc.shape[0]] = (p1d_emu[i0, ii, :]-p1d_sm[ii, :])/err_p1d[ii, :]
    
        for i0 in range(2):
            chi2[iemu, i0, 0, isim] = np.sum((p1d_emu[i0]-p1d_sim)**2/err_p1d**2)/ndeg
            chi2[iemu, i0, 1, isim] = np.sum((p1d_emu[i0]-p1d_sm)**2/err_p1d**2)/ndeg
            fob[iemu, i0, 0, isim]  = np.sum((p1d_emu[i0]-p1d_sim)/err_p1d)/ndeg
            fob[iemu, i0, 1, isim]  = np.sum((p1d_emu[i0]-p1d_sm)/err_p1d)/ndeg
    

# %% [markdown]
# #### Convex hull

# %%
flags = ["$\\Delta^2_\\star$", "$n_\\star$"]
lab = np.array(
    [
    ["C23 w/o l10", "C23 w/ l10"], 
    ["P21 w/o l10", "P21 w/ l10"], 
    ["P23 w/o l10", "P23 w/ l10"], 
    ]
)

data = [chi2, fob]
labcols = ["chi2", "fob"]

for jj in range(2):        
    fig, ax = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(12, 9))
    
    for iemu in range(3):
        for i0 in range(2):
            i1 = 1
            _ = ax[iemu, i0].scatter(arr_Ap, arr_np, c=data[jj][iemu, i0, i1])
    
            for ii in range(len(archive_c23.list_sim_cube)):
                ax[iemu, i0].annotate(
                    archive_c23.list_sim_cube[ii],
                    (arr_Ap[ii], arr_np[ii]),
                    textcoords="offset points",
                    xytext=(0, -15),
                    size=7,
                    ha="center",
                )
    
            plt.colorbar(_, ax=ax[iemu, i0])
            ax[iemu, i0].set_title(lab[iemu, i0])
                
    plt.suptitle(labcols[jj])
            
    ax[0, 0].set_ylabel(flags[1])
    ax[1, 0].set_ylabel(flags[1])
    ax[2, 0].set_ylabel(flags[1])
    ax[2, 0].set_xlabel(flags[0])
    ax[2, 1].set_xlabel(flags[0])
    plt.tight_layout()
    plt.savefig(path_fig + labcols[jj] + '.png')


# %% [markdown]
# #### Diference between emulator and data

# %%
zz = np.array(zz)
zz

# %%

lab = np.array(
    [
    ["C23 w/o l10", "C23 w/ l10"], 
    ["P21 w/o l10", "P21 w/ l10"], 
    ["P23 w/o l10", "P23 w/ l10"], 
    ]
)

for isim, sim_label in enumerate(archive_c23.list_sim_cube):
# for isim in range(5, 6):
    
    fig, ax = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(12, 9))
    
    for i0 in range(3):
        if(i0 == 0):
            training_data = training_data_c23
            kmax = 4
            i2 = 1
        else:
            training_data = training_data_p21
            kmax = 3
            if(i0 == 1):                    
                i2 = 0
            else:
                i2 = 1
        for i1 in range(2):
            mask = np.argwhere((training_data[0]['k_Mpc'] > 0) & (training_data[0]['k_Mpc'] < kmax))[:,0]
            kMpc = training_data[0]['k_Mpc'][mask]
            
            for ii in range(11):
                ax[i0, i1].plot(kMpc, reldiff_P1D[i0, i1, i2, isim, ii, :kMpc.shape[0]], label=str(zz[ii]))
        
            ax[i0, i1].axhline(color='k', ls='--')
            ax[i0, i1].axhline(0.01, color='k', ls='--')
            ax[i0, i1].axhline(-0.01, color='k', ls='--')
            ax[i0, i1].set_title(lab[i0, i1])
    ax[0, 0].legend(ncol=5)
    ax[0, 0].set_xscale('log')
    ax[0, 0].set_ylabel("model/data-1")
    ax[1, 0].set_ylabel("model/data-1")
    ax[2, 0].set_ylabel("model/data-1")
    ax[2, 0].set_xlabel("kpar [1/Mpc]")
    ax[2, 1].set_xlabel("kpar [1/Mpc]")

    plt.tight_layout()
    plt.savefig(path_fig + '/reldiff/' + sim_label + '.png')

# %% [markdown]
# #### weights

# %%

lab = np.array(
    [
    ["C23 w/o l10", "C23 w/ l10"], 
    ["P21 w/o l10", "P21 w/ l10"], 
    ["P23 w/o l10", "P23 w/ l10"], 
    ]
)

for isim, sim_label in enumerate(archive_c23.list_sim_cube):
# for isim in range(5, 6):
    
    fig, ax = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(12, 9))
    
    for i0 in range(3):
        if(i0 == 0):
            training_data = training_data_c23
            kmax = 4
            i2 = 1
        else:
            training_data = training_data_p21
            kmax = 3
            if(i0 == 1):                    
                i2 = 0
            else:
                i2 = 1
        for i1 in range(2):
            mask = np.argwhere((training_data[0]['k_Mpc'] > 0) & (training_data[0]['k_Mpc'] < kmax))[:,0]
            kMpc = training_data[0]['k_Mpc'][mask]
            
            for ii in range(11):
                ax[i0, i1].plot(kMpc, wdiff_P1D[i0, i1, i2, isim, ii, :kMpc.shape[0]], label=str(zz[ii]))
        
            ax[i0, i1].axhline(color='k', ls='--')
            ax[i0, i1].axhline(0.5, color='k', ls='--')
            ax[i0, i1].axhline(-0.5, color='k', ls='--')
            ax[i0, i1].set_title(lab[i0, i1])
    ax[0, 0].legend(ncol=5)
    ax[0, 0].set_xscale('log')
    ax[0, 0].set_ylabel("(model-data)/err")
    ax[1, 0].set_ylabel("(model-data)/err")
    ax[2, 0].set_ylabel("(model-data)/err")
    ax[2, 0].set_xlabel("kpar [1/Mpc]")
    ax[2, 1].set_xlabel("kpar [1/Mpc]")

    plt.tight_layout()
    plt.savefig(path_fig + '/wdiff/' + sim_label + '.png')

# %%

# %% [markdown]
# ## Extra sims

# %%
# emu_type, smooth (n, y), etc
ereldiff_P1D = np.zeros((3, 2, len(archive_c23.list_sim_test), 11, len_max))
ewdiff_P1D = np.zeros((3, 2, len(archive_c23.list_sim_test), 11, len_max))

for isim, sim_label in enumerate(archive_c23.list_sim_test):
    testing_data = archive_c23.get_testing_data(sim_label=sim_label)
    
    for iemu in range(3):
        if(iemu == 0):
            kmax = 4
        else:
            kmax = 3
        
        mask = np.argwhere((testing_data[0]['k_Mpc'] > 0) & (testing_data[0]['k_Mpc'] < kmax))[:,0]
        kMpc = testing_data[0]['k_Mpc'][mask]
    
        p1d_emu = np.zeros((11, kMpc.shape[0]))
        p1d_sim = np.zeros((11, kMpc.shape[0]))
        p1d_sm = np.zeros((11, kMpc.shape[0]))
        err_p1d = np.zeros((11, kMpc.shape[0]))
        ndeg = p1d_emu.shape[0] + p1d_emu.shape[1] - len(emu_params)
        
        for ii in range(len(testing_data)):
            p1d_emu[ii, :] = full_emulator[iemu].emulate_p1d_Mpc(testing_data[ii], kMpc) * kMpc / np.pi
            
            fit_p1d = poly_p1d.PolyP1D(
                kMpc, 
                testing_data[ii]['p1d_Mpc'][mask],
                kmin_Mpc=1.e-5,
                kmax_Mpc=full_emulator[iemu].kmax_Mpc, 
                deg=full_emulator[iemu].ndeg
            )
            p1d_sim[ii, :] = testing_data[ii]['p1d_Mpc'][mask] * kMpc / np.pi
            p1d_sm[ii, :] = fit_p1d.P_Mpc(kMpc) * kMpc / np.pi
    
            k_kms = kMpc/testing_data[ii]['dkms_dMpc']
    
            indz_cov = np.argmin(np.abs(data_boss.z - testing_data[ii]['z']))
            err_p1d[ii, :] = np.interp(k_kms, data_boss.k_kms, norm_cov[indz_cov], right=1000)

            ereldiff_P1D[iemu, 0, isim, ii, :kMpc.shape[0]] = p1d_emu[ii, :]/p1d_sim[ii, :] - 1
            ereldiff_P1D[iemu, 1, isim, ii, :kMpc.shape[0]] = p1d_emu[ii, :]/p1d_sm[ii, :] - 1
            
            ewdiff_P1D[iemu, 0, isim, ii, :kMpc.shape[0]] = (p1d_emu[ii, :]-p1d_sim[ii, :])/err_p1d[ii, :]
            ewdiff_P1D[iemu, 1, isim, ii, :kMpc.shape[0]] = (p1d_emu[ii, :]-p1d_sm[ii, :])/err_p1d[ii, :]
    

# %%

# %%

lab = np.array(
    [
    ["C23 w/o l10", "C23 w/ l10"], 
    ["P21 w/o l10", "P21 w/ l10"], 
    ["P23 w/o l10", "P23 w/ l10"], 
    ]
)

for isim, sim_label in enumerate(archive_c23.list_sim_test):
# for isim in range(5, 6):
    
    fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(12, 9))
    
    for i0 in range(3):
        if(i0 == 0):
            kmax = 4
            i2 = 1
        else:
            kmax = 3
            if(i0 == 1):
                i2 = 0
            else:
                i2 = 1
                
        mask = np.argwhere((testing_data[0]['k_Mpc'] > 0) & (testing_data[0]['k_Mpc'] < kmax))[:,0]
        kMpc = testing_data[0]['k_Mpc'][mask]
        
        for ii in range(11):
            
            ax[i0].plot(kMpc, ereldiff_P1D[i0, i2, isim, ii, :kMpc.shape[0]], label=str(zz[ii]))
        
            ax[i0].axhline(color='k', ls='--')
            ax[i0].axhline(0.01, color='k', ls='--')
            ax[i0].axhline(-0.01, color='k', ls='--')
            ax[i0].set_title(lab[i0, i1])
    ax[0].legend(ncol=5)
    ax[0].set_xscale('log')
    ax[0].set_ylabel("model/data-1")
    ax[1].set_ylabel("model/data-1")
    ax[2].set_ylabel("model/data-1")
    ax[2].set_xlabel("kpar [1/Mpc]")

    plt.tight_layout()
    plt.savefig(path_fig + '/extra_reldiff/' + sim_label + '.png')

# %%

lab = np.array(
    [
    ["C23 w/o l10", "C23 w/ l10"], 
    ["P21 w/o l10", "P21 w/ l10"], 
    ["P23 w/o l10", "P23 w/ l10"], 
    ]
)

for isim, sim_label in enumerate(archive_c23.list_sim_test):
# for isim in range(5, 6):
    
    fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(12, 9))
    
    for i0 in range(3):
        if(i0 == 0):
            kmax = 4
            i2 = 1
        else:
            kmax = 3
            if(i0 == 1):
                i2 = 0
            else:
                i2 = 1
                
        mask = np.argwhere((testing_data[0]['k_Mpc'] > 0) & (testing_data[0]['k_Mpc'] < kmax))[:,0]
        kMpc = testing_data[0]['k_Mpc'][mask]
        
        for ii in range(11):
            
            ax[i0].plot(kMpc, ewdiff_P1D[i0, i2, isim, ii, :kMpc.shape[0]], label=str(zz[ii]))
        
            ax[i0].axhline(color='k', ls='--')
            ax[i0].axhline(1, color='k', ls='--')
            ax[i0].axhline(-1, color='k', ls='--')
            ax[i0].set_title(lab[i0, i1])
    ax[0].legend(ncol=5)
    ax[0].set_xscale('log')
    ax[0].set_ylabel("(model-data)/err")
    ax[1].set_ylabel("(model-data)/err")
    ax[2].set_ylabel("(model-data)/err")
    ax[2].set_xlabel("kpar [1/Mpc]")

    plt.tight_layout()
    plt.savefig(path_fig + '/extra_wdiff/' + sim_label + '.png')

# %%
