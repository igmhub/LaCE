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

# %% [markdown]
# Traditional

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

# %% [markdown]
# New

# %%
full_emulator_p21_ext = GPEmulator(archive=archive_c23, emulator_label='Pedersen21_ext')
full_emulator_p23_ext = GPEmulator(archive=archive_c23, emulator_label='Pedersen23_ext')

# %%
full_emulator_c23_ext = NNEmulator(
    archive=archive_c23,
    training_set='Cabayol23',
    emulator_label='Cabayol23_extended',
    model_path='NNmodels/Cabayol23/Cabayol23_extended.pt',
    train=False
)

# %%
full_emulator_p21_ext8 = GPEmulator(archive=archive_c23, emulator_label='Pedersen21_ext8')
full_emulator_p23_ext8 = GPEmulator(archive=archive_c23, emulator_label='Pedersen23_ext8')

# %%
full_emulator_ch24 = GPEmulator(archive=archive_c23, emulator_label='k_bin_sm')
full_emulator_ch24 = GPEmulator(archive=archive_c23, kmax_Mpc=4, 
                                emu_type="k_bin_sm", bn=[0.8, 0.4, 0.2], klist=[0.15, 1, 2.5, 4])


# %% [markdown]
# ForestFlow

# %%
use_forestflow = False
if use_forestflow:

    import forestflow
    from forestflow.archive import GadgetArchive3D
    from forestflow.P3D_cINN import P3DEmulator
    
    folder_lya_data = os.path.dirname(forestflow.__path__[0]) + "/data/best_arinyo/"
    
    Archive3D = GadgetArchive3D(
        base_folder=os.path.dirname(forestflow.__path__[0]),
        folder_data=folder_lya_data,
        force_recompute_plin=False,
        average="both",
    )
    print(len(Archive3D.training_data))
    
    p3d_emu = P3DEmulator(
        Archive3D.training_data,
        Archive3D.emu_params,
        nepochs=300,
        lr=0.001,  # 0.005
        batch_size=20,
        step_size=200,
        gamma=0.1,
        weight_decay=0,
        adamw=True,
        nLayers_inn=12,  # 15
        Archive=Archive3D,
        chain_samp=100_000,
        model_path=os.path.dirname(forestflow.__path__[0]) + "/data/emulator_models/mpg_hypercube.pt",
    )

# %%
full_emulator = [
    full_emulator_c23, 
    # full_emulator_p21, 
    # full_emulator_p23, 
    full_emulator_p21_ext, 
    full_emulator_p23_ext,
    # full_emulator_c23_ext,
    # full_emulator_p21_ext8, 
    # full_emulator_p23_ext8,
    full_emulator_ch24,
]

# emu_labs = ['C23', 'P21', 'P23', 'P21_ext', 'P23_ext', 'C23_ext', 'P21_ext8', 'P23_ext8', 'CH24']
# arr_kmax = [4, 3, 3, 4, 4, 8, 8, 8, 4]
# arr_sm =   [1, 0, 1, 0, 1, 1, 0, 1, 2]
emu_labs = ['C23',  'P21_ext', 'P23_ext', 'CH24']
arr_kmax = [4, 4, 4, 4]
arr_sm =   [1, 0, 1, 1]
arr_ids =  [0, 0, 0, 0]
arr_l1O =  [0, 3, 4, 8]
arr_training_data = [
    training_data_c23,
    # training_data_p21,
    # training_data_p21,
    training_data_c23,
    training_data_c23,
    # training_data_c23,
    # training_data_c23,
    # training_data_c23,
    training_data_c23,
]

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

# %%
range_par = {
    'Delta2_p': 0.07,
    'n_p': 0.1,
    'mF': 0.02,
    'gamma': 0.2,
    'sigT_Mpc': 0.02,
    'kF_Mpc': 2
}

# %% [markdown]
# Range of Dp values in the sim

# %%
for kk, par in enumerate(input):
    print(kk, par, range_par[par])

# Dp_range = np.array([10., -10.])
# for ii in range(len(archive.data)):
#     dp = archive.data[ii]['Delta2_p']
#     if(dp > Dp_range[1]):
#         Dp_range[1] = dp
#     if(dp < Dp_range[0]):
#         Dp_range[0] = dp
# Dp_range

# fol = '/home/jchaves/Proyectos/projects/lya/data/cup1d/sampler/v1/Cabayol23_lres/emu_Cabayol23_cov_Chabanier2019_mocksim_mpg_5_cosmosim_mpg_5_igmsim_mpg_5_nigm_2_ydrop_ypoly/chain_1/'
# file = fol + 'chain.npy'
# res = np.load(file)
# Dp_range = np.percentile(res[:, -2], [5, 95])
# Dp_range

# %%
nn = 10
use_forestflow = False

training_data = training_data_c23
len_max = np.argwhere((training_data_c23[0]['k_Mpc'] > 0) & (training_data_c23[0]['k_Mpc'] < 8))[:,0].shape[0]

if use_forestflow:
    len_emus = len(full_emulator) + 1
else:
    len_emus = len(full_emulator)
    
orig_dp = np.zeros((len_emus, len_max))
var_dp = np.zeros((len_emus, 6, nn, len_max))
all_dp_vals = []
for jj in range(len_emus):
    
    mask = (training_data[0]['k_Mpc'] > 0) & (training_data[0]['k_Mpc'] < arr_kmax[jj])
    kMpc = training_data[0]['k_Mpc'][mask]

    
    if(jj != len(full_emulator)):
        orig_dp[jj, :kMpc.shape[0]] = full_emulator[jj].emulate_p1d_Mpc(input, kMpc)
        for kk, par in enumerate(input):
            dp_vals = np.linspace(input[par]-range_par[par], input[par]+range_par[par], nn)
            all_dp_vals.append(dp_vals)
            
            for ii in range(nn):
                input2 = input.copy()
                input2[par] = dp_vals[ii]
                var_dp[jj, kk, ii, :kMpc.shape[0]] = full_emulator[jj].emulate_p1d_Mpc(input2, kMpc)
    else:
        out = p3d_emu.predict_P3D_Mpc(
            sim_label='mpg_central',
            z=3, 
            emu_params=input,
            k_Mpc=np.linspace(.1, 1, 10),
            mu = np.zeros(10),
            kpar_Mpc = kMpc,
            return_cov=False,
        )
        orig_dp[jj, :kMpc.shape[0]] = out['p1d']

        for ii in range(nn):
            input2 = input.copy()
            input2['Delta2_p'] = dp_vals[ii]
            out = p3d_emu.predict_P3D_Mpc(
                sim_label='mpg_central',
                z=3, 
                emu_params=input2,
                k_Mpc=np.linspace(.1, 1, 10),
                mu = np.zeros(10),
                kpar_Mpc = kMpc,
                return_cov=False,
            )
            var_dp[jj, ii, :kMpc.shape[0]] = out['p1d']
        

# %%

for kk, par in enumerate(input):
    fig, ax = plt.subplots(len(emu_labs), sharex=True, sharey=True, figsize=(15,10))
    for jj in range(len(emu_labs)):
        for ii in range(nn):
            if(jj == 1):
                leg = par + '= ' + str(np.round(all_dp_vals[kk][ii],2))
            else:
                leg = None
                
            mask = (training_data[0]['k_Mpc'] > 0) & (training_data[0]['k_Mpc'] < arr_kmax[jj])
            kMpc = training_data[0]['k_Mpc'][mask]
                
            ax[jj].plot(kMpc, (var_dp[jj, kk, ii, :kMpc.shape[0]]/orig_dp[jj, :kMpc.shape[0]]-1), label=leg)
            
        if(jj == 1):
            ax[jj].legend(ncol=4)
            
        ax[jj].axhline(color='k')
        ax[jj].set_title(emu_labs[jj])
        ax[jj].set_ylabel('P1D/P1Dcen-1')
        
    ax[jj].set_xlabel(r'$k$[1/Mpc]')
    plt.suptitle(par)
    plt.tight_layout()
    plt.savefig(path_fig+'/sensitivity/'+par+'_red.pdf')

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


ids_all = [[],[]]

arr_Ap = np.zeros(len(archive_c23.list_sim_cube))
arr_np = np.zeros(len(archive_c23.list_sim_cube))

for iemu in range(2):
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
zz = np.array(zz)
zz

# %%
# emu_type, l10 (n, y), smooth (n, y), etc
reldiff_P1D = np.zeros((len(full_emulator), 2, 2, len(archive_c23.list_sim_cube), 11, len_max))
wdiff_P1D = np.zeros((len(full_emulator), 2, 2, len(archive_c23.list_sim_cube), 11, len_max))
chi2 = np.zeros((len(full_emulator), 2, 2, len(archive_c23.list_sim_cube)))
fob = np.zeros((len(full_emulator), 2, 2, len(archive_c23.list_sim_cube)))

for isim, sim_label in enumerate(archive_c23.list_sim_cube):
    print(sim_label)
    print("")
    print("")
    
    for iemu in range(len(full_emulator)):
        _ids_all = ids_all[arr_ids[iemu]]
        training_data = arr_training_data[iemu]
        kmax = arr_kmax[iemu]
        mask = np.argwhere((training_data[0]['k_Mpc'] > 0) & (training_data[0]['k_Mpc'] < kmax))[:,0]
        kMpc = training_data[0]['k_Mpc'][mask]
        
        if(arr_l1O[iemu] == 0):
            emulator = NNEmulator(
                archive=archive_c23,
                training_set='Cabayol23',
                emulator_label='Cabayol23',
                model_path='NNmodels/Cabayol23/Cabayol23_drop_sim_'+sim_label+'.pt', 
                drop_sim=sim_label,
                train=False
            )
        elif(arr_l1O[iemu] == 1):
            emulator = GPEmulator(archive=archive_p21, emulator_label='Pedersen21', drop_sim=sim_label)
        elif(arr_l1O[iemu] == 2):
            emulator = GPEmulator(archive=archive_p21, emulator_label='Pedersen23', drop_sim=sim_label)
        elif(arr_l1O[iemu] == 3):
            emulator = GPEmulator(archive=archive_c23, emulator_label='Pedersen21_ext', drop_sim=sim_label)
        elif(arr_l1O[iemu] == 4):
            emulator = GPEmulator(archive=archive_c23, emulator_label='Pedersen23_ext', drop_sim=sim_label)
        elif(arr_l1O[iemu] == 5):
            emulator = NNEmulator(
                archive=archive_c23,
                training_set='Cabayol23',
                emulator_label='Cabayol23_extended',
                model_path='NNmodels/Cabayol23/Cabayol23_extended_drop_sim_'+sim_label+'.pt', 
                drop_sim=sim_label,
                train=False
            )
        elif(arr_l1O[iemu] == 6):
            emulator = GPEmulator(archive=archive_c23, emulator_label='Pedersen21_ext8', drop_sim=sim_label)
        elif(arr_l1O[iemu] == 7):
            emulator = GPEmulator(archive=archive_c23, emulator_label='Pedersen23_ext8', drop_sim=sim_label)
        elif(arr_l1O[iemu] == 8):
            # emulator = GPEmulator(archive=archive_c23, emulator_label='k_bin_sm', drop_sim=sim_label)
            emulator = GPEmulator(archive=archive_c23, kmax_Mpc=4, drop_sim=sim_label,
                                            emu_type="k_bin_sm", bn=[0.8, 0.4, 0.2], klist=[0.15, 1, 2.5, 4])
    
        p1d_emu = np.zeros((2, 11, kMpc.shape[0]))
        p1d_sim = np.zeros((11, kMpc.shape[0]))
        p1d_sm = np.zeros((11, kMpc.shape[0]))
        err_p1d = np.zeros((11, kMpc.shape[0]))
        ndeg = p1d_emu.shape[1] + p1d_emu.shape[2] - len(emu_params)
        
        for ii, id in enumerate(_ids_all[isim]):
            p1d_emu[0, ii, :] = full_emulator[iemu].emulate_p1d_Mpc(training_data[id], kMpc) * kMpc / np.pi
            p1d_emu[1, ii, :] = emulator.emulate_p1d_Mpc(training_data[id], kMpc) * kMpc / np.pi
            
            p1d_sim[ii, :] = training_data[id]['p1d_Mpc'][mask] * kMpc / np.pi

            if(emu_labs[iemu] != "CH24"):
                fit_p1d = poly_p1d.PolyP1D(
                    kMpc, 
                    training_data[id]['p1d_Mpc'][mask],
                    kmin_Mpc=1.e-5,
                    kmax_Mpc=emulator.kmax_Mpc, 
                    deg=emulator.ndeg
                ).P_Mpc(kMpc)
            else:
                fit_p1d = full_emulator[iemu].apply_kernel_smoothing([training_data[id]], kmax)[0]
                
            p1d_sm[ii, :] = fit_p1d * kMpc / np.pi
    
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
lab = np.array([" w/o l10", " w/ l10"])

data = [chi2, fob]
labcols = ["chi2", "fob"]

for jj in range(2):        
    fig, ax = plt.subplots(len(full_emulator), 2, sharex=True, sharey=True, figsize=(12, 20))
    
    for iemu in range(len(full_emulator)):
        for i0 in range(2):
            i1 = arr_sm[iemu]
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
            ax[iemu, i0].set_title(emu_labs[iemu] + lab[i0])
        ax[iemu, 0].set_ylabel(flags[1])
                
    plt.suptitle(labcols[jj])
            
    ax[-1, 0].set_xlabel(flags[0])
    ax[-1, 1].set_xlabel(flags[0])
    plt.tight_layout()
    plt.savefig(path_fig + labcols[jj] + '_red.pdf')


# %% [markdown]
# #### Diference between emulator and data

# %%

lab = np.array([" w/o l10", " w/ l10"])


for isim, sim_label in enumerate(np.array(archive_c23.list_sim_cube)):
# for isim in range(5, 6):
    
    fig, ax = plt.subplots(len(full_emulator), 2, sharex=True, sharey=True, figsize=(12, 20))
    
    for i0 in range(len(full_emulator)):
        i2 = arr_sm[i0]
        training_data = arr_training_data[i0]
        kmax = arr_kmax[i0]
        
        for i1 in range(2):
            mask = np.argwhere((training_data[0]['k_Mpc'] > 0) & (training_data[0]['k_Mpc'] < kmax))[:,0]
            kMpc = training_data[0]['k_Mpc'][mask]
            
            for ii in range(11):
                ax[i0, i1].plot(kMpc, reldiff_P1D[i0, i1, i2, isim, ii, :kMpc.shape[0]], label=str(zz[ii]))
        
            ax[i0, i1].axhline(color='k', ls='--')
            ax[i0, i1].axhline(0.01, color='k', ls='--')
            ax[i0, i1].axhline(-0.01, color='k', ls='--')
            ax[i0, i1].set_title(emu_labs[i0] + lab[i1])
        ax[i0, 0].set_ylabel("m/d-1")
    ax[0, 0].legend(ncol=5)
    ax[0, 0].set_xscale('log')
    ax[0, 0].set_ylim([-0.04, 0.04])
    ax[-1, 0].set_xlabel("kpar [1/Mpc]")
    ax[-1, 1].set_xlabel("kpar [1/Mpc]")
    plt.suptitle(sim_label)

    plt.tight_layout()
    plt.savefig(path_fig + '/reldiff/' + sim_label + '_red.png')

# %% [markdown]
# #### weights

# %%

for isim, sim_label in enumerate(archive_c23.list_sim_cube):
# for isim in range(5, 6):
    
    fig, ax = plt.subplots(len(full_emulator), 2, sharex=True, sharey=True, figsize=(12, 20))
    
    for i0 in range(len(full_emulator)):
        i2 = arr_sm[i0]
        training_data = arr_training_data[i0]
        kmax = arr_kmax[i0]
        
        for i1 in range(2):
            mask = np.argwhere((training_data[0]['k_Mpc'] > 0) & (training_data[0]['k_Mpc'] < kmax))[:,0]
            kMpc = training_data[0]['k_Mpc'][mask]
            
            for ii in range(11):
                ax[i0, i1].plot(kMpc, wdiff_P1D[i0, i1, i2, isim, ii, :kMpc.shape[0]], label=str(zz[ii]))
        
            ax[i0, i1].axhline(color='k', ls='--')
            ax[i0, i1].axhline(0.5, color='k', ls='--')
            ax[i0, i1].axhline(-0.5, color='k', ls='--')
            ax[i0, i1].set_title(emu_labs[i0] + lab[i1])
        ax[i0, 0].set_ylabel("(m-d)/err")
    ax[0, 0].legend(ncol=5)
    ax[0, 0].set_xscale('log')
    ax[0, 0].set_ylim([-1.4, 1.4])
    ax[-1, 0].set_xlabel("kpar [1/Mpc]")
    ax[-1, 1].set_xlabel("kpar [1/Mpc]")
    plt.suptitle(sim_label)

    plt.tight_layout()
    plt.savefig(path_fig + '/wdiff/' + sim_label + '_red.png')

# %%

# %% [markdown]
# ## Extra sims

# %%

# %%
# emu_type, smooth (n, y), etc
nz = 11
len_max = np.argwhere((training_data_c23[0]['k_Mpc'] > 0) & (training_data_c23[0]['k_Mpc'] < 8))[:,0].shape[0]
ereldiff_P1D = np.zeros((len(full_emulator), 2, len(archive_c23.list_sim_test), nz, len_max))
ewdiff_P1D = np.zeros((len(full_emulator), 2, len(archive_c23.list_sim_test), nz, len_max))

for isim, sim_label in enumerate(archive_c23.list_sim_test):
    
    for iemu in range(3, len(full_emulator)):

        training_data = arr_training_data[iemu]
        kmax = arr_kmax[iemu]
        mask = np.argwhere((training_data[0]['k_Mpc'] > 0) & (training_data[0]['k_Mpc'] < kmax))[:,0]
        kMpc = training_data[0]['k_Mpc'][mask]
        
        kmax = arr_kmax[iemu]
        if(arr_ids[iemu] == 1):
            testing_data = archive_p21.get_testing_data(sim_label=sim_label)
        else:
            testing_data = archive_c23.get_testing_data(sim_label=sim_label)
    
        p1d_emu = np.zeros((nz, kMpc.shape[0]))
        p1d_sim = np.zeros((nz, kMpc.shape[0]))
        p1d_sm = np.zeros((nz, kMpc.shape[0]))
        err_p1d = np.zeros((nz, kMpc.shape[0]))
        ndeg = p1d_emu.shape[0] + p1d_emu.shape[1] - len(emu_params)
        
        for ii in range(len(testing_data)):
            p1d_emu[ii, :] = full_emulator[iemu].emulate_p1d_Mpc(testing_data[ii], kMpc) * kMpc / np.pi
            p1d_sim[ii, :] = testing_data[ii]['p1d_Mpc'][mask] * kMpc / np.pi
            
            if(emu_labs[iemu] != "CH24"):
                fit_p1d = poly_p1d.PolyP1D(
                    kMpc, 
                    testing_data[ii]['p1d_Mpc'][mask],
                    kmin_Mpc=1.e-5,
                    kmax_Mpc=full_emulator[iemu].kmax_Mpc, 
                    deg=full_emulator[iemu].ndeg
                ).P_Mpc(kMpc)
            else:
                fit_p1d = full_emulator[iemu].apply_kernel_smoothing([testing_data[ii]], kmax)[0]
                
            p1d_sm[ii, :] = fit_p1d * kMpc / np.pi
    
            k_kms = kMpc/testing_data[ii]['dkms_dMpc']
    
            indz_cov = np.argmin(np.abs(data_boss.z - testing_data[ii]['z']))
            err_p1d[ii, :] = np.interp(k_kms, data_boss.k_kms, norm_cov[indz_cov], right=1000)

            ereldiff_P1D[iemu, 0, isim, ii, :kMpc.shape[0]] = p1d_emu[ii, :]/p1d_sim[ii, :] - 1
            ereldiff_P1D[iemu, 1, isim, ii, :kMpc.shape[0]] = p1d_emu[ii, :]/p1d_sm[ii, :] - 1
            
            ewdiff_P1D[iemu, 0, isim, ii, :kMpc.shape[0]] = (p1d_emu[ii, :]-p1d_sim[ii, :])/err_p1d[ii, :]
            ewdiff_P1D[iemu, 1, isim, ii, :kMpc.shape[0]] = (p1d_emu[ii, :]-p1d_sm[ii, :])/err_p1d[ii, :]


# %%
archive_c23.list_sim_test

# %%
for isim, sim_label in enumerate(np.array(archive_c23.list_sim_test)):
    # if((sim_label == "mpg_central") | (sim_label == "mpg_seed")):
    #     pass
    # else:
    #     continue

    fig, ax = plt.subplots(len(full_emulator), 1, sharex=True, sharey=True, figsize=(12, 20))
    
    for i0 in range(len(full_emulator)):
        i2 = arr_sm[i0]
        kmax = arr_kmax[i0]
    
        mask = np.argwhere((testing_data[0]['k_Mpc'] > 0) & (testing_data[0]['k_Mpc'] < kmax))[:,0]
        kMpc = testing_data[0]['k_Mpc'][mask]
        
        for ii in range(11):
            ax[i0].plot(kMpc, ereldiff_P1D[i0, i2, isim, ii, :kMpc.shape[0]], label=str(zz[ii]))
    
        ax[i0].axhline(color='k', ls='--')
        ax[i0].axhline(0.01, color='k', ls='--')
        ax[i0].axhline(-0.01, color='k', ls='--')
        ax[i0].set_title(emu_labs[i0])
        ax[i0].set_ylabel("m/d-1")
    ax[0].legend(ncol=5)
    ax[0].set_xscale('log')
    ax[0].set_ylim([-0.02, 0.02])
    ax[-1].set_xlabel("kpar [1/Mpc]")
    plt.suptitle(sim_label)

    plt.tight_layout()
    plt.savefig(path_fig + '/extra_reldiff/' + sim_label + '_red.png')


    
    

# %%

# %%

# for isim, sim_label in enumerate(archive_c23.list_sim_test):
# # for isim in range(1):
#     # sim_label = 'mpg_central'
    
#     fig, ax = plt.subplots(len(full_emulator), 1, sharex=True, sharey=True, figsize=(12, 20))
    
#     for i0 in range(len(full_emulator)):
#         i2 = arr_sm[i0]
#         kmax = arr_kmax[i0]
                
#         mask = np.argwhere((testing_data[0]['k_Mpc'] > 0) & (testing_data[0]['k_Mpc'] < kmax))[:,0]
#         kMpc = testing_data[0]['k_Mpc'][mask]
        
#         for ii in range(nz):
            
#             ax[i0].plot(kMpc, ewdiff_P1D[i0, i2, isim, ii, :kMpc.shape[0]], label=str(zz[ii]))
        
#             ax[i0].axhline(color='k', ls='--')
#             ax[i0].axhline(0.25, color='k', ls='--')
#             ax[i0].axhline(-0.25, color='k', ls='--')            
#             ax[i0].set_title(emu_labs[i0])
#         ax[i0].set_ylabel("(m-d)/err")
        
#     ax[-1].legend(ncol=5, loc="lower right")
#     ax[0].set_xscale('log')
#     ax[0].set_ylim([-0.5, 0.5])
#     ax[-1].set_xlabel("kpar [1/Mpc]")

#     fig.suptitle(sim_label)

#     plt.tight_layout()
#     plt.savefig(path_fig + '/extra_wdiff/' + sim_label + '.png')

# %%
