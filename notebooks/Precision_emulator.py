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
# %load_ext autoreload
# %autoreload 2


import os, sys
import numpy as np
from matplotlib import pyplot as plt

# our modules00
from lace.archive import gadget_archive, nyx_archive
from lace.emulator.emulator_manager import set_emulator
from lace.emulator.gp_emulator_new import GPEmulator
from lace.emulator.nn_emulator import NNEmulator
from lace.utils import poly_p1d
import lace
from cup1d.likelihood.pipeline import set_archive

from cup1d.p1ds.data_Chabanier2019 import P1D_Chabanier2019

# %%
from scipy.linalg import block_diag

# %%

# %%
path_fig = '/home/jchaves/Proyectos/projects/lya/data/lace/precision/'

# %% [markdown]
# #### Read eBOSS covariance matrix

# %%
data_boss = P1D_Chabanier2019()

norm_cov = np.zeros((len(data_boss.cov_Pk_kms), data_boss.cov_Pk_kms[0].shape[0]))
for ii in range(len(data_boss.cov_Pk_kms)):
    norm_cov[ii] = np.sqrt(np.diagonal(data_boss.cov_Pk_kms[ii])) * data_boss.k_kms[ii] / np.pi

# %% [markdown]
# #### Read sim data

# %%
# Gadget archive with the post-processing using in Pedersen21
# archive_p21 = gadget_archive.GadgetArchive(postproc="Pedersen21")
# Gadget archive with the post-processing using in Cabayol23
archive_c23 = gadget_archive.GadgetArchive(postproc="Cabayol23")

# %%

training_set = "Nyx23_Jul2024"
archive_nyx = set_archive(training_set)

# %%
# # let X, Y be data loaded above
# # Model creation:
# m = GPy.models.GPRegression(X, Y)
# m.optimize()
# # 1: Saving a model:
# np.save('model_save.npy', m.param_array)
# # 2: loading a model
# # Model creation, without initialization:
# m_load = GPy.models.GPRegression(X, Y, initialize=False)
# m_load.update_model(False) # do not call the underlying expensive algebra on load
# m_load.initialize_parameter() # Initialize the parameters (connect the parameters up)
# m_load[:] = np.load('model_save.npy') # Load the parameters
# m_load.update_model(True) # Call the algebra only once
# print(m_load)

# %%
# list of emulator parameters used with Gadget sims
emu_params=['Delta2_p', 'n_p','mF', 'sigT_Mpc', 'gamma', 'kF_Mpc']

# training_data_p21 = archive_p21.get_training_data(emu_params=emu_params)
training_data_c23 = archive_c23.get_training_data(emu_params=emu_params, average="both")

# %%

training_data_nyx = archive_nyx.get_training_data(emu_params=emu_params, average="both")

# %% [markdown]
# #### Get emulator

# %% [markdown]
# Traditional

# %%
# full_emulator_c23 = NNEmulator(
#     archive=archive_c23,
#     training_set='Cabayol23',
#     emulator_label='Cabayol23',
#     # model_path='NNmodels/Cabayol23/Cabayol23.pt',
#     model_path='NNmodels/Cabayol23_Feb2024/Cabayol23.pt',
#     train=False
# )
# full_emulator_p21 = GPEmulator(training_set='Pedersen21', emulator_label='Pedersen21')
# full_emulator_p23 = GPEmulator(training_set='Pedersen21', emulator_label='Pedersen23')

# full_emulator_c23plus = NNEmulator(
#     archive=archive_c23,
#     training_set='Cabayol23',
#     emulator_label='Cabayol23+',
#     model_path='NNmodels/Cabayol23+/Cabayol23+.pt',
#     train=False
# )

full_emulator_p21_ext = GPEmulator(archive=archive_c23, emulator_label='Pedersen21_ext')
full_emulator_p23_ext = GPEmulator(archive=archive_c23, emulator_label='Pedersen23_ext')


emulator_label = "Cabayol23+"
model_path = "NNmodels" +"/"+ emulator_label +"/"+ emulator_label + ".pt"
full_emulator_c23plus = NNEmulator(
        archive=archive_c23,
        model_path=model_path, 
        emulator_label=emulator_label, 
        train=False,
    )


# %%

# %%
emulator_label = 'CH24'
full_emulator_ch24 = NNEmulator(
    model_path="NNmodels" +"/"+ emulator_label +"/"+ emulator_label + ".pt", 
    emulator_label=emulator_label, 
    train=False,
)

# %%

CH24_mpg_gp = GPEmulator(emulator_label="CH24_mpg_gp")
CH24_nyx_gp = GPEmulator(emulator_label="CH24_nyx_gp")


emulator_label = 'CH24'
full_emulator_ch24 = NNEmulator(
    model_path="NNmodels" +"/"+ emulator_label +"/"+ emulator_label + ".pt", 
    emulator_label=emulator_label, 
    train=False,
)

emulator_label = 'CH24_NYX'
full_emulator_ch24_nyx = NNEmulator(
    model_path="NNmodels" +"/"+ emulator_label +"/"+ emulator_label + ".pt", 
    emulator_label=emulator_label, 
    train=False,
)


# var_emu_label = 'CH24_NYX_MPGlike'
# model_path="NNmodels" +"/"+ emulator_label +"/"+ var_emu_label + ".pt"
# full_emulator_ch24_nyx2 = NNEmulator(
#     model_path=model_path, 
#     emulator_label=emulator_label, 
#     train=False,
# )


# var_emu_label = 'CH24_NYX_MPGlikez'
# model_path="NNmodels" +"/"+ emulator_label +"/"+ var_emu_label + ".pt"
# full_emulator_ch24_nyx3 = NNEmulator(
#     model_path=model_path, 
#     emulator_label=emulator_label, 
#     train=False,
# )

# %%

emulator_label = "Nyx_alphap_cov"
model_path = "NNmodels/Nyxap_Jul2024_cov/"+ emulator_label + ".pt"
full_emulator_nyx = NNEmulator(
        archive=archive_nyx,
        model_path=model_path, 
        emulator_label=emulator_label, 
        train=False,
    )

# %%

# %%
# full_emulator_c23_ext = NNEmulator(
#     archive=archive_c23,
#     training_set='Cabayol23',
#     emulator_label='Cabayol23_extended',
#     model_path='NNmodels/Cabayol23/Cabayol23_extended.pt',
#     train=False
# )

# %%
# full_emulator_p21_ext8 = GPEmulator(archive=archive_c23, emulator_label='Pedersen21_ext8')
# full_emulator_p23_ext8 = GPEmulator(archive=archive_c23, emulator_label='Pedersen23_ext8')

# %% [markdown]
# ForestFlow

# %%
use_forestflow = True
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
        training_type='Arinyo_min',
        model_path=os.path.dirname(forestflow.__path__[0])+"/data/emulator_models/mpg_hypercube.pt",
    )

# %%
# full_emulator = [ 
#     p3d_emu,
#     full_emulator_ch24, 
#     full_emulator_c23plus,
#     full_emulator_p21_ext, 
#     full_emulator_p23_ext, 
#     full_emulator_nyx,
#     full_emulator_ch24_nyx,
# ]

full_emulator = [ 
    # p3d_emu,
    # full_emulator_c23plus,
    # full_emulator_p21_ext, 
    # full_emulator_p23_ext, 
    CH24_mpg_gp,
    CH24_nyx_gp,
    full_emulator_ch24,
    # full_emulator_ch24_nyx,
    full_emulator_ch24_nyx
    # full_emulator_ch24_nyx2,
    # full_emulator_ch24_nyx3
]

# emu_labs = ['C23', 'P21', 'P23', 'P21_ext', 'P23_ext', 'C23_ext', 'P21_ext8', 'P23_ext8', 'CH24']
# arr_kmax = [4, 3, 3, 4, 4, 8, 8, 8, 4]
# arr_sm =   [1, 0, 1, 0, 1, 1, 0, 1, 2]
# emu_labs = ['C23', 'C23+', 'P21_ext', 'P23_ext']
# emu_labs = ["forestflow", "CH24", "C23+", 'P21_ext', 'P23_ext',  "Nyx_alphap_cov", "CH24_NYX", ]
# emu_labs = ["forestflow", "CH24", "C23+", 'P21_ext', 'P23_ext' ]
# emu_labs = ["forestflow", "CH24", "C23+", 'P21_ext', 'P23_ext',  "Nyx_alphap_cov", "CH24_NYX" ]
emu_labs = [
    # "forestflow",
    # "C23+",
    # "P21_ext",
    # "P23_ext",
    "CH24_MPG_GP",
    "CH24_NYX_GP",
    "CH24_MPG_NN",
    "CH24_NYX_NN",
    # "Nyx"
    # "CH24_NYX_MPG_like",
    # "CH24_NYX_MPGZ_like",
]
arr_kmax = [4, 4, 4, 4, 4, 4, 4]
# arr_sm =   [0, 1, 1]
# arr_ids =  [0, 0, 0]
# arr_l1O =  [0, 0, 0]
# arr_training_data = [
#     training_data_c23,
#     # training_data_p21,
#     # training_data_p21,
#     training_data_c23,
#     training_data_c23,
#     # training_data_c23,
#     # training_data_c23,
#     # training_data_c23,
#     # training_data_c23,
# ]

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
    'alpha_p': -0.21539,
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
# range_par = {
#     'Delta2_p': 0.07,
#     'n_p': 0.1,
#     'mF': 0.02,
#     'gamma': 0.2,
#     'sigT_Mpc': 0.02,
#     'kF_Mpc': 2
# }

range_par = {
    'Delta2_p': 0.1 * input['Delta2_p'],
    'n_p': 0.05,
    'mF': 0.01,
    'gamma': 0.1,
    'sigT_Mpc': 0.02,
    'kF_Mpc': 2
}

# %% [markdown]
# Range of Dp values in the sim

# %%
for kk, par in enumerate(range_par):
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

# %% [markdown]
# ## Fig 10 Mcdonald 2003 paper approx

# %%
k3d = np.logspace(-1, np.log10(5), 20)
k3d_Mpc = np.zeros((20, 2))
mu = np.zeros_like(k3d_Mpc)
k3d_Mpc[:, 0] = k3d
k3d_Mpc[:, 1] = k3d
mu[:, 0] = 0
mu[:, 1] = 1

ztar = 3
info_power = {
    "z":ztar,
    "sim_label":"mpg_central",
    "k3d_Mpc": k3d_Mpc,
    "mu": mu,
    "return_p3d": True,
}

repo = os.path.dirname(lace.__path__[0]) + "/"
fname = repo + ("data/sim_suites/Australia20/mpg_emu_cosmo.npy")
data_cosmo = np.load(fname, allow_pickle=True).item()

ind = data_cosmo["mpg_central"]["linP_params"]["z"] == ztar
input = {
    'Delta2_p': data_cosmo["mpg_central"]["linP_params"]["Delta2_p"][ind][0],
    'n_p': data_cosmo["mpg_central"]["linP_params"]["n_p"][ind][0],
    'mF': 0.8711781695077168,
    'gamma': 1.7573934933568836,
    'sigT_Mpc': 0.1548976469710291,
    'kF_Mpc': 7.923157506298608,
    'alpha_p': data_cosmo["mpg_central"]["linP_params"]["alpha_p"][ind][0],
}



# %%
# %%time

da = 0.29
db = 0.1

# da = 0.029
# db = 0.01
Nrealizations = 1000

out = p3d_emu.evaluate(
    emu_params=input,
    info_power=info_power,
    Nrealizations=Nrealizations
)

input2 = input.copy()
input2["Delta2_p"] += da
out_at = p3d_emu.evaluate(
    emu_params=input2,
    info_power=info_power,
    Nrealizations=Nrealizations
)

input2 = input.copy()
input2["Delta2_p"] -= da
out_ab = p3d_emu.evaluate(
    emu_params=input2,
    info_power=info_power,
    Nrealizations=Nrealizations
)

input2 = input.copy()
input2["n_p"] += db
out_bt = p3d_emu.evaluate(
    emu_params=input2,
    info_power=info_power,
    Nrealizations=Nrealizations
)

input2 = input.copy()
input2["n_p"] -= db
out_bb = p3d_emu.evaluate(
    emu_params=input2,
    info_power=info_power,
    Nrealizations=Nrealizations
)


# %%
diff_pa = (out_at["p3d"] - out_ab["p3d"])/out["p3d"]
plt.plot(out["k_Mpc"][:, 0], diff_pa[:,0], "k:")
plt.plot(out["k_Mpc"][:, 1], diff_pa[:,1], "k-")


diff_pb = (out_bt["p3d"] - out_bb["p3d"])/out["p3d"]
plt.plot(out["k_Mpc"][:, 0], diff_pb[:,0], "r:")
plt.plot(out["k_Mpc"][:, 1], diff_pb[:,1], "r-")

plt.plot(out["k_Mpc"][:, 0], out["k_Mpc"][:, 0]*0, "k")
plt.xscale("log")
plt.axvline(5, color="k")


# %% [markdown]
# ### Differences

# %%
nn = 6

training_data = training_data_c23
len_max = np.argwhere((training_data_c23[0]['k_Mpc'] > 0) & (training_data_c23[0]['k_Mpc'] < 4))[:,0].shape[0]
len_emus = len(full_emulator)
    
orig_dp = np.zeros((len_emus, len_max))
var_dp = np.zeros((len_emus, 6, nn, len_max))
all_dp_vals = []
for jj in range(len_emus):
    print(emu_labs[jj])
    
    mask = (training_data[0]['k_Mpc'] > 0) & (training_data[0]['k_Mpc'] < arr_kmax[jj])
    kMpc = training_data[0]['k_Mpc'][mask]
    
    if(emu_labs[jj] != "forestflow"):
        p1d = full_emulator[jj].emulate_p1d_Mpc(input, kMpc)
    else:
        info_power = {
            "z":3,
            "sim_label":"mpg_central",
            "k1d_Mpc": kMpc,
            "return_p1d": True,
        }
        out = p3d_emu.evaluate(
            emu_params=input,
            info_power=info_power,
            Nrealizations=200
        )
        p1d = out['p1d']
    orig_dp[jj, :kMpc.shape[0]] = p1d

    
    for kk, par in enumerate(range_par):
        dp_vals = np.linspace(input[par]-range_par[par], input[par]+range_par[par], nn)
        all_dp_vals.append(dp_vals)
        
        for ii in range(nn):
            input2 = input.copy()
            input2[par] = dp_vals[ii]

            if(emu_labs[jj] != "forestflow"):
                p1d = full_emulator[jj].emulate_p1d_Mpc(input2, kMpc)
            else:
                out = p3d_emu.evaluate(
                    emu_params=input2,
                    info_power=info_power,
                    Nrealizations=200
                )
                p1d = out['p1d']

            var_dp[jj, kk, ii, :kMpc.shape[0]] = p1d


# %%

for kk, par in enumerate(range_par):
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
    ax[jj].set_xscale("log")
    plt.suptitle(par)
    plt.tight_layout()
    # plt.savefig(path_fig+'/sensitivity/last/'+par+'_red_new.pdf')
    plt.savefig(path_fig+'/sensitivity/ch24/'+par+'_red_new.png')

# %% [markdown]
# # Derivaties

# %%
repo = os.path.dirname(lace.__path__[0]) + "/"
fname = repo + ("data/sim_suites/Australia20/mpg_emu_cosmo.npy")
data_cosmo = np.load(fname, allow_pickle=True).item()

# %%
testing_data = archive_c23.get_testing_data("mpg_central")

# %%


# %% [markdown]
# ## Diff

# %%

pars = ['Delta2_p', 'n_p', 'mF', 'gamma', 'sigT_Mpc', 'kF_Mpc']
ztar = 3.

# in mpg and nyx
input = {
    'Delta2_p': 0.4, # 0.2-0.5
    'n_p': -2.3,
    'alpha_p': -0.215,
    'mF':0.5,
    'sigT_Mpc': 0.12,
    'gamma': 1.5,
    'kF_Mpc': 12.5,
}

range_par = {}
for pp in pars:
    range_par[pp] = 0.001 * np.abs(input[pp])
range_par['Delta2_p'] = 0.4 * 0.001

pos_par = np.linspace(0.2, 0.5, 5)

# %%
nn = 2
Nrealizations = 10000

# training_data = training_data_c23
# len_max = np.argwhere((training_data_c23[0]['k_Mpc'] > 0) & (training_data_c23[0]['k_Mpc'] < 4))[:,0].shape[0]
len_max = 100
kMpc = np.logspace(np.log10(0.05), np.log10(5), len_max) 
len_emus = len(full_emulator)
    
orig_dp = np.zeros((len_emus, pos_par.shape[0], len_max))
var_dp = np.zeros((len_emus, len(pars), pos_par.shape[0], nn, len_max))
all_dp_vals = []

for kk1 in range(len(pos_par)):
    input['Delta2_p'] = pos_par[kk1]
    for jj in range(len_emus):
        print(emu_labs[jj])
    
        # mask = (training_data[0]['k_Mpc'] > 0) & (training_data[0]['k_Mpc'] < arr_kmax[jj])
        # kMpc = training_data[0]['k_Mpc'][mask]
        
        if(emu_labs[jj] != "forestflow"):
            p1d = full_emulator[jj].emulate_p1d_Mpc(input, kMpc)
        else:
            info_power = {
                "z":ztar,
                "sim_label":"mpg_central",
                "k1d_Mpc": kMpc,
                "return_p1d": True,
            }
            out = p3d_emu.evaluate(
                emu_params=input,
                info_power=info_power,
                Nrealizations=Nrealizations
            )
            p1d = out['p1d']
        orig_dp[jj, kk1, :] = p1d
    
        
        for kk, par in enumerate(range_par):
            dp_vals = np.linspace(input[par]-range_par[par], input[par]+range_par[par], nn)
            all_dp_vals.append(dp_vals)
            
            for ii in range(nn):
                input2 = input.copy()
                input2[par] = dp_vals[ii]
    
                if(emu_labs[jj] != "forestflow"):
                    p1d = full_emulator[jj].emulate_p1d_Mpc(input2, kMpc)
                else:
                    out = p3d_emu.evaluate(
                        emu_params=input2,
                        info_power=info_power,
                        Nrealizations=Nrealizations
                    )
                    p1d = out['p1d']
    
                var_dp[jj, kk, kk1, ii, :] = p1d

# %%

# %%
jj = 0
kk = 0
# for kk, par in enumerate(range_par):
# for kk, par in enumerate(range_par):
plt.plot(kMpc, kMpc * var_dp[jj, kk, 0, 1, :])
plt.plot(kMpc, kMpc * var_dp[jj, kk, 0, 0, :])
plt.plot(kMpc, kMpc * orig_dp[jj, 0])
# plt.plot(kMpc, var_dp[jj, kk, 0, 0, :]/orig_dp[jj, 0])
# plt.plot(kMpc, var_dp[jj, kk, 0, 1, :]/orig_dp[jj, 0])
plt.xscale("log")

# %%
var_dp.shape

# %%
kk = 1
par = pars[kk]
for jj in range(var_dp.shape[0]):
# for jj in range(2):
    col = "C"+str(jj)
    for ii in range(var_dp.shape[2]):
        diffp1d = var_dp[jj, kk, ii, 1, :] - var_dp[jj, kk, ii, 0, :]
        diffh = 2*range_par[par]
        der = diffp1d/diffh/orig_dp[jj, ii]
        # der = var_dp[jj, kk, ii, 1, :]/orig_dp[jj, ii]
        if ii == 0:
            lab = emu_labs[jj]
        else:
            lab = None
        plt.plot(kMpc, der, col, label=lab)
plt.legend()
plt.xscale("log")

# %%

# %%

# %% [markdown]
# # range_par

# %%
from scipy.optimize import curve_fit

# %%
emu_params = ['Delta2_p', 'n_p', 'mF', 'sigT_Mpc', 'gamma', 'kF_Mpc']
# archive = archive_c23
archive = archive_nyx
# training_data = archive.get_training_data(emu_params=emu_params, average="both", val_scaling=None)
# training_data = archive.get_testing_data("mpg_central")
training_data = archive.get_testing_data("nyx_central")
len(training_data)

# %%
CH24_nyx_gp = GPEmulator(emulator_label="CH24_nyx_gp", archive=archive_nyx, train=True)

# %% [markdown]
# 10 min

# %%

# CH24_mpg_gp = GPEmulator(emulator_label="CH24_mpg_gp", archive=archive_c23, train=True)

# %%

CH24_mpg_gp = GPEmulator(emulator_label="CH24_mpg_gp")

# %% [markdown]
# Mirar residuals

# %%
# emulator = CH24_mpg_gp
emulator = CH24_nyx_gp
kmax_Mpc = emulator.kmax_Mpc
kmax_Mpc_use = 4

_k_Mpc = training_data[0]['k_Mpc']
ind = (_k_Mpc < kmax_Mpc_use) & (_k_Mpc > 0)
# ind = (_k_Mpc < kmax_Mpc_use)
k_Mpc = _k_Mpc[ind]
k_fit = k_Mpc/kmax_Mpc

nsam = len(training_data)
p1d_Mpc_sim = np.zeros((nsam, k_Mpc.shape[0]))
p1d_Mpc_emu = np.zeros((nsam, k_Mpc.shape[0]))
p1d_Mpc_sm = np.zeros((nsam, k_Mpc.shape[0]))

for ii in range(nsam):
# for ii in range(1):

    if "kF_Mpc" not in training_data[ii]:
        continue
    
    p1d_Mpc_sim[ii] = training_data[ii]['p1d_Mpc'][ind]
    p1d_Mpc_emu[ii] = emulator.emulate_p1d_Mpc(
        training_data[ii], 
        k_Mpc
    )

    norm = np.interp(
        k_Mpc, emulator.input_norm["k_Mpc"], emulator.norm_imF(training_data[ii]["mF"])
    )
    yfit = np.log(training_data[ii]["p1d_Mpc"][ind]/norm)

    popt, _ = curve_fit(emulator.func_poly, k_fit, yfit)
    p1d_Mpc_sm[ii] = norm * np.exp(emulator.func_poly(k_fit, *popt))

# %%
ind = p1d_Mpc_sm[:,0] != 0

# rat = p1d_Mpc_sm/p1d_Mpc_sim-1
rat = p1d_Mpc_emu[ind]/p1d_Mpc_sm[ind]-1
# rat = p1d_Mpc_emu/p1d_Mpc_sim-1
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
plt.ylim(-0.03, 0.03)
plt.xlim(0.08, 4)
plt.ylabel("emu/data-1")
# plt.savefig("ch24/average_training.png")
# plt.savefig("ch24_nyx/average_training.png")
# plt.savefig("cabayol23+/average_training.png")
# plt.savefig("nyx_alphap_cov/average_training.png")

# %%
full_emulator = [ 
    # p3d_emu,
    # full_emulator_c23plus,
    # full_emulator_p21_ext, 
    # full_emulator_p23_ext, 
    CH24_mpg_gp,
    full_emulator_ch24,
    CH24_nyx_gp,
    full_emulator_ch24_nyx,
    # full_emulator_nyx
    # full_emulator_ch24_nyx2,
    # full_emulator_ch24_nyx3
]

emu_labs = ["CH24_mpg_gp", "CH24_mpg_nn", "CH24_nyx_gp", "CH24_nyx_nn"]

# %% [markdown]
# # HERE

# %%
path_fig = '/home/jchaves/Proyectos/projects/lya/data/lace/precision/'

# %%

pars = ['Delta2_p', 'n_p', 'mF', 'gamma', 'sigT_Mpc', 'kF_Mpc']
ztar = 3.

# input = {
#     'Delta2_p': 0.3501252719027313,
#      'n_p': -2.300047197725595,
#      'mF': 0.6604100706377194,
#      'gamma': 1.512170923999183,
#      'sigT_Mpc': 0.12817463664956008,
#      'kF_Mpc': 10.6348381789184,
#      'alpha_p': -0.21536751361410592
# }
input = {
    'Delta2_p': 0.5, # 0.2-0.5
    'n_p': -2.3,
    'alpha_p': -0.215,
    'mF':0.7,
    'sigT_Mpc': 0.12,
    'gamma': 1.5,
    'kF_Mpc': 12.5,
    'kF_Mpc': 15,
}

range_par = {}
for pp in pars:
    range_par[pp] = 0.001 * np.abs(input[pp])


# %%
nn = 2
Nrealizations = 10000

# training_data = training_data_c23
# len_max = np.argwhere((training_data_c23[0]['k_Mpc'] > 0) & (training_data_c23[0]['k_Mpc'] < 4))[:,0].shape[0]
len_max = 100
kMpc = np.logspace(np.log10(0.05), np.log10(5), len_max) 
len_emus = len(full_emulator)
    
orig_dp = np.zeros((len_emus, len_max))
var_dp = np.zeros((len_emus, len(pars), nn, len_max))
all_dp_vals = []

for jj in range(len_emus):
    print(emu_labs[jj])

    # mask = (training_data[0]['k_Mpc'] > 0) & (training_data[0]['k_Mpc'] < arr_kmax[jj])
    # kMpc = training_data[0]['k_Mpc'][mask]
    
    if(emu_labs[jj] != "forestflow"):
        p1d = full_emulator[jj].emulate_p1d_Mpc(input, kMpc)
    else:
        info_power = {
            "z":ztar,
            "sim_label":"mpg_central",
            "k1d_Mpc": kMpc,
            "return_p1d": True,
        }
        out = p3d_emu.evaluate(
            emu_params=input,
            info_power=info_power,
            Nrealizations=Nrealizations
        )
        p1d = out['p1d']
    orig_dp[jj, :] = p1d

    
    for kk, par in enumerate(range_par):
        dp_vals = np.linspace(input[par]-range_par[par], input[par]+range_par[par], nn)
        all_dp_vals.append(dp_vals)
        
        for ii in range(nn):
            input2 = input.copy()
            input2[par] = dp_vals[ii]

            if(emu_labs[jj] != "forestflow"):
                p1d = full_emulator[jj].emulate_p1d_Mpc(input2, kMpc)
            else:
                out = p3d_emu.evaluate(
                    emu_params=input2,
                    info_power=info_power,
                    Nrealizations=Nrealizations
                )
                p1d = out['p1d']

            var_dp[jj, kk, ii, :] = p1d

# %%
# lw = [1, 1, 1, 2, 1, 2, 3]
lw = np.ones(len(emu_labs)) + 2
# lw[0] = 3
# lw[1] = 3
# lw[-1] = 3
# for kk, par in enumerate(range_par):
for kk, par in enumerate(range_par):
    # if kk > 1:
        # continue
    fig, ax = plt.subplots(figsize=(8,6))
    der = np.zeros((len(emu_labs), kMpc.shape[0]))
    for jj in range(len(emu_labs)):
    # for jj in range(2):
                
        # mask = (training_data[0]['k_Mpc'] > 0) & (training_data[0]['k_Mpc'] < arr_kmax[jj])
        # kMpc = training_data[0]['k_Mpc'][mask]

        diffp1d = var_dp[jj, kk, 1, :] - var_dp[jj, kk, 0, :]
        diffh = 2*range_par[par]
        der[jj] = diffp1d/diffh/orig_dp[jj]
        # der[jj] = diffp1d/orig_dp[jj]
        ax.plot(kMpc, der[jj], label=emu_labs[jj], lw=lw[jj])
            
    ax.legend()
        
    ax.axhline(ls=":", color='k')
    ax.axvline(0.09, ls=":", color='k')
    ax.axvline(4, ls=":", color='k')
    # ax.set_title(emu_labs[jj])
    # ax.set_ylabel('dP1D/d'+par)
    ax.set_ylabel(r"$dln(P1D)/d"+par+"$")
        
    ax.set_xlabel(r'$k$[1/Mpc]')
    ax.set_xscale("log")
    plt.suptitle(par)
    plt.tight_layout()
    mask = kMpc < 4
    diff = 0.05*(der[:, mask].max() - der[:, mask].min())
    ax.set_ylim(der[:, mask].min() - diff, der[:, mask].max() + diff)
    # plt.savefig(path_fig+'/sensitivity/last/'+par+'_red_new.pdf')
    # plt.savefig(path_fig+'/sensitivity/ch24/ader_'+par+'_red_new.png')
    # plt.savefig(path_fig+'/sensitivity/ch24/a4der_'+par+'_red_new.png')
    plt.savefig(path_fig+'/sensitivity/ch24/'+str(ztar)+'der_'+par+'_red_new.png')

# %%

# %%

# %%

# %%

# %%
emu_pars = ["mF", 'gamma', 'sigT_Mpc', 'kF_Mpc']
Nrealizations = 2000
kMpc = np.logspace(-2, np.log10(6), 200)
zz = np.zeros(len(testing_data))
p1d_z = np.zeros((len(testing_data), kMpc.shape[0]))

for ii in range(len(testing_data)):
    print(ii, testing_data[ii]["z"])
    zz[ii] = testing_data[ii]["z"]

    info_power = {
        "z":testing_data[ii]["z"],
        "sim_label":"mpg_central",
        "k1d_Mpc": kMpc,
        "return_p1d": True,
    }

    in_emu = {}
    for pp in emu_pars:
        in_emu[pp] = testing_data[ii][pp]
    

    out = p3d_emu.evaluate(
        emu_params=in_emu,
        info_power=info_power,
        Nrealizations=Nrealizations
    )
    p1d_z[ii] = out['p1d']



# %%
# for ii in range(len(testing_data)):
#     plt.plot(testing_data[ii]["k_Mpc"][ind], testing_data[ii]["k_Mpc"][ind]*testing_data[ii]["p1d_Mpc"][ind])
#     plt.plot(kMpc, kMpc*p1d_z[ii])
# plt.xscale("log")

# %%
dic = {}
dic["z"] = zz
dic["p1d"] = p1d_z
dic["k"] = kMpc
np.save("ff_mpgcen.npy", dic)

# %%

# %%
dp_range = np.zeros((3, orig_dp[0].shape[0]))
for ii in range(3):
    dp_range[ii] = np.max(var_dp[ii]/orig_dp[ii], axis=0) - np.min(var_dp[ii]/orig_dp[ii], axis=0)

# %% [markdown]
# # L1O
#
# We compute chi2 and bias.

# %% [markdown] jp-MarkdownHeadingCollapsed=true
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
    # for iemu in range(1):
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
                model_path='NNmodels/Cabayol23_Feb2024/Cabayol23_drop_sim_'+sim_label+'.pt', 
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
    
    for iemu in range(len(full_emulator)):
    # for iemu in range(1):

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
