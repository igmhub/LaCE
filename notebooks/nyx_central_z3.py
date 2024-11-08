# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: lace
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from loguru import logger

from lace.archive import nyx_archive
from lace.emulator.nn_emulator import NNEmulator
from lace.utils import poly_p1d
from lace.emulator.constants import PROJ_ROOT
os.environ["NYX_PATH"]="/Users/lauracabayol/Documents/DESI/nyx"

archive = nyx_archive.NyxArchive(verbose=True)

emu_params=['Delta2_p', 'n_p','alpha_p', 'mF', 'sigT_Mpc', 'gamma', 'kF_Mpc']
training_data=archive.get_training_data(emu_params=emu_params)
testing_data_central = archive.get_testing_data('nyx_central')

# ## Extract parameters

take_pars = ['Delta2_p', 'n_p','alpha_p','mF', 'sigT_Mpc', 'gamma', 'kF_Mpc']
nelem = len(training_data)
pars_all_nyx = np.array([[training_data[ii][param] for param in take_pars] for ii in range(nelem)])

nelem = len(testing_data_central)
pars_all_nyx_central = np.array([
    [testing_data_central[ii][param] for param in take_pars]
    for ii in range(len(testing_data_central))
    if all(param in testing_data_central[ii] for param in take_pars)
])

testing_data_central_z3 = [d for d in testing_data_central if d['z']==3]
nelem = len(testing_data_central_z3)
pars_all_nyx_central_z3 = np.array([
    [testing_data_central_z3[ii][param] for param in take_pars]
    for ii in range(len(testing_data_central_z3))
    if all(param in testing_data_central_z3[ii] for param in take_pars)
])

# ## PARAMETER SPACE

# +
data_nyx = pars_all_nyx.copy()
num_dimensions = data_nyx.shape[1]

num_rows = 3  
num_cols = (num_dimensions + 1) // num_rows

# Create subplots
fig, axs = plt.subplots(num_rows, num_cols, figsize=(8, 8))

if num_rows > 1:
    axs = axs.flatten()

# Create scatter plots for all pairs of dimensions
for i in range(num_dimensions-1):
    axs[i].scatter(data_nyx[:, i], data_nyx[:, i+1],  s=1,color='goldenrod', alpha=0.3)
    axs[i].scatter(pars_all_nyx_central[:, i], pars_all_nyx_central[:, i+1],  s=10, color='crimson')
    axs[i].scatter(pars_all_nyx_central_z3[:, i], pars_all_nyx_central_z3[:, i+1],  s=10, color='navy')


    axs[i].set_xlabel(take_pars[i])
    axs[i].set_ylabel(take_pars[i+1])

plt.tight_layout()
# -

# ## REDSHIFT EVOLUTION

igm_nyx = np.load(os.environ["NYX_PATH"] + "/IGM_histories.npy", allow_pickle=True).item()

# +
fig, ax = plt.subplots(2, 2, sharex=True)
ax = ax.reshape(-1)
for sim in igm_nyx.keys():
    if(sim == "nyx_14"):
        continue
    if((sim == 'nyx_central') | (sim == 'mpg_reio')):
        col = 'r'
        alpha = 1
    else:
        col = 'k'
        alpha = 0.2
        
    par = ['tau_eff', 'gamma', 'sigT_kms', 'kF_kms']
    for jj in range(len(par)):
        _ = igm_nyx[sim][par[jj]] != 0
        ax[jj].plot(igm_nyx[sim]['z'][_], igm_nyx[sim][par[jj]][_], col, alpha=alpha)

xlabs = [None, None, r'$z$', r'$z$']
ylabs = [r'$\tau_\mathrm{eff}$', r'$\gamma$', r'$\sigma_T$', r'$k_F$']
for ii in range(4):
    ax[ii].set_xlabel(xlabs[ii])
    ax[ii].set_ylabel(ylabs[ii])
plt.tight_layout()
# -

# ## PLOT P1D

nelem = len(training_data)
k_Mpc_LH = training_data[0]["k_Mpc"]
mask_kMpc = (k_Mpc_LH>0)&(k_Mpc_LH<4)
p1d_all_nyx = np.array([training_data[ii]["p1d_Mpc"][mask_kMpc] for ii in range(nelem)])
k_Mpc_LH = k_Mpc_LH[mask_kMpc]

nelem = len(testing_data_central)
k_Mpc = testing_data_central[0]["k_Mpc"]
mask_kMpc = (k_Mpc>0)&(k_Mpc<4)
p1d_all_nyx_central = np.array([testing_data_central[ii]["p1d_Mpc"][mask_kMpc] for ii in range(nelem)])

testing_data_central_z3 = [d for d in testing_data_central if d['z']==3]
nelem = len(testing_data_central_z3)
k_Mpc_central = testing_data_central_z3[0]["k_Mpc"]
mask_kMpc = (k_Mpc_central>0)&(k_Mpc_central<4)
p1d_all_nyx_central_z3 = np.array([testing_data_central_z3[ii]["p1d_Mpc"][mask_kMpc] for ii in range(nelem)])
k_Mpc_central = k_Mpc_central[mask_kMpc]

# +

fig, ax = plt.subplots(figsize=(6, 4))

ax.loglog(k_Mpc_LH, p1d_all_nyx.T, color='goldenrod', alpha=0.3)
ax.loglog(k_Mpc_central, p1d_all_nyx_central.T, color='crimson', alpha=0.3)
ax.loglog(k_Mpc_central, p1d_all_nyx_central_z3.T, color='navy', alpha=0.3)


ax.set_xlabel(r"$k$ [1/Mpc]")
ax.set_ylabel("P1D [Mpc]")

plt.tight_layout()
plt.show
# -

# ## FIND NEAREST NEIGHBOURS TO CENTRAL AT z=3

# +
from sklearn.neighbors import NearestNeighbors

# Normalize parameters (keeping your existing normalization)
pars_min = pars_all_nyx.min(axis=0)
pars_max = pars_all_nyx.max(axis=0)
pars_range = pars_max - pars_min

pars_nyx_norm = (pars_all_nyx - pars_min) / pars_range
central_z3_norm = (pars_all_nyx_central_z3 - pars_min) / pars_range

central_z3_norm[:,3] = central_z3_norm[:,3] * 2
pars_nyx_norm[:,3] = pars_nyx_norm[:,3] * 2

# Initialize and fit the NearestNeighbors model
n_neighbors = 50
nn = NearestNeighbors(n_neighbors=n_neighbors)
nn.fit(pars_nyx_norm)

# Find nearest neighbors for all central points
distances, indices = nn.kneighbors(central_z3_norm)

logger.info(f"Indices of all unique nearest neighbors: {unique_nn_indices}")

for i, (central_params, idx, dist) in enumerate(zip(pars_all_nyx_central_z3, indices, distances)):
    logger.info(f"\nCentral point {i}:")
    logger.info(f"Parameters: {dict(zip(take_pars, central_params))}")
    logger.info("\nNearest neighbors:")
    for neighbor_idx, neighbor_dist in zip(idx, dist):
        logger.info(f"Simulation: {training_data[neighbor_idx]['sim_label']} at z={np.round(training_data[neighbor_idx]['z'],2)}, Distance: {neighbor_dist:.3f}, Parameters: {dict(zip(take_pars, pars_all_nyx[neighbor_idx]))}")


# +
data_nyx = pars_all_nyx.copy()
num_dimensions = data_nyx.shape[1]

num_rows = 3  
num_cols = (num_dimensions + 1) // num_rows

# Create subplots
fig, axs = plt.subplots(num_rows, num_cols, figsize=(8, 8))

if num_rows > 1:
    axs = axs.flatten()

# Create scatter plots for all pairs of dimensions
for i in range(num_dimensions-1):
    axs[i].scatter(data_nyx[unique_nn_indices, i], data_nyx[unique_nn_indices, i+1],  s=3,color='goldenrod', alpha=0.8)
    axs[i].scatter(pars_all_nyx_central_z3[:, i], pars_all_nyx_central_z3[:, i+1],  s=3, color='navy')


    axs[i].set_xlabel(take_pars[i])
    axs[i].set_ylabel(take_pars[i+1])

plt.tight_layout()

# +

fig, ax = plt.subplots(figsize=(6, 4))

ax.loglog(k_Mpc_LH, p1d_all_nyx[unique_nn_indices].T, color='goldenrod', alpha=0.3)
ax.loglog(k_Mpc_central, p1d_all_nyx_central_z3.T, color='navy', alpha=0.3)


ax.set_xlabel(r"$k$ [1/Mpc]")
ax.set_ylabel("P1D [Mpc]")

plt.tight_layout()
plt.show
# -


