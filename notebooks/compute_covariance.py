# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Compute covariance matrix

# First show how to train and load emulators, then compute covariance matrix

# +
# %load_ext autoreload
# %autoreload 2

import numpy as np
import time, os, sys
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import lace
from lace.archive import gadget_archive, nyx_archive
from lace.emulator.gp_emulator_multi import GPEmulator
from matplotlib.ticker import FormatStrFormatter
# -

# ### Set archive

archive = gadget_archive.GadgetArchive(postproc="Cabayol23")

archive = nyx_archive.NyxArchive(nyx_version="models_Nyx_Sept2025_include_Nyx_fid_rseed")

# +
# train = True
# # train = True
# # emulator_label = "CH24_mpg_gp"
# emulator_label = "CH24_gpr"
# emulator_keep = GPEmulator(emulator_label=emulator_label, archive=archive, archive2=archive2, train=train, drop_sim=None)
# -

# # Full

train = True
# train = False
emulator_label = "CH24_mpgcen_gpr"
# emulator_label = "CH24_mpg_gpr"
# emulator_label = "CH24_nyxcen_gpr"
emulator = GPEmulator(emulator_label=emulator_label, archive=archive, train=train, drop_sim=None)

# +

emulator_label = "CH24_mpgcen_gpr"
# emulator_label = "CH24_nyxcen_gpr"

emulator = GPEmulator(
    emulator_label=emulator_label, 
    train=False,
)
# -

# ## L10

# +
train = True

# emulator_label = "CH24_mpgcen_gpr"
emulator_label = "CH24_nyxcen_gpr"
if train:
    for ii, isim in enumerate(archive.list_sim_cube):
        if (ii >= 14) & ("nyx" in emulator_label):
            continue
        print(ii, isim)
        emulator = GPEmulator(
            emulator_label=emulator_label, 
            archive=archive, 
            train=True, 
            drop_sim=isim
        )
# -

# ## Compute covariance matrix

from lace.emulator.covariance import data_for_l10

# suite = "mpg"
suite = "nyx"
emulator_label = "CH24_"+suite+"cen_gpr"
zz, k_Mpc, p1d_Mpc_sm, p1d_Mpc_emu, mask = data_for_l10(archive, emulator_label, suite=suite)

# +
arr_zz = np.zeros_like(p1d_Mpc_emu)
arr_k_Mpc = np.zeros_like(p1d_Mpc_emu)

for ii in range(arr_zz.shape[1]):
    arr_zz[:, ii, :] = zz[ii]
    
for ii in range(arr_k_Mpc.shape[2]):
    arr_k_Mpc[:, :, ii] = k_Mpc[ii]

# +
rel_diff = p1d_Mpc_emu/p1d_Mpc_sm - 1
# cov z-k
rel_diff_zk = rel_diff.reshape(rel_diff.shape[0], -1)
zz_zk = arr_zz.reshape(arr_zz.shape[0], -1)[0]
k_Mpc_zk = arr_k_Mpc.reshape(arr_k_Mpc.shape[0], -1)[0]
# cov k
rel_diff_k = rel_diff.reshape(-1, rel_diff.shape[-1])
k_Mpc_k = arr_k_Mpc.reshape(-1, arr_k_Mpc.shape[-1])[0]

cov_zk = np.cov(rel_diff_zk.T)
cov_k = np.cov(rel_diff_k.T)
print(cov_zk.shape, cov_k.shape)
# -

# #### massage ONLY nyx data

# +
_ = (np.isnan(rel_diff_zk) | (rel_diff_zk == 0))
rel_diff_zk[_] = 0
cov_zk = np.cov(rel_diff_zk.T)

_ = (np.isnan(rel_diff_k) | (rel_diff_k == 0))
rel_diff_k[_] = 0
cov_k = np.cov(rel_diff_k.T)
# -



# #### mpg

plt.imshow(cov_zk)

# #### nyx

plt.imshow(cov_zk)

# #### mpg

cov = cov_zk
corr = np.zeros_like(cov)
for ii in range(cov.shape[0]):
    for jj in range(cov.shape[0]):
        corr[ii, jj] = cov[ii, jj]/np.sqrt(cov[ii, ii] * cov[jj, jj])
plt.imshow(corr)
plt.colorbar()

# #### nyx

cov = cov_zk
corr = np.zeros_like(cov)
for ii in range(cov.shape[0]):
    for jj in range(cov.shape[0]):
        corr[ii, jj] = cov[ii, jj]/np.sqrt(cov[ii, ii] * cov[jj, jj])
plt.imshow(corr)
plt.colorbar()

# #### mpg

# +

for ii in range(len(zz)):
    _ = zz_zk == zz[ii]
    plt.plot(k_Mpc_zk[_], np.sqrt(np.diag(cov_zk))[_], label=str(zz[ii]))

plt.legend()
plt.xlabel(r"$k$[1/Mpc]")
plt.ylabel(r"Relative error")
# bias = np.mean(rel_diff, axis=(0, 1))
# plt.plot(k_Mpc, bias)
plt.xscale("log")
plt.tight_layout()
# plt.savefig("figs/err_CH24_mpgcen_gpr.png")
# plt.savefig("figs/err_CH24_mpgcen_gpr.pdf")
# -


# ### nyx

# +

for ii in range(len(zz)):
    _ = zz_zk == zz[ii]
    plt.plot(k_Mpc_zk[_], np.sqrt(np.diag(cov_zk))[_], label=str(zz[ii]))

plt.legend()
plt.xlabel(r"$k$[1/Mpc]")
plt.ylabel(r"Relative error")
# bias = np.mean(rel_diff, axis=(0, 1))
# plt.plot(k_Mpc, bias)
plt.xscale("log")
plt.tight_layout()
# plt.savefig("figs/err_CH24_mpgcen_gpr.png")
# plt.savefig("figs/err_CH24_mpgcen_gpr.pdf")

# +
filename = "l1O_cov_" + emulator_label + ".npy"
full_path = os.path.join(os.path.dirname(lace.__path__[0]), "data", "covariance", filename)
dict_save = {}
dict_save["zz"] = zz
dict_save["k_Mpc"] = k_Mpc

dict_save["k_Mpc_k"] = k_Mpc_k
dict_save["cov_k"] = cov_k

dict_save["zz_zk"] = zz_zk
dict_save["k_Mpc_zk"] = k_Mpc_zk
dict_save["cov_zk"] = cov_zk
np.save(full_path, dict_save)
# -

# #### Load data

suite = "mpg"
# suite = "nyx"
emulator_label = "CH24_"+suite+"cen_gpr"
filename = "l1O_cov_" + emulator_label + ".npy"
full_path = os.path.join(os.path.dirname(lace.__path__[0]), "data", "covariance", filename)
dat = np.load(full_path, allow_pickle=True).item()

cov = dat["cov"]
k_Mpc = dat["k_Mpc"]


