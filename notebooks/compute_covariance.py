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

# -

# ### Set archive

archive = gadget_archive.GadgetArchive(postproc="Cabayol23")

archive = nyx_archive.NyxArchive(nyx_version="models_Nyx_Mar2025_with_CGAN_val_3axes")

# ## Train GPs and check precision

# +
# train = True
# # train = True
# # emulator_label = "CH24_mpg_gp"
# emulator_label = "CH24_gpr"
# emulator_keep = GPEmulator(emulator_label=emulator_label, archive=archive, archive2=archive2, train=train, drop_sim=None)
# -

# Full

# train = True
train = False
# emulator_label = "CH24_mpgcen_gpr"
emulator_label = "CH24_mpg_gpr"
# emulator_label = "CH24_nyxcen_gpr"
emulator = GPEmulator(emulator_label=emulator_label, archive=archive, train=train, drop_sim=None)

emulator_load = GPEmulator(
    emulator_label="CH24_nyx_gpr", 
    train=False,
)
emulator = emulator_load
# emulator = GPEmulator(
#     emulator_label=emulator_label, 
#     train=False, 
#     drop_sim="nyx_0"
# )

emulator = emulator_keep

# Validate central!

# +
# testing_data = archive.get_testing_data("nyx_central")
# testing_data = archive.get_testing_data("mpg_central")
testing_data = archive.get_testing_data("mpg_seed")
# testing_data = archive.get_testing_data("nyx_0")
# emulator = GPEmulator(emulator_label=emulator_label, train=False, drop_sim=isim)

_k_Mpc = testing_data[0]['k_Mpc']
ind = (_k_Mpc < emulator.kmax_Mpc) & (_k_Mpc > 0)
k_Mpc = _k_Mpc[ind]
k_fit = k_Mpc/emulator.kmax_Mpc
jj = 0
if jj == 0:
    k_Mpc_0 = k_Mpc.copy()
else:
    if np.allclose(k_Mpc_0, k_Mpc) == False:
        print(jj, k_Mpc)
        

nz = len(testing_data)
# p1d_Mpc_sim = np.zeros((nsam, k_Mpc.shape[0]))
p1d_Mpc_emu = np.zeros((nz, k_Mpc.shape[0]))
p1d_Mpc_sm = np.zeros((nz, k_Mpc.shape[0]))
zz_full = np.zeros((nz, k_Mpc.shape[0]))
k_Mpc_full = np.zeros((nz, k_Mpc.shape[0]))

for ii in range(nz):
    if ("kF_Mpc" not in testing_data[ii]) | (np.isfinite(testing_data[ii]["kF_Mpc"]) == False):
        continue

    # i2 = np.argwhere(np.abs(zz - testing_data[ii]["z"]) < 0.05)[:, 0]
    # if len(i2) == 0:
        # continue
    # else:
        # i2 = i2[0]
    # print(jj, ii, testing_data[ii]["z"], i2)

    zz_full[ii] = testing_data[ii]["z"]
    k_Mpc_full[ii] = k_Mpc_0

    # p1d_Mpc_sim[i2] = testing_data[ii]['p1d_Mpc'][ind]
    p1d_Mpc_emu[ii] = emulator.emulate_p1d_Mpc(
        testing_data[ii], 
        k_Mpc
    )
    norm = np.interp(
        k_Mpc, emulator.input_norm["k_Mpc"], emulator.norm_imF(testing_data[ii]["mF"])
    )
    yfit = np.log(testing_data[ii]["p1d_Mpc"][ind]/norm)
    popt, _ = curve_fit(emulator.func_poly, k_fit, yfit)
    p1d_Mpc_sm[ii] = norm * np.exp(emulator.func_poly(k_fit, *popt))
# -

for ii in range(nz):
    plt.plot(k_Mpc_full[ii], p1d_Mpc_emu[ii]/p1d_Mpc_sm[ii])

# +
# train = False
# # train = True
# # emulator_label = "CH24_mpg_gp"
# emulator_label = "CH24_nyx_gp"
# emulator = GPEmulator(emulator_label=emulator_label, archive=archive, train=train, drop_sim=None)
# -

# L10

# +
train = True

# emulator_label = "CH24_mpg_gp"
# emulator_label = "CH24_nyx_gpr"
# emulator_label = "CH24_nyxcen_gpr"
# emulator_label = "CH24_mpgcen_gpr"
emulator_label = "CH24_mpg_gpr"
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

# #### mpg

rel_diff = p1d_Mpc_emu/p1d_Mpc_sm - 1
cov = np.cov(rel_diff.reshape(-1, rel_diff.shape[-1]).T)
plt.imshow(cov)

# #### nyx

rel_diff = p1d_Mpc_emu/p1d_Mpc_sm - 1
rel_diff = rel_diff[mask, :]
mask2 = np.any(np.isfinite(rel_diff), axis=1)
rel_diff = rel_diff[mask2, :]
cov = np.cov(rel_diff.reshape(-1, rel_diff.shape[-1]).T)
plt.imshow(cov)

# #### mpg

corr = np.zeros_like(cov)
for ii in range(cov.shape[0]):
    for jj in range(cov.shape[0]):
        corr[ii, jj] = cov[ii, jj]/np.sqrt(cov[ii, ii] * cov[jj, jj])
plt.imshow(corr)

# #### nyx

corr = np.zeros_like(cov)
for ii in range(cov.shape[0]):
    for jj in range(cov.shape[0]):
        corr[ii, jj] = cov[ii, jj]/np.sqrt(cov[ii, ii] * cov[jj, jj])
plt.imshow(corr)

# #### mpg

bias = np.mean(rel_diff, axis=(0, 1))
plt.plot(k_Mpc, np.sqrt(np.diag(cov)))
plt.plot(k_Mpc, bias)

# #### nyx

bias = np.mean(rel_diff, axis=0)
plt.plot(k_Mpc, np.sqrt(np.diag(cov)))
plt.plot(k_Mpc, bias)

filename = "l1O_cov_" + emulator_label + ".npy"
full_path = os.path.join(os.path.dirname(lace.__path__[0]), "data", "covariance", filename)
dict_save = {}
dict_save["cov"] = cov
dict_save["zz"] = zz
dict_save["k_Mpc"] = k_Mpc
np.save(full_path, dict_save)




