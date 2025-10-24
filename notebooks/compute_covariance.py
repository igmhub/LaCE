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

# # Compute covariance matrix, also figure smoothing

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

archive = nyx_archive.NyxArchive(nyx_version="models_Nyx_Sept2025_include_Nyx_fid_rseed")

# ## Train GPs and check precision

# +
# train = True
# # train = True
# # emulator_label = "CH24_mpg_gp"
# emulator_label = "CH24_gpr"
# emulator_keep = GPEmulator(emulator_label=emulator_label, archive=archive, archive2=archive2, train=train, drop_sim=None)
# -

# Full

train = True
# train = False
# emulator_label = "CH24_mpgcen_gpr"
# emulator_label = "CH24_mpg_gpr"
emulator_label = "CH24_nyxcen_gpr"
emulator = GPEmulator(emulator_label=emulator_label, archive=archive, train=train, drop_sim=None)

emulator_load = GPEmulator(
    emulator_label="CH24_nyxcen_gpr", 
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

# testing_data = archive.get_testing_data("mpg_central")
# testing_data = archive.get_testing_data("mpg_seed")

testing_data = archive.get_testing_data("nyx_central")
# testing_data = archive.get_testing_data("nyx_seed")

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

p1d_Mpc_sim = np.zeros((nz, k_Mpc.shape[0]))
p1d_Mpc_emu = np.zeros((nz, k_Mpc.shape[0]))
p1d_Mpc_sm = np.zeros((nz, k_Mpc.shape[0]))
zz_full = np.zeros((nz, k_Mpc.shape[0]))
k_Mpc_full = np.zeros((nz, k_Mpc.shape[0]))

for ii in range(nz):
    # if ("kF_Mpc" not in testing_data[ii]) | (np.isfinite(testing_data[ii]["kF_Mpc"]) == False):
        
    #     continue

    # i2 = np.argwhere(np.abs(zz - testing_data[ii]["z"]) < 0.05)[:, 0]
    # if len(i2) == 0:
        # continue
    # else:
        # i2 = i2[0]
    # print(jj, ii, testing_data[ii]["z"], i2)

    zz_full[ii] = testing_data[ii]["z"]
    k_Mpc_full[ii] = k_Mpc_0

    p1d_Mpc_sim[ii] = testing_data[ii]['p1d_Mpc'][ind]
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

zz_full_seed = zz_full.copy()
p1d_Mpc_sim_seed = p1d_Mpc_sim.copy()
p1d_Mpc_sm_seed = p1d_Mpc_sm.copy()

# +

from matplotlib import rcParams

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"


# +
fig, ax = plt.subplots(1, figsize=(8, 6))
ftsize = 24

ztar = 3
ii = np.argwhere(zz_full == ztar)[0,0]
jj = np.argwhere(zz_full_seed == ztar)[0,0] 
    
psm = 0.5 * (p1d_Mpc_sm_seed[jj] + p1d_Mpc_sm[ii])
# ax.plot(k_Mpc_full[ii], p1d_Mpc_sm[ii]/p1d_Mpc_sim[ii]-1, "C0-", lw=2, label="mpg-central")
# ax.plot(k_Mpc_full[ii], p1d_Mpc_sim_seed[ii]/p1d_Mpc_sm_seed[ii]-1, "C1--", lw=2, label="mpg-seed")
ax.plot(k_Mpc_full[ii], p1d_Mpc_sim[ii]/psm-1, "C0-", lw=2, label="central")
ax.plot(k_Mpc_full[ii], p1d_Mpc_sim_seed[jj]/psm-1, "C1-", lw=2, label="seed")
ax.plot(k_Mpc_full[ii], p1d_Mpc_sm[ii]/psm-1, "C0--", lw=2, label="smooth central")
ax.plot(k_Mpc_full[ii], p1d_Mpc_sm_seed[jj]/psm-1, "C1--", lw=2, label="smooth seed")


ax.axhline(color="k", linestyle=":")
ax.axhline(0.01, color="k", linestyle="--")
ax.axhline(-0.01, color="k", linestyle="--")
ax.set_ylim(-0.022, 0.028)

# ax.set_ylim(-0.06, 0.1)

ax.set_xscale("log")
ax.set_ylabel(r"$P_\mathrm{1D}^\mathrm{x}/P_\mathrm{1D}^\mathrm{smooth}-1$", fontsize=ftsize)
ax.set_xlabel(r"$k\,\left[\mathrm{Mpc}^{-1}\right]$", fontsize=ftsize)
ax.tick_params(axis="both", which="major", labelsize=ftsize)
plt.legend(fontsize=ftsize-4, loc="upper right", ncol=2)
plt.tight_layout()
# plt.savefig("figs/smooth_cen_seed.png")
# plt.savefig("figs/smooth_cen_seed.pdf")
# plt.savefig("figs/smooth_cen_seed_nyx.png")
# plt.savefig("figs/smooth_cen_seed_nyx.pdf")
# -
zz_full.shape

p1d_Mpc_emu1 = p1d_Mpc_emu.copy()

for ii in range(nz):
    plt.plot(k_Mpc_full[ii], p1d_Mpc_emu1[ii]/p1d_Mpc_emu[ii], label=zz_full[ii,0 ])
plt.legend()

for ii in range(nz):
    plt.plot(k_Mpc_full[ii], p1d_Mpc_emu[ii]/p1d_Mpc_sm[ii], label=zz_full[ii,0 ])
plt.legend()

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
emulator_label = "CH24_nyxcen_gpr"
# emulator_label = "CH24_mpgcen_gpr"
# emulator_label = "CH24_mpg_gpr"
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

suite = "mpg"
# suite = "nyx"
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
plt.colorbar()

corr = np.zeros_like(cov)
for ii in range(cov.shape[0]):
    for jj in range(cov.shape[0]):
        corr[ii, jj] = cov[ii, jj]/np.sqrt(cov[ii, ii] * cov[jj, jj])
plt.imshow(corr)
plt.colorbar()

# #### nyx

corr = np.zeros_like(cov)
for ii in range(cov.shape[0]):
    for jj in range(cov.shape[0]):
        corr[ii, jj] = cov[ii, jj]/np.sqrt(cov[ii, ii] * cov[jj, jj])
plt.imshow(corr)
plt.colorbar()



# #### mpg

# +

plt.plot(k_Mpc, np.sqrt(np.diag(cov)))
plt.xlabel(r"$k$[1/Mpc]")
plt.ylabel(r"Relative error")
# bias = np.mean(rel_diff, axis=(0, 1))
# plt.plot(k_Mpc, bias)
plt.xscale("log")
plt.tight_layout()
plt.savefig("figs/err_CH24_mpgcen_gpr.png")
plt.savefig("figs/err_CH24_mpgcen_gpr.pdf")


# +

plt.plot(k_Mpc, np.sqrt(np.diag(cov)))
plt.xlabel(r"$k$[1/Mpc]")
plt.ylabel(r"Relative error")
# bias = np.mean(rel_diff, axis=(0, 1))
# plt.plot(k_Mpc, bias)
plt.xscale("log")
plt.tight_layout()
# -

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

# #### Load data

suite = "mpg"
# suite = "nyx"
emulator_label = "CH24_"+suite+"cen_gpr"
filename = "l1O_cov_" + emulator_label + ".npy"
full_path = os.path.join(os.path.dirname(lace.__path__[0]), "data", "covariance", filename)
dat = np.load(full_path, allow_pickle=True).item()

cov = dat["cov"]
k_Mpc = dat["k_Mpc"]


