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

# # Optimize star params
#
# go down to rescale to use previous results

# +
# %load_ext autoreload
# %autoreload 2

import numpy as np
import time, os, sys
import matplotlib.pyplot as plt

# our own modules
from lace.cosmo import camb_cosmo
from cup1d.likelihood import CAMB_model

# +
base = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"
folder = "DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_7/"
chain = np.load(base + folder + "chain.npy")
blobs = np.load(base + folder + "blobs.npy")
fdict = np.load(base + folder + "fitter_results.npy", allow_pickle=True).item()
labels = fdict["like"]["free_param_names"]

chain = chain[..., :2].reshape(-1, 2)

dat_Asns = np.zeros((chain.shape[0], 2))
dat_Asns_star = np.zeros((chain.shape[0], 2))
for ii in range(2):
    vmax = fdict["like"]["free_params"][labels[ii]]["max_value"]
    vmin = fdict["like"]["free_params"][labels[ii]]["min_value"]
    print(vmin, vmax)
    dat_Asns[:, ii] = chain[:, ii] * (vmax - vmin) + vmin

dat_Asns_star[:,0] = blobs["Delta2_star"].reshape(-1)
dat_Asns_star[:,1] = blobs["n_star"].reshape(-1)
# dat_Asns[:,0] = np.log(dat_Asns[:,0] * 1e10)
chain = 0
# blobs = 0

# +
d2s = blobs["Delta2_star"].reshape(-1)
ns = blobs["n_star"].reshape(-1)

shift1 = 0
shift2 = 0
pd2s = np.percentile(d2s, [16, 50, 84]) - shift1
pns = np.percentile(ns, [16, 50, 84]) - shift2
# -

pd2s[1]/(0.5 * (pd2s[2]-pd2s[1] + pd2s[1]-pd2s[0]))

-pns[1]/(0.5 * (pns[2]-pns[1] + pns[1]-pns[0]))





nn = 20000
ind = np.random.permutation(np.arange(dat_Asns.shape[0]))[:nn]


# #### Camb cosmo

def rescale_star(fid_cosmo, new_cosmo, kp_Mpc, ks_Mpc=0.05):
    """Fast computation of blob when running with fixed background"""

    # differences in primordial power (at CMB pivot point)
    ratio_As = new_cosmo["As"] / fid_cosmo["As"]
    delta_ns = new_cosmo["ns"] - fid_cosmo["ns"]
    delta_nrun = new_cosmo["nrun"] - fid_cosmo["nrun"]

    # logarithm of ratio of pivot points
    ln_kp_ks = np.log(kp_Mpc / ks_Mpc)

    # rescale blobs
    delta_alpha_star = delta_nrun
    delta_n_star = delta_ns + delta_nrun * ln_kp_ks
    ln_ratio_A_star = (
        np.log(ratio_As)
        + (delta_ns + 0.5 * delta_nrun * ln_kp_ks) * ln_kp_ks
    )

    out_star={
        "alpha_star": fid_cosmo["alpha_star"] + delta_alpha_star,
        "n_star": fid_cosmo["n_star"] + delta_n_star,
        "Delta2_star": fid_cosmo["Delta2_star"] * np.exp(ln_ratio_A_star)
    }

    return out_star


# +

cosmo_camb = camb_cosmo.get_cosmology(
    H0=67.66,
    mnu=0.0,
    omch2=0.119,
    ombh2=0.0224,
    omk=0.0,
    As=2.105e-09,
    ns=0.9665,
    nrun=0.0,
    pivot_scalar=0.05,
    w=-1,
)
# -

new_Asns_star = 0

# +
2.5 0.0053
10.8999
99.6009

3.0 0.0051
10.893
99.6072

3.5 0.0048
# -

np.linspace(0.006, 0.020, 6)

# +
# arr_z_star = np.array([2.5, 3.0, 3.5, 4.0, 8])
# arr_kp_kms = np.array([0.0005, 0.001, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.080])
# arr_z_star = np.array([2.6, 2.8, 3.0, 3.2, 3.4])
arr_z_star = np.array([3.0])
# arr_kp_kms = np.linspace(0.0045, 0.0055, 5)
arr_kp_kms = np.linspace(0.02, 0.03, 6)

arr_corr = np.zeros((arr_z_star.shape[0], arr_kp_kms.shape[0]))
arr_percen = np.zeros((arr_z_star.shape[0], arr_kp_kms.shape[0], 3, 2))


# new_Asns_star = np.zeros((dat_Asns.shape[0], dat_Asns.shape[1], arr_kp_kms.shape[0], arr_z_star.shape[0]))

for kk in range(arr_z_star.shape[0]):
    for jj in range(arr_kp_kms.shape[0]):
        print(kk, jj)
    
        fun_cosmo = CAMB_model.CAMBModel(
            zs=[arr_z_star[kk]],
            cosmo=cosmo_camb,
            z_star=arr_z_star[kk],
            kp_kms=arr_kp_kms[jj],
            fast_camb=False
        )
        results_camb = fun_cosmo.get_linP_params()
        
        # likelihood pivot point, in velocity units
        dkms_dMpc = fun_cosmo.dkms_dMpc(arr_z_star[kk])
        kp_Mpc = arr_kp_kms[jj] * dkms_dMpc
        
        fid_cosmo = {
            "As":2.105e-09,
            "ns":0.9665,
            "nrun":0,
            "Delta2_star":results_camb["Delta2_star"],
            "n_star":results_camb["n_star"],
            "alpha_star":results_camb["alpha_star"],
        }
        
        _Asns_star = np.zeros_like(dat_Asns)
        for ii in range(dat_Asns.shape[0]):
            new_cosmo = {
                "As":dat_Asns[ii,0],
                "ns":dat_Asns[ii,1],
                "nrun":0
            }
            res = rescale_star(fid_cosmo, new_cosmo, kp_Mpc, ks_Mpc=0.05)
            # new_Asns_star[ii, 0, jj, kk] = res["Delta2_star"]
            # new_Asns_star[ii, 1, jj, kk] = res["n_star"]
            _Asns_star[ii, 0] = res["Delta2_star"]
            _Asns_star[ii, 1] = res["n_star"]

        
        percen = np.percentile(_Asns_star, [5, 50, 95], axis=0)
        _ = (
            (_Asns_star[:,0] > percen[0, 0]) 
            & (_Asns_star[:,0] < percen[2, 0]) 
            & (_Asns_star[:,1] > percen[0, 1]) 
            & (_Asns_star[:,1] < percen[2, 1])
        )
        corr = np.corrcoef(_Asns_star[_, 0], _Asns_star[_, 1])[0,1]
        percen = np.percentile(_Asns_star, [16, 50, 84], axis=0)
        arr_corr[kk, jj] = corr
        arr_percen[kk, jj] = percen

# +


for kk in range(arr_z_star.shape[0]):
    for jj in range(arr_kp_kms.shape[0]):
        
        print()
        print("zstar", np.round(arr_z_star[kk],2), "kstar", np.round(arr_kp_kms[jj], 3))
        print("corr", np.round(arr_corr[kk, jj], 2))
        print(
            "Delta2star err, err/val",
            np.round(0.5 * (percen[2, 0] - percen[0, 0]), 3), 
            np.round(0.5 * (percen[2, 0] - percen[0, 0])/percen[1, 0], 4)
        )
        print(
            "nstar2star err, err/val",
            np.round(0.5 * (percen[2, 1] - percen[0, 1]), 3), 
            np.round(0.5 * (percen[2, 1] - percen[0, 1])/np.abs(percen[1, 1]), 4)
        )
# -



# +

rel_err_param = np.zeros((arr_z_star.shape[0], arr_kp_kms.shape[0], 2))

fig, ax = plt.subplots(4, figsize=(8,6), sharex=True)
fig2, ax2 = plt.subplots(1, figsize=(8,6))
for kk in range(arr_z_star.shape[0]):
    for ii in range(2):
        # percen1 = 0.5 * (arr_percen[kk, :, 2, ii] - arr_percen[kk, :, 0, ii])/np.abs(arr_percen[kk, :, 1, ii])
        percen1 = np.abs(arr_percen[kk, :, 1, ii])/(arr_percen[kk, :, 1, ii] - arr_percen[kk, :, 0, ii])
        rel_err_param[kk, :, ii] = percen1
        # percen1 = 0.5 * (arr_percen[kk, :, 2, ii] - arr_percen[kk, :, 0, ii])
        ax[ii].plot(arr_kp_kms, percen1, label=r"$z=$"+str(arr_z_star[kk]))
    ax2.plot(arr_kp_kms, arr_corr[kk])
    ii = 0
    errx = 0.5 * (arr_percen[kk, :, 2, ii] - arr_percen[kk, :, 0, ii])/np.abs(arr_percen[kk, :, 1, ii])
    ii = 1
    erry = 0.5 * (arr_percen[kk, :, 2, ii] - arr_percen[kk, :, 0, ii])/np.abs(arr_percen[kk, :, 1, ii])
    area = np.pi * errx * erry * np.sqrt(1 - arr_corr[kk]**2)
    ax[3].plot(arr_kp_kms, area)
    ax[2].plot(arr_kp_kms, area)
ax[1].legend()
ax2.axhline()
# ax2.set_xscale("log")
# ax2.set_yscale("log")

# +

fig, ax = plt.subplots(3, figsize=(10, 10))
arr_kp_kms = np.linspace(0.02, 0.03, 6)
sup_kp_kms = np.linspace(0.02, 0.03, 1000)
for kk in range(arr_z_star.shape[0]):
    print()
    yyinter = np.interp(sup_kp_kms, arr_kp_kms, arr_corr[kk])
    ind = np.argmin(np.abs(yyinter))

    ax[0].plot(arr_kp_kms, arr_corr[kk], ".-")
    
    print(np.round(arr_z_star[kk], 2), np.round(sup_kp_kms[ind], 5))
    for ii in range(3):
        ax[ii].axvline(sup_kp_kms[ind], color="C"+str(kk))

    yy0 = 0
    for ii in range(2):
        ax[ii+1].plot(arr_kp_kms, rel_err_param[kk, :, ii], ".-")
        yy = np.interp(sup_kp_kms, arr_kp_kms, rel_err_param[kk, :, ii])
        yy0 += yy[ind]
        print(np.round(yy[ind], 4))
        ax[ii+1].axhline(yy[ind], color="C"+str(kk))
    print(np.round(yy0, 4))
# -

rel_err_param[kk, :, 1]

# +

from matplotlib import rcParams
rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"
# -

fig, ax = plt.subplots(1, figsize=(8,6))
sup_kp_kms = np.logspace(-3, -1, 200)
ftsize = 20
lymin = 2.1296
lymin = 1
for kk in range(arr_z_star.shape[0]):
    # for ii in range(2):
    #     ax[ii].plot(arr_kp_kms, rel_err_param[kk, :, ii]/rel_err_param[kk, :, ii].min(), label=r"$z=$"+str(arr_z_star[kk]))
    # yy = (rel_err_param[kk, :, 0]/rel_err_param[kk, :, 0].min() + rel_err_param[kk, :, 1]/rel_err_param[kk, :, 1].min())/lymin
    yy = (rel_err_param[kk, :, 0] + rel_err_param[kk, :, 1])
    ax.plot(arr_kp_kms, yy, ".-", label=r"$z=$"+str(np.round(arr_z_star[kk], 2)), lw=2, alpha=0.95)
    yyinter = np.interp(sup_kp_kms, arr_kp_kms, yy)
    # ax.plot(sup_kp_kms, yyinter)
    yymin = yyinter.min()
    ind = np.argmin(yyinter)
    print(np.round(arr_z_star[kk], 2), np.round(yymin, 4), np.round(sup_kp_kms[ind], 4))
ax.legend(loc="upper right", fontsize=ftsize)
# ax.set_xscale("log")
ax.set_xlabel(r"$k_\star\,[\mathrm{km}^{-1}\mathrm{s}]$", fontsize=ftsize)
ax.set_ylabel(r"$(\sigma_{\Delta^2_\star}/\Delta^2_\star) + (\sigma_{n_\star}/n_\star$)", fontsize=ftsize)
ax.tick_params(
    axis="both", which="major", labelsize=ftsize
)
# ax[0].set_yscale("log")
# ax[0].set_yscale("log")
# ax.set_ylim(0.125, 0.2)

# +
# fig = corner(new_Asns_star[...,0], bins=30, range=((0.29, 0.8), (-2.45, -2.1)), color="C0");
# corner(new_Asns_star[...,1], fig=fig, bins=30, range=((0.29, 0.8), (-2.45, -2.1)), color="C1");
# corner(new_Asns_star[...,2], fig=fig, bins=30, range=((0.29, 0.8), (-2.45, -2.1)), color="C2");
# -

# # Rescale!

# +

z_star = 3.0
# kp_kms = 0.00493 # best
kp_kms = 0.019 # simple

cosmo_camb = camb_cosmo.get_cosmology(
    H0=67.66,
    mnu=0.0,
    omch2=0.119,
    ombh2=0.0224,
    omk=0.0,
    As=2.105e-09,
    ns=0.9665,
    nrun=0.0,
    pivot_scalar=0.05,
    w=-1,
)
fun_cosmo = CAMB_model.CAMBModel(
    zs=[z_star],
    cosmo=cosmo_camb,
    z_star=z_star,
    kp_kms=kp_kms,
    fast_camb=False
)
results_camb = fun_cosmo.get_linP_params()

# likelihood pivot point, in velocity units
dkms_dMpc = fun_cosmo.dkms_dMpc(z_star)
kp_Mpc = kp_kms * dkms_dMpc

fid_cosmo = {
    "As":2.105e-09,
    "ns":0.9665,
    "nrun":0,
    "Delta2_star":results_camb["Delta2_star"],
    "n_star":results_camb["n_star"],
    "alpha_star":results_camb["alpha_star"],
}

Asns_star_opt = np.zeros_like(dat_Asns)
for ii in range(dat_Asns.shape[0]):
    new_cosmo = {
        "As":dat_Asns[ii, 0],
        "ns":dat_Asns[ii, 1],
        "nrun":0
    }
    res = rescale_star(fid_cosmo, new_cosmo, kp_Mpc, ks_Mpc=0.05)
    # new_Asns_star[ii, 0, jj, kk] = res["Delta2_star"]
    # new_Asns_star[ii, 1, jj, kk] = res["n_star"]
    Asns_star_opt[ii, 0] = res["Delta2_star"]
    Asns_star_opt[ii, 1] = res["n_star"]

# percen = np.percentile(Asns_star_opt, [5, 50, 95], axis=0)
# _ = (
#     (Asns_star_opt[:,0] > percen[0, 0]) 
#     & (Asns_star_opt[:,0] < percen[2, 0]) 
#     & (Asns_star_opt[:,1] > percen[0, 1]) 
#     & (Asns_star_opt[:,1] < percen[2, 1])
# )
# corr = np.corrcoef(Asns_star_opt[_, 0], Asns_star_opt[_, 1])[0,1]
corr = np.corrcoef(Asns_star_opt[:, 0], Asns_star_opt[:, 1])[0,1]
percen = np.percentile(Asns_star_opt, [16, 50, 84], axis=0)

print("corr", np.round(corr, 4))
print(
    "Delta2star err, err/val",
    np.round(percen[1, 0], 3), np.round(percen[1, 0] - percen[0, 0], 3), np.round(percen[2, 0] - percen[1, 0], 3),
    np.round(0.5 * (percen[2, 0] - percen[0, 0]), 3), 
    np.round(0.5 * (percen[2, 0] - percen[0, 0])/percen[1, 0], 5),
    np.round(percen[1, 0]/(0.5 * (percen[2, 0] - percen[0, 0])), 5)
)
print(
    "nstar2star err, err/val",
    np.round(percen[1, 1], 3), np.round(percen[1, 1] - percen[0, 1], 3), np.round(percen[2, 1] - percen[1, 1], 3),
    np.round(0.5 * (percen[2, 1] - percen[0, 1]), 3), 
    np.round(0.5 * (percen[2, 1] - percen[0, 1])/np.abs(percen[1, 1]), 5),
    np.round(np.abs(percen[1, 1])/(0.5 * (percen[2, 1] - percen[0, 1])), 5)
)
# +
def subsample_correlations(ds, ns, A=2, nrepeat=2000, seed=0):
    """
    subsample without replacement: select n//A indices per repeat by permutation.
    If the chain has weights, warn and use weighted bootstrap instead.
    method: 'pearson' or 'spearman'
    winsorize_frac: fraction to winsorize for Pearson (e.g. 0.05), or None.
    Returns: array of correlation estimates (length nrepeat)
    """
    rng = np.random.default_rng(seed)
    n = len(ds)
    subsz = max(2, n // A)   # ensure at least 2
    corr_vals = np.empty(nrepeat)
    # Unweighted: do permutation + take first subsz (no repeats inside each subsample)
    for i in range(nrepeat):
        idx = rng.permutation(n)[:subsz]
        xs = ds[idx]
        ys = ns[idx]
        corr_vals[i] = np.corrcoef(xs, ys)[0, 1]
    return corr_vals

corr_samps = subsample_correlations(Asns_star_opt[:, 0], Asns_star_opt[:, 1], nrepeat=100, A=2)
percen = np.percentile(corr_samps, [16, 50, 84])
print(np.round(percen[1], 4), np.round(percen[1]-percen[0], 4), np.round(percen[2]-percen[1], 4))
# -

-0.0011 0.0007 0.0006

from corner import corner

fig = corner(Asns_star_opt, range=(0.999, 0.999), show_titles=True, title_fmt='.3f')

percen = np.percentile(Asns_star_opt, [16, 50, 84], axis=0)
print(percen[1])
print(percen - percen[1,:])

[ 0.6644243  -2.47431022]
[[-0.05771302 -0.01897526]
 [ 0.          0.        ]
 [ 0.05443054  0.01863571]]

0.6644243 / (0.5*(0.05771302 + 0.05443054))

2.47431022/ (0.5 * (0.01863571 + 0.01897526))

# +
# blind0 = 0.04775338959498957
# blind1 = 0.016382119648465407
# udat_Asns_star = dat_Asns_star.copy()
# udat_Asns_star[:, 0] -= blind0
# udat_Asns_star[:, 1] -= blind1
# -

percen = np.percentile(udat_Asns_star, [16, 50, 84], axis=0)
print(percen[1])
print(percen - percen[1,:])

# +

percen = np.percentile(udat_Asns_star, [5, 50, 95], axis=0)
_ = (
    (udat_Asns_star[:,0] > percen[0, 0]) 
    & (udat_Asns_star[:,0] < percen[2, 0]) 
    & (udat_Asns_star[:,1] > percen[0, 1]) 
    & (udat_Asns_star[:,1] < percen[2, 1])
)
corr = np.corrcoef(udat_Asns_star[_, 0], udat_Asns_star[_, 1])[0,1]

percen = np.percentile(udat_Asns_star, [16, 50, 84], axis=0)

print("corr", np.round(corr, 3))
print(
    "Delta2star err, err/val",
    np.round(0.5 * (percen[2, 0] - percen[0, 0]), 3), 
    np.round(0.5 * (percen[2, 0] - percen[0, 0])/percen[1, 0], 5),
    np.round(percen[1, 0]/(percen[1, 0] - percen[0, 0]), 5)
)
print(
    "nstar2star err, err/val",
    np.round(0.5 * (percen[2, 1] - percen[0, 1]), 3), 
    np.round(0.5 * (percen[2, 1] - percen[0, 1])/np.abs(percen[1, 1]), 5),
    np.round(np.abs(percen[1, 1])/(percen[1, 1] - percen[0, 1]), 5)
)

# +

percen = np.percentile(udat_Asns_star, [5, 50, 95], axis=0)
# -



fig = corner(udat_Asns_star, range=(0.999, 0.999), show_titles=True, title_fmt='.3f')

fig = corner(udat_Asns_star[_,:], show_titles=True, title_fmt='.3f')




