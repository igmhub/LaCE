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

# # Investigate compressed parameters

# +
import matplotlib.pyplot as plt
import numpy as np
from cup1d.likelihood.cosmologies import set_cosmo
from cup1d.likelihood import CAMB_model
from lace.cosmo import camb_cosmo

from matplotlib import rcParams

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

# +
from scipy.integrate import simpson

def _P1D_lnkperp(ln_k_perp, kpars, k_pk, pk):
    """
    Compute P1D by integrating P3D in terms of ln(k_perp).

    Parameters:
        z (float): Redshift.
        ln_k_perp (array-like): Array of natural logarithms of the perpendicular wavenumber.
        kpars (array-like): Array of parallel wavenumbers.
        parameters (dict, optional): Additional parameters for the model. Defaults to {}.

    Returns:
        array-like: Computed values of P1D.
    """

    # get interval for integration
    dlnk = ln_k_perp[1] - ln_k_perp[0]

    # for each value of k_par, integrate P3D over ln(k_perp) to get P1D
    p1d = np.empty_like(kpars)
    for i in range(kpars.size):
        # get function to be integrated
        
        # compute k and mu from ln_k_perp and k_par
        k_perp = np.exp(ln_k_perp)
        k_par = kpars[i]
        k = np.sqrt(k_par**2 + k_perp**2)

        p3d = np.exp(np.interp(np.log(k), np.log(k_pk), np.log(pk)))
        p3d_fix_k_par = (1 / (2 * np.pi)) * k_perp**2 * p3d
        # perform numerical integration
        p1d[i] = simpson(p3d_fix_k_par, ln_k_perp, dx=dlnk)

    return p1d


def P1D_Mpc(
    k_par,
    k_pk, 
    pk,
    k_perp_min=0.001,
    # k_perp_max=300,
    # n_k_perp=999,
    k_perp_max=100,
    n_k_perp=99,
    kpressure=30,
):
    """
    Returns P1D for specified values of k_par, with the option to specify values of k_perp to be integrated over.

    Parameters:
        z (float): Redshift.
        k_par (array-like): Array or list of values for which P1D is to be computed.
        k_perp_min (float, optional): Lower bound of integral. Defaults to 0.001.
        k_perp_max (float, optional): Upper bound of integral. Defaults to 100.
        n_k_perp (int, optional): Number of points in integral. Defaults to 99.
        parameters (dict, optional): Additional parameters for the model. Defaults to {}.

    Returns:
        array-like: Computed values of P1D.
    """

    ln_k_perp = np.linspace(
        np.log(k_perp_min), np.log(k_perp_max), n_k_perp
    )

    # p1d = _P1D_lnkperp(ln_k_perp, k_par, k_pk, pk) * np.exp(-k_par**2 / kpressure**2)
    p1d = _P1D_lnkperp(ln_k_perp, k_par, k_pk, pk)

    return p1d
    
def P1D_kms(
    k_par_kms,
    k_pk_Mpc, 
    pk_Mpc,
    dkms_dMpc,
    k_perp_min=0.001,
    k_perp_max=400,
    n_k_perp=99,
    kpressure=30,
):
    """
    Returns P1D for specified values of k_par, with the option to specify values of k_perp to be integrated over.

    Parameters:
        z (float): Redshift.
        k_par (array-like): Array or list of values for which P1D is to be computed.
        k_perp_min (float, optional): Lower bound of integral. Defaults to 0.001.
        k_perp_max (float, optional): Upper bound of integral. Defaults to 100.
        n_k_perp (int, optional): Number of points in integral. Defaults to 99.
        parameters (dict, optional): Additional parameters for the model. Defaults to {}.

    Returns:
        array-like: Computed values of P1D.
    """

    ln_k_perp = np.linspace(
        np.log(k_perp_min/dkms_dMpc), np.log(k_perp_max/dkms_dMpc), n_k_perp
    )
    damp = np.exp(-pk_Mpc**2 / kpressure**2)
    p1d = _P1D_lnkperp(ln_k_perp, k_par_kms, k_pk_Mpc/dkms_dMpc, pk_Mpc*damp*dkms_dMpc**3)

    return p1d
# -



# +
kmax_Mpc = 400

mm = 40
k_par_Mpc = np.geomspace(0.01, 5, mm)
# k_par_kms = np.geomspace(0.001, 0.05, mm)
k_par_kms = np.linspace(0.01, 0.05, mm)

zstar = 3
kp_kms = 0.009

cosmo = set_cosmo("Planck18")
camb_object = CAMB_model.CAMBModel([zstar], cosmo=cosmo, z_star=zstar, kp_kms=kp_kms, fast_camb=False)
camb_results = camb_cosmo.get_camb_results(cosmo, zs=[zstar], fast_camb=False, camb_kmax_Mpc=kmax_Mpc)
target_linP = camb_object.get_linP_params()

fid_Astar = target_linP["Delta2_star"]
fid_nstar = target_linP["n_star"]
fid_alphastar = target_linP["alpha_star"]

dkms_dMpc = camb_object.dkms_dMpc(zstar)
kp_Mpc = kp_kms * dkms_dMpc

# kp_Mpc = 0.7
ks_Mpc = camb_object.cosmo.InitPower.pivot_scalar
ln_kp_ks = np.log(kp_Mpc / ks_Mpc)

k_Mpc, zs_out, P_Mpc = camb_cosmo.get_linP_Mpc(cosmo, [zstar], camb_results=camb_results, kmax_Mpc=kmax_Mpc)
fid_P_Mpc = P_Mpc[0].copy()
fid_dkms_dMpc = dkms_dMpc
fid_p1d_Mpc = P1D_Mpc(k_par_Mpc, k_Mpc, P_Mpc[0])

fid_p1d_kms = P1D_kms(k_par_kms, k_Mpc, P_Mpc[0], fid_dkms_dMpc)


# +
# fid_p1d_kms2 = fid_p1d_kms.copy()

# +
# plt.plot(k_par_kms, k_par_kms*fid_p1d_kms, ".-")
# plt.plot(k_par_kms, k_par_kms*fid_p1d_kms2, ".-")

# +


nn = 6
omh2 = np.linspace(0.1071, 0.1309, nn)

list_Pk_dsa = []
list_P1D_dsa = []
list_dkms_dsa = []

list_Pk_ds = []
list_P1D_ds = []
list_dkms_ds = []
for ii in range(nn):
    print(omh2[ii])
    cosmo = camb_cosmo.get_cosmology(
            H0=67.66,
            mnu=0.0,
            omch2=omh2[ii],
            ombh2=0.0224,
            omk=0.0,
            As=2.105e-09,
            ns=0.9665,
            nrun=0.0,
            pivot_scalar=0.05,
            w=-1,
        )    
    camb_object = CAMB_model.CAMBModel([zstar], cosmo=cosmo, z_star=zstar, kp_kms=kp_kms, fast_camb=False)
    inter_params = camb_object.get_linP_params()
    
    dkms_dMpc = camb_object.dkms_dMpc(zstar)
    kp_Mpc = kp_kms * dkms_dMpc
    ln_kp_ks0 = np.log(kp_Mpc / ks_Mpc)

    test_Astar = inter_params["Delta2_star"]
    test_nstar = inter_params["n_star"]
    test_alphastar = inter_params["alpha_star"]    
    
    ln_ratio_Astar = np.log(test_Astar / fid_Astar)
    delta_nstar = test_nstar - fid_nstar
    delta_alphastar = test_alphastar - fid_alphastar

    # scale first    
    delta_nrun = delta_alphastar
    delta_ns = delta_nstar - delta_nrun * ln_kp_ks0
    ln_ratio_As = (
        ln_ratio_Astar
        - (delta_ns + 0.5 * delta_nrun * ln_kp_ks0) * ln_kp_ks0
    )

    cosmo = camb_cosmo.get_cosmology(
            H0=67.66,
            mnu=0.0,
            omch2=omh2[ii],
            ombh2=0.0224,
            omk=0.0,
            As=2.105e-09 / np.exp(ln_ratio_As),
            ns=0.9665 - delta_ns,
            nrun=0.0 - delta_nrun,
            pivot_scalar=0.05,
            w=-1,
        )
    camb_results = camb_cosmo.get_camb_results(cosmo, zs=[zstar], fast_camb=False, camb_kmax_Mpc=kmax_Mpc)
    camb_object = CAMB_model.CAMBModel([zstar], cosmo=cosmo, z_star=zstar, kp_kms=kp_kms, fast_camb=False)
    dkms_dMpc = camb_object.dkms_dMpc(zstar)
    list_dkms_dsa.append(dkms_dMpc)
    inter_params = camb_object.get_linP_params()
    print(inter_params)
    
    k_Mpc, zs_out, P_Mpc = camb_cosmo.get_linP_Mpc(cosmo, [zstar], kmax_Mpc=kmax_Mpc, camb_results=camb_results)
    list_Pk_dsa.append(P_Mpc[0])
    list_P1D_dsa.append(P1D_kms(k_par_kms, k_Mpc, P_Mpc[0], dkms_dMpc))

    # scale second    
    delta_nrun = 0
    delta_ns = delta_nstar - delta_nrun * ln_kp_ks0
    ln_ratio_As = (
        ln_ratio_Astar
        - (delta_ns + 0.5 * delta_nrun * ln_kp_ks0) * ln_kp_ks0
    )
    
    cosmo = camb_cosmo.get_cosmology(
            H0=67.66,
            mnu=0.0,
            omch2=omh2[ii],
            ombh2=0.0224,
            omk=0.0,
            As=2.105e-09 / np.exp(ln_ratio_As),
            ns=0.9665 - delta_ns,
            nrun=0.0,
            pivot_scalar=0.05,
            w=-1,
        )
    camb_results = camb_cosmo.get_camb_results(cosmo, zs=[zstar], fast_camb=False, camb_kmax_Mpc=kmax_Mpc)
    k_Mpc, zs_out, P_Mpc = camb_cosmo.get_linP_Mpc(cosmo, [zstar], kmax_Mpc=kmax_Mpc, camb_results=camb_results)
    camb_object = CAMB_model.CAMBModel([zstar], cosmo=cosmo, z_star=zstar, kp_kms=kp_kms, fast_camb=False)
    dkms_dMpc = camb_object.dkms_dMpc(zstar)
    list_dkms_ds.append(dkms_dMpc)
    
    inter_params = camb_object.get_linP_params()
    print(inter_params)
    
    list_Pk_ds.append(P_Mpc[0])
    # list_P1D_ds.append(P1D_Mpc(k_par, k_Mpc, P_Mpc[0]))
    list_P1D_ds.append(P1D_kms(k_par_kms, k_Mpc, P_Mpc[0], dkms_dMpc))

# -

fig, ax = plt.subplots(figsize=(8,6))
for ii in range(nn):
    col = "C"+str(ii)
    ax.plot(k_Mpc, list_Pk_dsa[ii]/fid_P_Mpc, col+"-")
    ax.plot(k_Mpc, list_Pk_ds[ii]/fid_P_Mpc, col+"--")
plt.xscale("log")

# +

fid_k_kms = k_Mpc/fid_dkms_dMpc
fid_P_kms = fid_P_Mpc * fid_dkms_dMpc**3

fig, ax = plt.subplots(figsize=(8,6))
for ii in range(nn):
    col = "C"+str(ii)
    
    x = k_Mpc/list_dkms_dsa[ii]
    y = list_Pk_dsa[ii] * list_dkms_dsa[ii]**3
    ynew = np.exp(np.interp(np.log(fid_k_kms), np.log(x), np.log(y)))
    ax.plot(fid_k_kms, ynew/fid_P_kms-1, col+"-")
    
    x = k_Mpc/list_dkms_ds[ii]
    y = list_Pk_ds[ii] * list_dkms_ds[ii]**3
    ynew = np.exp(np.interp(np.log(fid_k_kms), np.log(x), np.log(y)))
    ax.plot(fid_k_kms, ynew/fid_P_kms-1, col+"--")
# plt.xscale("log")
plt.xlim(0.003, 0.018)
# plt.xlim(0.35, 1.4)
plt.ylim(-0.001, 0.001)

# Mpc

# +
cmap = plt.get_cmap("turbo")


fig, ax = plt.subplots(2, figsize=(8,6), sharex=True, sharey=True)
ftsize = 20
for ii in range(nn):
    col = cmap(ii/nn)
    lab = r"$\Omega_\mathrm{cdm} h^2=$"+str(np.round(omh2[ii], 3))
    if ii < nn//2:
        lab1 = lab
        lab2 = None
    else:
        lab1 = None
        lab2 = lab
          
    ax[0].plot(k_par_kms, list_P1D_ds[ii]/fid_p1d_kms-1, color=col, label=lab1)          
    ax[1].plot(k_par_kms, list_P1D_dsa[ii]/fid_p1d_kms-1, color=col, label=lab2)

for ii in range(2):
    ax[ii].axhline(color="k", ls=":")
    # ax[ii].set_xscale("log")
    ax[ii].legend(loc="upper right", fontsize=ftsize-7, ncol=3)
    # ax[ii].set_xlim(0.04, 55)
    # ax[ii].set_ylim(-0.05, 0.08)
    ax[ii].tick_params(axis="both", which="major", labelsize=ftsize)

ax[0].set_title(r"Scaling $\Delta^2_\star$, $n_\star$", fontsize=ftsize)
ax[1].set_title(r"Scaling $\Delta^2_\star$, $n_\star$, $\alpha_\star$", fontsize=ftsize)

fig.supylabel(r"$P_\mathrm{1D}^\mathrm{scaled}/P_\mathrm{1D}^\mathrm{Planck}-1$", fontsize=ftsize)
fig.supxlabel(r"$k_\parallel\,\left[\mathrm{km}^{-1}\mathrm{s}\right]$", fontsize=ftsize)
plt.tight_layout()
plt.savefig("figs/scalings_omh2_DN_DNA.pdf", bbox_inches='tight')
plt.savefig("figs/scalings_omh2_DN_DNA.png", bbox_inches='tight')

# +
nn = 6
h0 = np.linspace(67.66-10, 67.66+10, nn)

list_Pk_dsa = []
list_P1D_dsa = []
list_dkms_dsa = []

list_Pk_ds = []
list_P1D_ds = []
list_dkms_ds = []
for ii in range(nn):
    print(h0[ii])
    cosmo = camb_cosmo.get_cosmology(
            H0=h0[ii],
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
    camb_object = CAMB_model.CAMBModel([zstar], cosmo=cosmo, z_star=zstar, kp_kms=kp_kms, fast_camb=False)
    inter_params = camb_object.get_linP_params()
    
    dkms_dMpc = camb_object.dkms_dMpc(zstar)
    kp_Mpc = kp_kms * dkms_dMpc
    ln_kp_ks0 = np.log(kp_Mpc / ks_Mpc)

    test_Astar = inter_params["Delta2_star"]
    test_nstar = inter_params["n_star"]
    test_alphastar = inter_params["alpha_star"]    
    
    ln_ratio_Astar = np.log(test_Astar / fid_Astar)
    delta_nstar = test_nstar - fid_nstar
    delta_alphastar = test_alphastar - fid_alphastar

    # scale first    
    delta_nrun = delta_alphastar
    delta_ns = delta_nstar - delta_nrun * ln_kp_ks0
    ln_ratio_As = (
        ln_ratio_Astar
        - (delta_ns + 0.5 * delta_nrun * ln_kp_ks0) * ln_kp_ks0
    )

    cosmo = camb_cosmo.get_cosmology(
            H0=h0[ii],
            mnu=0.0,
            omch2=0.119,
            ombh2=0.0224,
            omk=0.0,
            As=2.105e-09 / np.exp(ln_ratio_As),
            ns=0.9665 - delta_ns,
            nrun=0.0 - delta_nrun,
            pivot_scalar=0.05,
            w=-1,
        )
    camb_results = camb_cosmo.get_camb_results(cosmo, zs=[zstar], fast_camb=False, camb_kmax_Mpc=kmax_Mpc)
    camb_object = CAMB_model.CAMBModel([zstar], cosmo=cosmo, z_star=zstar, kp_kms=kp_kms, fast_camb=False)
    dkms_dMpc = camb_object.dkms_dMpc(zstar)
    list_dkms_dsa.append(dkms_dMpc)
    inter_params = camb_object.get_linP_params()
    print(inter_params)
    
    k_Mpc, zs_out, P_Mpc = camb_cosmo.get_linP_Mpc(cosmo, [zstar], kmax_Mpc=kmax_Mpc, camb_results=camb_results)
    list_Pk_dsa.append(P_Mpc[0])
    list_P1D_dsa.append(P1D_kms(k_par_kms, k_Mpc, P_Mpc[0], dkms_dMpc))

    # scale second    
    delta_nrun = 0
    delta_ns = delta_nstar - delta_nrun * ln_kp_ks0
    ln_ratio_As = (
        ln_ratio_Astar
        - (delta_ns + 0.5 * delta_nrun * ln_kp_ks0) * ln_kp_ks0
    )
    
    cosmo = camb_cosmo.get_cosmology(
            H0=h0[ii],
            mnu=0.0,
            omch2=0.119,
            ombh2=0.0224,
            omk=0.0,
            As=2.105e-09 / np.exp(ln_ratio_As),
            ns=0.9665 - delta_ns,
            nrun=0.0,
            pivot_scalar=0.05,
            w=-1,
        )
    camb_results = camb_cosmo.get_camb_results(cosmo, zs=[zstar], fast_camb=False, camb_kmax_Mpc=kmax_Mpc)
    k_Mpc, zs_out, P_Mpc = camb_cosmo.get_linP_Mpc(cosmo, [zstar], kmax_Mpc=kmax_Mpc, camb_results=camb_results)
    camb_object = CAMB_model.CAMBModel([zstar], cosmo=cosmo, z_star=zstar, kp_kms=kp_kms, fast_camb=False)
    dkms_dMpc = camb_object.dkms_dMpc(zstar)
    list_dkms_ds.append(dkms_dMpc)
    
    inter_params = camb_object.get_linP_params()
    print(inter_params)
    
    list_Pk_ds.append(P_Mpc[0])
    # list_P1D_ds.append(P1D_Mpc(k_par, k_Mpc, P_Mpc[0]))
    list_P1D_ds.append(P1D_kms(k_par_kms, k_Mpc, P_Mpc[0], dkms_dMpc))



# +
cmap = plt.get_cmap("turbo")
fig, ax = plt.subplots(2, figsize=(8,6), sharex=True, sharey=True)
ftsize = 20
for ii in range(nn):
    col = cmap(ii/nn)
    lab = r"$h=$"+str(np.round(h0[ii]/100, 3))
    if ii < nn//2:
        lab1 = lab
        lab2 = None
    else:
        lab1 = None
        lab2 = lab
          
    ax[0].plot(k_par_kms, list_P1D_ds[ii]/fid_p1d_kms-1, color=col, label=lab1)          
    ax[1].plot(k_par_kms, list_P1D_dsa[ii]/fid_p1d_kms-1, color=col, label=lab2)

for ii in range(2):
    ax[ii].axhline(color="k", ls=":")
    # ax[ii].set_xscale("log")
    ax[ii].legend(loc="upper right", fontsize=ftsize-7, ncol=3)
    # ax[ii].set_xlim(0.04, 55)
    # ax[ii].set_ylim(-0.05, 0.08)
    ax[ii].tick_params(axis="both", which="major", labelsize=ftsize)

ax[0].set_title(r"Scaling $\Delta^2_\star$, $n_\star$", fontsize=ftsize)
ax[1].set_title(r"Scaling $\Delta^2_\star$, $n_\star$, $\alpha_\star$", fontsize=ftsize)

fig.supylabel(r"$P_\mathrm{1D}^\mathrm{scaled}/P_\mathrm{1D}^\mathrm{Planck}-1$", fontsize=ftsize)
fig.supxlabel(r"$k_\parallel\,\left[\mathrm{km}^{-1}\mathrm{s}\right]$", fontsize=ftsize)
plt.tight_layout()
plt.savefig("figs/scalings_h_DN_DNA.pdf", bbox_inches='tight')
plt.savefig("figs/scalings_h_DN_DNA.png", bbox_inches='tight')

# +
nn = 6
mnu = np.linspace(0, 0.3, nn)
zstar = 3
kp_kms=0.009
ks_Mpc = 0.05

list_Pk_dsa = []
list_P1D_dsa = []
list_dkms_dsa = []

list_Pk_ds = []
list_P1D_ds = []
list_dkms_ds = []
for ii in range(nn):
    print(mnu[ii])
    cosmo = camb_cosmo.get_cosmology(
            H0=67.66,
            mnu=mnu[ii],
            omch2=0.119,
            ombh2=0.0224,
            omk=0.0,
            As=2.105e-09,
            ns=0.9665,
            nrun=0.0,
            pivot_scalar=0.05,
            w=-1,
        )
    fid_omnuh2 = cosmo.omnuh2 * 1
    cosmo = camb_cosmo.get_cosmology(
            H0=67.66,
            mnu=mnu[ii],
            omch2=0.119 - fid_omnuh2,
            ombh2=0.0224,
            omk=0.0,
            As=2.105e-09,
            ns=0.9665,
            nrun=0.0,
            pivot_scalar=0.05,
            w=-1,
        )
    camb_object = CAMB_model.CAMBModel([zstar], cosmo=cosmo, z_star=zstar, kp_kms=kp_kms, fast_camb=False)
    inter_params = camb_object.get_linP_params()
    
    dkms_dMpc = camb_object.dkms_dMpc(zstar)
    kp_Mpc = kp_kms * dkms_dMpc
    ln_kp_ks0 = np.log(kp_Mpc / ks_Mpc)

    test_Astar = inter_params["Delta2_star"]
    test_nstar = inter_params["n_star"]
    test_alphastar = inter_params["alpha_star"]    
    
    ln_ratio_Astar = np.log(test_Astar / fid_Astar)
    delta_nstar = test_nstar - fid_nstar
    delta_alphastar = test_alphastar - fid_alphastar

    # scale first    
    delta_nrun = delta_alphastar
    delta_ns = delta_nstar - delta_nrun * ln_kp_ks0
    ln_ratio_As = (
        ln_ratio_Astar
        - (delta_ns + 0.5 * delta_nrun * ln_kp_ks0) * ln_kp_ks0
    )

    cosmo = camb_cosmo.get_cosmology(
            H0=67.66,
            mnu=mnu[ii],
            omch2=0.119 - fid_omnuh2,
            ombh2=0.0224,
            omk=0.0,
            As=2.105e-09 / np.exp(ln_ratio_As),
            ns=0.9665 - delta_ns,
            nrun=0.0 - delta_nrun,
            pivot_scalar=0.05,
            w=-1,
        )
    camb_results = camb_cosmo.get_camb_results(cosmo, zs=[zstar], fast_camb=False, camb_kmax_Mpc=kmax_Mpc)
    camb_object = CAMB_model.CAMBModel([zstar], cosmo=cosmo, z_star=zstar, kp_kms=kp_kms, fast_camb=False)
    dkms_dMpc = camb_object.dkms_dMpc(zstar)
    list_dkms_dsa.append(dkms_dMpc)
    inter_params = camb_object.get_linP_params()
    print(inter_params)
    
    k_Mpc, zs_out, P_Mpc = camb_cosmo.get_linP_Mpc(cosmo, [zstar], kmax_Mpc=kmax_Mpc, camb_results=camb_results)
    list_Pk_dsa.append(P_Mpc[0])
    list_P1D_dsa.append(P1D_kms(k_par_kms, k_Mpc, P_Mpc[0], dkms_dMpc))

    # scale second    
    delta_nrun = 0
    delta_ns = delta_nstar - delta_nrun * ln_kp_ks0
    ln_ratio_As = (
        ln_ratio_Astar
        - (delta_ns + 0.5 * delta_nrun * ln_kp_ks0) * ln_kp_ks0
    )
    
    cosmo = camb_cosmo.get_cosmology(
            H0=67.66,
            mnu=mnu[ii],
            omch2=0.119 - fid_omnuh2,
            ombh2=0.0224,
            omk=0.0,
            As=2.105e-09 / np.exp(ln_ratio_As),
            ns=0.9665 - delta_ns,
            nrun=0.0,
            pivot_scalar=0.05,
            w=-1,
        )
    camb_results = camb_cosmo.get_camb_results(cosmo, zs=[zstar], fast_camb=False, camb_kmax_Mpc=kmax_Mpc)
    k_Mpc, zs_out, P_Mpc = camb_cosmo.get_linP_Mpc(cosmo, [zstar], kmax_Mpc=kmax_Mpc, camb_results=camb_results)
    camb_object = CAMB_model.CAMBModel([zstar], cosmo=cosmo, z_star=zstar, kp_kms=kp_kms, fast_camb=False)
    dkms_dMpc = camb_object.dkms_dMpc(zstar)
    list_dkms_ds.append(dkms_dMpc)
    
    inter_params = camb_object.get_linP_params()
    print(inter_params)
    
    list_Pk_ds.append(P_Mpc[0])
    # list_P1D_ds.append(P1D_Mpc(k_par, k_Mpc, P_Mpc[0]))
    list_P1D_ds.append(P1D_kms(k_par_kms, k_Mpc, P_Mpc[0], dkms_dMpc))



# +
cmap = plt.get_cmap("turbo")
fig, ax = plt.subplots(2, figsize=(8,6), sharex=True, sharey=True)
ftsize = 20
for ii in range(nn):
    col = cmap(ii/nn)
    lab = r"$\sum m_\nu=$"+str(np.round(mnu[ii], 2))+" eV"
    if ii < nn//2:
        lab1 = lab
        lab2 = None
    else:
        lab1 = None
        lab2 = lab
          
    ax[0].plot(k_par_kms, list_P1D_ds[ii]/fid_p1d_kms-1, color=col, label=lab1)          
    ax[1].plot(k_par_kms, list_P1D_dsa[ii]/fid_p1d_kms-1, color=col, label=lab2)

for ii in range(2):
    ax[ii].axhline(color="k", ls=":")
    # ax[ii].set_xscale("log")
    ax[ii].legend(loc="upper right", fontsize=ftsize-7, ncol=3)
    # ax[ii].set_xlim(0.04, 55)
    ax[ii].set_ylim(-0.011, 0.011)
    ax[ii].tick_params(axis="both", which="major", labelsize=ftsize)
    
ax[0].legend(loc="lower right", fontsize=ftsize-6, ncol=3)
ax[1].legend(loc="upper right", fontsize=ftsize-6, ncol=3)

ax[0].set_title(r"Scaling $\Delta^2_\star$, $n_\star$", fontsize=ftsize)
ax[1].set_title(r"Scaling $\Delta^2_\star$, $n_\star$, $\alpha_\star$", fontsize=ftsize)

fig.supylabel(r"$P_\mathrm{1D}^\mathrm{scaled}/P_\mathrm{1D}^\mathrm{Planck}-1$", fontsize=ftsize)
fig.supxlabel(r"$k_\parallel\,\left[\mathrm{km}^{-1}\mathrm{s}\right]$", fontsize=ftsize)
plt.tight_layout()
plt.savefig("figs/scalings_mnu_DN_DNA.pdf", bbox_inches='tight')
plt.savefig("figs/scalings_mnu_DN_DNA.png", bbox_inches='tight')
# -


