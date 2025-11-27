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
# %load_ext autoreload
# %autoreload 2

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
    
# def P1D_kms(
#     k_par_kms,
#     k_pk_Mpc, 
#     pk_Mpc,
#     dkms_dMpc,
#     k_perp_min=0.001,
#     k_perp_max=400,
#     n_k_perp=99,
#     kpressure_kms=0.4,
# ):
#     """
#     Returns P1D for specified values of k_par, with the option to specify values of k_perp to be integrated over.

#     Parameters:
#         z (float): Redshift.
#         k_par (array-like): Array or list of values for which P1D is to be computed.
#         k_perp_min (float, optional): Lower bound of integral. Defaults to 0.001.
#         k_perp_max (float, optional): Upper bound of integral. Defaults to 100.
#         n_k_perp (int, optional): Number of points in integral. Defaults to 99.
#         parameters (dict, optional): Additional parameters for the model. Defaults to {}.

#     Returns:
#         array-like: Computed values of P1D.
#     """

#     ln_k_perp_kms = np.linspace(
#         np.log(k_perp_min/dkms_dMpc), np.log(k_perp_max/dkms_dMpc), n_k_perp
#     )
#     kpressure = kpressure_kms * dkms_dMpc
#     damp = np.exp(-1 * pk_Mpc**2 / kpressure**2)
#     k_kms = k_pk_Mpc / dkms_dMpc
#     pk_kms = pk_Mpc * damp * dkms_dMpc**3
#     p1d = _P1D_lnkperp(ln_k_perp_kms, k_par_kms, k_kms, pk_kms)

#     return p1d

 
def P1D_kms(
    k_par_kms,
    k_pk_kms, 
    pk_kms,
    k_perp_min=1e-6,
    k_perp_max=5,
    n_k_perp=99,
    kpressure_kms=0.4,
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

    ln_k_perp_kms = np.linspace(
        np.log(k_perp_min), np.log(k_perp_max), n_k_perp
    )
    damp = np.exp(-1 * k_pk_kms**2 / kpressure_kms**2)
    p1d = _P1D_lnkperp(ln_k_perp_kms, k_par_kms, k_pk_kms, pk_kms * damp)

    return p1d
# -


# +
kmax_Mpc = 400

mm = 40
# k_par_Mpc = np.geomspace(0.01, 5, mm)
# k_par_kms = np.geomspace(0.001, 0.05, mm)
k_par_kms = np.linspace(0.001, 0.04, mm)

# zstar = 20
zstar = 3
kp_kms = 0.009

cosmo = camb_cosmo.get_cosmology(
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
camb_object = CAMB_model.CAMBModel([zstar], cosmo=cosmo, z_star=zstar, kp_kms=kp_kms, fast_camb=False)
camb_results = camb_cosmo.get_camb_results(cosmo, zs=[zstar], fast_camb=False, camb_kmax_Mpc=kmax_Mpc)
target_linP = camb_object.get_linP_params()

fid_Astar = target_linP["Delta2_star"]
fid_nstar = target_linP["n_star"]
fid_alphastar = target_linP["alpha_star"]

fid_dkms_dMpc = camb_object.dkms_dMpc(zstar)
kp_Mpc = kp_kms * fid_dkms_dMpc

# kp_Mpc = 0.7
ks_Mpc = camb_object.cosmo.InitPower.pivot_scalar
ln_kp_ks = np.log(kp_Mpc / ks_Mpc)

fid_k_kms, zs_out, fid_P_kms = camb_cosmo.get_linP_kms(cosmo, [zstar], camb_results=camb_results, kmax_Mpc=kmax_Mpc)
fid_k_kms = fid_k_kms[0]
fid_P_kms = fid_P_kms[0]

# fid_p1d_Mpc = P1D_Mpc(k_par_Mpc, k_Mpc, fid_P_Mpc)
# k_Mpc, zs_out, P_Mpc = camb_cosmo.get_linP_Mpc(cosmo, [zstar], camb_results=camb_results, kmax_Mpc=kmax_Mpc)
fid_p1d_kms = P1D_kms(k_par_kms, fid_k_kms, fid_P_kms)
fid_hz = camb_results.hubble_parameter(zstar)

# -




ind = np.argmin(np.abs(fid_k_kms - kp_kms))
fid_P_kms[ind] * fid_k_kms[ind]**3/2./np.pi**2

# +
# fid_p1d_kms2 = fid_p1d_kms.copy()
# -

plt.plot(k_par_kms, k_par_kms*fid_p1d_kms/np.pi, ".-")
# plt.plot(k_par_kms, k_par_kms*fid_p1d_kms2, ".-")
# plt.yscale("log")

# +
nn = 6
omh2 = np.linspace(0.1071, 0.1309, nn)

list_k_dsa = []
list_Pk_dsa = []
list_P1D_dsa = []
list_dkms_dsa = []

list_k_ds = []
list_Pk_ds = []
list_P1D_ds = []
list_dkms_ds = []

list_k_no = []
list_Pk_no = []
list_P1D_no = []
# list_dkms_ds = []
for ii in range(nn):
    print(omh2[ii])

    # cosmo with new value of Omh2
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
    camb_results = camb_cosmo.get_camb_results(cosmo, zs=[zstar], fast_camb=False, camb_kmax_Mpc=kmax_Mpc)
    k_kms, zs_out, P_kms = camb_cosmo.get_linP_kms(cosmo, [zstar], camb_results=camb_results, kmax_Mpc=kmax_Mpc)
    list_k_no.append(k_kms[0])
    list_Pk_no.append(P_kms[0])
    list_P1D_no.append(P1D_kms(k_par_kms, k_kms[0], P_kms[0]))
    
    camb_object = CAMB_model.CAMBModel([zstar], cosmo=cosmo, z_star=zstar, kp_kms=kp_kms, fast_camb=False)
    inter_params = camb_object.get_linP_params()
    
    dkms_dMpc = camb_object.dkms_dMpc(zstar)
    kp_Mpc = kp_kms * dkms_dMpc
    ln_kp_ks0 = np.log(kp_Mpc / ks_Mpc)

    test_Astar = inter_params["Delta2_star"]
    test_nstar = inter_params["n_star"]
    test_alphastar = inter_params["alpha_star"]

    # rescale cosmology to match fid values of the compressed parameters
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

    # evaluate rescaled cosmology
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
    
    k_kms, zs_out, P_kms = camb_cosmo.get_linP_kms(cosmo, [zstar], camb_results=camb_results, kmax_Mpc=kmax_Mpc)
    list_k_dsa.append(k_kms[0])
    list_Pk_dsa.append(P_kms[0])
    list_P1D_dsa.append(P1D_kms(k_par_kms, k_kms[0], P_kms[0]))

    # compute second scaling, not fixing nrun now
    delta_nrun = 0
    delta_ns = delta_nstar - delta_nrun * ln_kp_ks0
    ln_ratio_As = (
        ln_ratio_Astar
        - (delta_ns + 0.5 * delta_nrun * ln_kp_ks0) * ln_kp_ks0
    )

    # evaluate rescaled cosmology
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
    camb_object = CAMB_model.CAMBModel([zstar], cosmo=cosmo, z_star=zstar, kp_kms=kp_kms, fast_camb=False)
    dkms_dMpc = camb_object.dkms_dMpc(zstar)
    list_dkms_ds.append(dkms_dMpc)
    
    inter_params = camb_object.get_linP_params()
    print(inter_params)
    
    k_kms, zs_out, P_kms = camb_cosmo.get_linP_kms(cosmo, [zstar], camb_results=camb_results, kmax_Mpc=kmax_Mpc)
    list_k_ds.append(k_kms[0])
    list_Pk_ds.append(P_kms[0])
    list_P1D_ds.append(P1D_kms(k_par_kms, k_kms[0], P_kms[0]))


# +
cmap = plt.get_cmap("turbo")


fig, ax = plt.subplots(3, figsize=(8,8), sharex=True)
ftsize = 20
for ii in range(nn):
    col = cmap(ii/nn)
    lab = r"$\Omega_\mathrm{cdm} h^2=$"+str(np.round(omh2[ii], 3))
    if ii < nn//3:
        lab1 = lab
        lab2 = None
        lab3 = None
    elif ii < 2*nn//3:
        lab1 = None
        lab2 = lab
        lab3 = None
    else:
        lab1 = None
        lab2 = None
        lab3 = lab
          
    ax[0].plot(k_par_kms, list_P1D_no[ii]/fid_p1d_kms-1, color=col, label=lab1) 
    ax[1].plot(k_par_kms, list_P1D_ds[ii]/fid_p1d_kms-1, color=col, label=lab2)          
    ax[2].plot(k_par_kms, list_P1D_dsa[ii]/fid_p1d_kms-1, color=col, label=lab3)

for ii in range(len(ax)):
    ax[ii].axhline(color="k", ls=":")
    ax[ii].legend(loc="upper left", fontsize=ftsize-5, ncol=3)
    ax[ii].tick_params(axis="both", which="major", labelsize=ftsize)

yrange = [-0.02, 0.02]
for ii in range(1, 3):
    ax[ii].set_ylim(yrange)

ax[0].set_title(r"No scaling", fontsize=ftsize)
ax[1].set_title(r"Scaling $\Delta^2_\star$, $n_\star$", fontsize=ftsize)
ax[2].set_title(r"Scaling $\Delta^2_\star$, $n_\star$, $\alpha_\star$", fontsize=ftsize)

fig.supylabel(r"$P_\mathrm{1D}^\mathrm{scaled}/P_\mathrm{1D}^\mathrm{fid}-1$", fontsize=ftsize)
fig.supxlabel(r"$k_\parallel\,\left[\mathrm{km}^{-1}\mathrm{s}\right]$", fontsize=ftsize)
plt.tight_layout()
plt.savefig("figs/scalings_omh2_DN_DNA.pdf", bbox_inches='tight')
plt.savefig("figs/scalings_omh2_DN_DNA.png", bbox_inches='tight')

# +
cmap = plt.get_cmap("turbo")


fig, ax = plt.subplots(3, figsize=(8,8), sharex=True)
ftsize = 20
for ii in range(nn):
    col = cmap(ii/nn)
    lab = r"$\Omega_\mathrm{cdm} h^2=$"+str(np.round(omh2[ii], 3))
    if ii < nn//3:
        lab1 = lab
        lab2 = None
        lab3 = None
    elif ii < 2*nn//3:
        lab1 = None
        lab2 = lab
        lab3 = None
    else:
        lab1 = None
        lab2 = None
        lab3 = lab

    yy = np.interp(fid_k_kms, list_k_no[ii], list_Pk_no[ii])
    ax[0].plot(fid_k_kms, yy/fid_P_kms-1, color=col, label=lab1)
    yy = np.interp(fid_k_kms, list_k_ds[ii], list_Pk_ds[ii])
    ax[1].plot(fid_k_kms, yy/fid_P_kms-1, color=col, label=lab2)
    yy = np.interp(fid_k_kms, list_k_dsa[ii], list_Pk_dsa[ii])
    ax[2].plot(fid_k_kms, yy/fid_P_kms-1, color=col, label=lab3)

for ii in range(len(ax)):
    ax[ii].axhline(color="k", ls=":")
    ax[ii].legend(loc="upper left", fontsize=ftsize-5, ncol=3)
    ax[ii].tick_params(axis="both", which="major", labelsize=ftsize)

yrange = [-0.02, 0.02]
for ii in range(1, 3):
    ax[ii].set_ylim(yrange)
    ax[ii].axvline(kp_kms, ls=":", color="k")
ax[0].set_xscale("log")
ax[0].set_xlim(1e-3, 0.1)
ax[0].set_ylim(-0.25, 0.25)

ax[0].set_title(r"No scaling", fontsize=ftsize)
ax[1].set_title(r"Scaling $\Delta^2_\star$, $n_\star$", fontsize=ftsize)
ax[2].set_title(r"Scaling $\Delta^2_\star$, $n_\star$, $\alpha_\star$", fontsize=ftsize)

fig.supylabel(r"$P_\mathrm{lin}^\mathrm{scaled}/P_\mathrm{lin}^\mathrm{fid}-1$", fontsize=ftsize)
fig.supxlabel(r"$k\,\left[\mathrm{km}^{-1}\mathrm{s}\right]$", fontsize=ftsize)
plt.tight_layout()
plt.savefig("figs/p3dscalings_omh2_DN_DNA.pdf", bbox_inches='tight')
plt.savefig("figs/p3dscalings_omh2_DN_DNA.png", bbox_inches='tight')
# -

h0 - 67.66

# +
nn = 6
h0 = np.linspace(67.66-10, 67.66+10, nn)
# h0 = np.linspace(67.66-5, 67.66+5, nn)

list_k_dsa = []
list_Pk_dsa = []
list_P1D_dsa = []
list_dkms_dsa = []

list_k_ds = []
list_Pk_ds = []
list_P1D_ds = []
list_dkms_ds = []

list_k_no = []
list_Pk_no = []
list_P1D_no = []
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
    camb_results = camb_cosmo.get_camb_results(cosmo, zs=[zstar], fast_camb=False, camb_kmax_Mpc=kmax_Mpc)
    k_kms, zs_out, P_kms = camb_cosmo.get_linP_kms(cosmo, [zstar], camb_results=camb_results, kmax_Mpc=kmax_Mpc)
    list_k_no.append(k_kms[0])
    list_Pk_no.append(P_kms[0])
    list_P1D_no.append(P1D_kms(k_par_kms, k_kms[0], P_kms[0]))
    
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
    print(camb_results.hubble_parameter(zstar)/fid_hz)
    
    
    k_kms, zs_out, P_kms = camb_cosmo.get_linP_kms(cosmo, [zstar], camb_results=camb_results, kmax_Mpc=kmax_Mpc)
    list_k_dsa.append(k_kms[0])
    list_Pk_dsa.append(P_kms[0])
    list_P1D_dsa.append(P1D_kms(k_par_kms, k_kms[0], P_kms[0]))

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
    # k_Mpc, zs_out, P_Mpc = camb_cosmo.get_linP_Mpc(cosmo, [zstar], kmax_Mpc=kmax_Mpc, camb_results=camb_results)
    camb_object = CAMB_model.CAMBModel([zstar], cosmo=cosmo, z_star=zstar, kp_kms=kp_kms, fast_camb=False)
    dkms_dMpc = camb_object.dkms_dMpc(zstar)
    list_dkms_ds.append(dkms_dMpc)
    
    inter_params = camb_object.get_linP_params()
    print(inter_params)
    print(camb_results.hubble_parameter(zstar)/fid_hz)
    
    # list_P1D_ds.append(P1D_Mpc(k_par, k_Mpc, P_Mpc[0]))
    k_kms, zs_out, P_kms = camb_cosmo.get_linP_kms(cosmo, [zstar], camb_results=camb_results, kmax_Mpc=kmax_Mpc)
    list_k_ds.append(k_kms[0])
    list_Pk_ds.append(P_kms[0])
    list_P1D_ds.append(P1D_kms(k_par_kms, k_kms[0], P_kms[0]))

    # break
# -


ind = np.argmin(np.abs(fid_k_kms - kp_kms))
fid_P_kms[ind] * fid_k_kms[ind]**3/2./np.pi**2

0.35422734587648913

ii = 0
ind = np.argmin(np.abs(list_k_dsa[ii] - kp_kms))
list_Pk_dsa[ii][ind] * list_k_dsa[ii][ind]**3/2./np.pi**2



fig, ax = plt.subplots(figsize=(8,6))
for ii in range(nn):
    col = "C"+str(ii)
    lab = r"$h=$"+str(np.round(h0[ii]/100, 3))
    yy = np.interp(fid_k_kms, list_k_dsa[ii], list_Pk_dsa[ii])
    ax.plot(fid_k_kms, yy/fid_P_kms-1, col+"-", label=lab)
    yy = np.interp(fid_k_kms, list_k_ds[ii], list_Pk_ds[ii])
    ax.plot(fid_k_kms, yy/fid_P_kms-1, col+"--")
plt.legend(loc="upper right")
plt.xscale("log")
plt.ylabel("P3D residual")
plt.xlabel("k [s/km]")
# plt.ylim(-0.01, 0.01)
plt.ylim(-0.0005, 0.0005)
plt.xlim(0.004, 0.02)

# +
cmap = plt.get_cmap("turbo")


fig, ax = plt.subplots(3, figsize=(8,8), sharex=True)
ftsize = 20
for ii in range(nn):
    col = cmap(ii/nn)
    # lab = r"$\Omega_\mathrm{cdm} h^2=$"+str(np.round(omh2[ii], 3))
    lab = r"$h=$"+str(np.round(h0[ii]/100, 3))
    if ii < nn//3:
        lab1 = lab
        lab2 = None
        lab3 = None
    elif ii < 2*nn//3:
        lab1 = None
        lab2 = lab
        lab3 = None
    else:
        lab1 = None
        lab2 = None
        lab3 = lab
          
    ax[0].plot(k_par_kms, list_P1D_no[ii]/fid_p1d_kms-1, color=col, label=lab1) 
    ax[1].plot(k_par_kms, list_P1D_ds[ii]/fid_p1d_kms-1, color=col, label=lab2)          
    ax[2].plot(k_par_kms, list_P1D_dsa[ii]/fid_p1d_kms-1, color=col, label=lab3)

for ii in range(len(ax)):
    ax[ii].axhline(color="k", ls=":")
    ax[ii].legend(loc="upper left", fontsize=ftsize-5, ncol=3)
    ax[ii].tick_params(axis="both", which="major", labelsize=ftsize)

yrange = [-0.003, 0.003]
for ii in range(0, 3):
    ax[ii].set_ylim(yrange)

ax[0].set_title(r"No scaling", fontsize=ftsize)
ax[1].set_title(r"Scaling $\Delta^2_\star$, $n_\star$", fontsize=ftsize)
ax[2].set_title(r"Scaling $\Delta^2_\star$, $n_\star$, $\alpha_\star$", fontsize=ftsize)

fig.supylabel(r"$P_\mathrm{1D}^\mathrm{scaled}/P_\mathrm{1D}^\mathrm{fid}-1$", fontsize=ftsize)
fig.supxlabel(r"$k_\parallel\,\left[\mathrm{km}^{-1}\mathrm{s}\right]$", fontsize=ftsize)
plt.tight_layout()
plt.savefig("figs/scalings_h_DN_DNA.pdf", bbox_inches='tight')
plt.savefig("figs/scalings_h_DN_DNA.png", bbox_inches='tight')

# +
cmap = plt.get_cmap("turbo")


fig, ax = plt.subplots(3, figsize=(8,8), sharex=True)
ftsize = 20
for ii in range(nn):
    col = cmap(ii/nn)
    # lab = r"$\Omega_\mathrm{cdm} h^2=$"+str(np.round(omh2[ii], 3))
    lab = r"$h=$"+str(np.round(h0[ii]/100, 3))
    if ii < nn//3:
        lab1 = lab
        lab2 = None
        lab3 = None
    elif ii < 2*nn//3:
        lab1 = None
        lab2 = lab
        lab3 = None
    else:
        lab1 = None
        lab2 = None
        lab3 = lab

    yy = np.interp(fid_k_kms, list_k_no[ii], list_Pk_no[ii])
    ax[0].plot(fid_k_kms, yy/fid_P_kms-1, color=col, label=lab1)
    yy = np.interp(fid_k_kms, list_k_ds[ii], list_Pk_ds[ii])
    ax[1].plot(fid_k_kms, yy/fid_P_kms-1, color=col, label=lab2)
    yy = np.interp(fid_k_kms, list_k_dsa[ii], list_Pk_dsa[ii])
    ax[2].plot(fid_k_kms, yy/fid_P_kms-1, color=col, label=lab3)

for ii in range(len(ax)):
    ax[ii].axhline(color="k", ls=":")
    ax[ii].legend(loc="upper left", fontsize=ftsize-5, ncol=3)
    ax[ii].tick_params(axis="both", which="major", labelsize=ftsize)

yrange = [-0.01, 0.01]
for ii in range(1, 3):
    ax[ii].set_ylim(yrange)
    ax[ii].axvline(kp_kms, ls=":", color="k")
ax[0].set_xscale("log")
ax[0].set_xlim(1e-3, 0.1)
ax[0].set_ylim(yrange)

ax[0].set_title(r"No scaling", fontsize=ftsize)
ax[1].set_title(r"Scaling $\Delta^2_\star$, $n_\star$", fontsize=ftsize)
ax[2].set_title(r"Scaling $\Delta^2_\star$, $n_\star$, $\alpha_\star$", fontsize=ftsize)

fig.supylabel(r"$P_\mathrm{lin}^\mathrm{scaled}/P_\mathrm{lin}^\mathrm{fid}-1$", fontsize=ftsize)
fig.supxlabel(r"$k\,\left[\mathrm{km}^{-1}\mathrm{s}\right]$", fontsize=ftsize)
plt.tight_layout()
plt.savefig("figs/p3dscalings_h_DN_DNA.pdf", bbox_inches='tight')
plt.savefig("figs/p3dscalings_h_DN_DNA.png", bbox_inches='tight')
# -







# +
nn = 6
mnu = np.linspace(0.06, 0.3, nn)
# zstar = 3
# kp_kms=0.009
# ks_Mpc = 0.05

list_k_dsa = []
list_Pk_dsa = []
list_P1D_dsa = []
list_dkms_dsa = []

list_k_ds = []
list_Pk_ds = []
list_P1D_ds = []
list_dkms_ds = []

list_k_no = []
list_Pk_no = []
list_P1D_no = []
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
    camb_results = camb_cosmo.get_camb_results(cosmo, zs=[zstar], fast_camb=False, camb_kmax_Mpc=kmax_Mpc)
    k_kms, zs_out, P_kms = camb_cosmo.get_linP_kms(cosmo, [zstar], camb_results=camb_results, kmax_Mpc=kmax_Mpc)
    list_k_no.append(k_kms[0])
    list_Pk_no.append(P_kms[0])
    list_P1D_no.append(P1D_kms(k_par_kms, k_kms[0], P_kms[0]))
    
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
    

    k_kms, zs_out, P_kms = camb_cosmo.get_linP_kms(cosmo, [zstar], camb_results=camb_results, kmax_Mpc=kmax_Mpc)
    list_k_dsa.append(k_kms[0])
    list_Pk_dsa.append(P_kms[0])
    list_P1D_dsa.append(P1D_kms(k_par_kms, k_kms[0], P_kms[0]))

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
    
    k_kms, zs_out, P_kms = camb_cosmo.get_linP_kms(cosmo, [zstar], camb_results=camb_results, kmax_Mpc=kmax_Mpc)
    list_k_ds.append(k_kms[0])
    list_Pk_ds.append(P_kms[0])
    list_P1D_ds.append(P1D_kms(k_par_kms, k_kms[0], P_kms[0]))

# -


fig, ax = plt.subplots(figsize=(8,6))
for ii in range(nn):
    col = "C"+str(ii)
    lab = r"$\sum m_\nu=$"+str(np.round(mnu[ii], 2))+" eV"
    yy = np.interp(fid_k_kms, list_k_dsa[ii], list_Pk_dsa[ii])
    ax.plot(fid_k_kms, yy/fid_P_kms-1, col+"-", label=lab)
    yy = np.interp(fid_k_kms, list_k_ds[ii], list_Pk_ds[ii])
    ax.plot(fid_k_kms, yy/fid_P_kms-1, col+"--")
plt.legend(loc="upper right")
plt.xscale("log")
plt.ylabel("P3D residual")
plt.xlabel("k [s/km]")
plt.ylim(-0.01, 0.01)

# +
cmap = plt.get_cmap("turbo")


fig, ax = plt.subplots(3, figsize=(8,8), sharex=True)
ftsize = 20
for ii in range(nn):
    col = cmap(ii/nn)
    # lab = r"$\Omega_\mathrm{cdm} h^2=$"+str(np.round(omh2[ii], 3))
    # lab = r"$h=$"+str(np.round(h0[ii]/100, 3))
    lab = r"$\sum m_\nu=$"+str(np.round(mnu[ii], 2))+" eV"
    if ii < nn//3:
        lab1 = lab
        lab2 = None
        lab3 = None
    elif ii < 2*nn//3:
        lab1 = None
        lab2 = lab
        lab3 = None
    else:
        lab1 = None
        lab2 = None
        lab3 = lab
          
    ax[0].plot(k_par_kms, list_P1D_no[ii]/fid_p1d_kms-1, color=col, label=lab1) 
    ax[1].plot(k_par_kms, list_P1D_ds[ii]/fid_p1d_kms-1, color=col, label=lab2)          
    ax[2].plot(k_par_kms, list_P1D_dsa[ii]/fid_p1d_kms-1, color=col, label=lab3)

for ii in range(len(ax)):
    ax[ii].axhline(color="k", ls=":")
    ax[ii].legend(loc="upper left", fontsize=ftsize-5, ncol=3)
    ax[ii].tick_params(axis="both", which="major", labelsize=ftsize)

yrange = [-0.008, 0.008]
for ii in range(1, 3):
    ax[ii].set_ylim(yrange)

ax[0].set_title(r"No scaling", fontsize=ftsize)
ax[1].set_title(r"Scaling $\Delta^2_\star$, $n_\star$", fontsize=ftsize)
ax[2].set_title(r"Scaling $\Delta^2_\star$, $n_\star$, $\alpha_\star$", fontsize=ftsize)

fig.supylabel(r"$P_\mathrm{1D}^\mathrm{scaled}/P_\mathrm{1D}^\mathrm{fid}-1$", fontsize=ftsize)
fig.supxlabel(r"$k_\parallel\,\left[\mathrm{km}^{-1}\mathrm{s}\right]$", fontsize=ftsize)
plt.tight_layout()
plt.savefig("figs/scalings_mnu_DN_DNA.pdf", bbox_inches='tight')
plt.savefig("figs/scalings_mnu_DN_DNA.png", bbox_inches='tight')

# +
cmap = plt.get_cmap("turbo")


fig, ax = plt.subplots(3, figsize=(8,8), sharex=True)
ftsize = 20
for ii in range(nn):
    col = cmap(ii/nn)
    # lab = r"$\Omega_\mathrm{cdm} h^2=$"+str(np.round(omh2[ii], 3))
    # lab = r"$h=$"+str(np.round(h0[ii]/100, 3))
    lab = r"$\sum m_\nu=$"+str(np.round(mnu[ii], 2))+" eV"
    if ii < nn//3:
        lab1 = lab
        lab2 = None
        lab3 = None
    elif ii < 2*nn//3:
        lab1 = None
        lab2 = lab
        lab3 = None
    else:
        lab1 = None
        lab2 = None
        lab3 = lab

    yy = np.interp(fid_k_kms, list_k_no[ii], list_Pk_no[ii])
    ax[0].plot(fid_k_kms, yy/fid_P_kms-1, color=col, label=lab1)
    yy = np.interp(fid_k_kms, list_k_ds[ii], list_Pk_ds[ii])
    ax[1].plot(fid_k_kms, yy/fid_P_kms-1, color=col, label=lab2)
    yy = np.interp(fid_k_kms, list_k_dsa[ii], list_Pk_dsa[ii])
    ax[2].plot(fid_k_kms, yy/fid_P_kms-1, color=col, label=lab3)

for ii in range(len(ax)):
    ax[ii].axhline(color="k", ls=":")
    ax[ii].legend(loc="upper left", fontsize=ftsize-5, ncol=3)
    ax[ii].tick_params(axis="both", which="major", labelsize=ftsize)

yrange = [-0.02, 0.02]
for ii in range(1, 3):
    ax[ii].set_ylim(yrange)
    ax[ii].axvline(kp_kms, ls=":", color="k")
ax[0].set_xscale("log")
ax[0].set_xlim(1e-3, 0.1)
ax[0].set_ylim(-0.25, 0.25)

ax[0].set_title(r"No scaling", fontsize=ftsize)
ax[1].set_title(r"Scaling $\Delta^2_\star$, $n_\star$", fontsize=ftsize)
ax[2].set_title(r"Scaling $\Delta^2_\star$, $n_\star$, $\alpha_\star$", fontsize=ftsize)

fig.supylabel(r"$P_\mathrm{lin}^\mathrm{scaled}/P_\mathrm{lin}^\mathrm{fid}-1$", fontsize=ftsize)
fig.supxlabel(r"$k\,\left[\mathrm{km}^{-1}\mathrm{s}\right]$", fontsize=ftsize)
plt.tight_layout()
plt.savefig("figs/p3dscalings_mnu_DN_DNA.pdf", bbox_inches='tight')
plt.savefig("figs/p3dscalings_mnu_DN_DNA.png", bbox_inches='tight')
# -




