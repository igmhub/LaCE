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

# # Get linP params from CAMELS

import numpy as np
from lace.cosmo import camb_cosmo
from cup1d.likelihood.cosmologies import set_cosmo
from cup1d.likelihood import CAMB_model
import matplotlib.pyplot as plt

file = "/home/jchaves/Proyectos/projects/lya/data/camels/cosmos.txt"
with open(file) as f:
    for line in f:
        if line.startswith('#'):
            col_names = line[1:].strip().split()
            break
data = np.genfromtxt(file, comments='#', names=col_names, dtype=None, encoding=None)
data.dtype

# +
# data["Name"][0]
# -

# #### Running CAMB twice since we do not have As

# +
zs = np.array([1.5, 2, 3, 4, 5, 6])
kp_Mpc = 0.7
Asfid = 2e-9
nsims = len(data["Name"])
nz = len(zs)
lab_pars = ['Delta2_p', 'n_p', 'alpha_p']

# results = np.zeros((nsims, nz, len(lab_pars)))

for ii in range(1463, nsims):
    print(ii)
    cosmo = camb_cosmo.get_cosmology(
        H0=data["HubbleParam"][ii] * 100,
        mnu=0.0,
        omch2=(data["Omega0"][ii] - data["OmegaBaryon"][ii]) * data["HubbleParam"][ii]**2,
        ombh2=data["OmegaBaryon"][ii] * data["HubbleParam"][ii]**2,
        omk=0.0,
        As=Asfid,
        ns=data["n_s"][ii],
        nrun=0.0,
        pivot_scalar=0.05,
        w=-1,
    )
    camb_res = camb_cosmo.get_camb_results(cosmo, zs=[0])
    sig8fid = camb_res.get_sigma8()
    
    cosmo = camb_cosmo.get_cosmology(
        H0=data["HubbleParam"][ii] * 100,
        mnu=0.0,
        omch2=(data["Omega0"][ii] - data["OmegaBaryon"][ii]) * data["HubbleParam"][ii]**2,
        ombh2=data["OmegaBaryon"][ii] * data["HubbleParam"][ii]**2,
        omk=0.0,
        As=Asfid * data["sigma8"][ii]**2/sig8fid**2,
        ns=data["n_s"][ii],
        nrun=0.0,
        pivot_scalar=0.05,
        w=-1,
    )
    
    # to check that it is working
    # camb_res = camb_cosmo.get_camb_results(cosmo, zs=[0])
    # print(camb_res.get_sigma8(), data["sigma8"][ii])
    
    camb_object = CAMB_model.CAMBModel(zs, cosmo=cosmo, z_star=3.0, kp_kms=0.009)
    lin_par = camb_object.get_linP_Mpc_params(kp_Mpc)

    for jj in range(nz):
        for kk, lab in enumerate(lab_pars):
            results[ii, jj, kk] = lin_par[jj][lab]

# +
out = {}
out["linP"] = results
out["zs"] = zs
out["lab_pars"] = lab_pars

np.save("/home/jchaves/Proyectos/projects/lya/data/camels/camels_linP.npy", out)

# +

out = np.load("/home/jchaves/Proyectos/projects/lya/data/camels/camels_linP.npy", allow_pickle=True).item()
results = out["linP"]

# -

idz = np.argwhere(out["zs"] == 3)[0,0]
idz

# ## Plot with camels suite, z

# +
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
for iz in range(len(zs)):
    col = "C"+str(iz)
    ax[0].scatter(results[:, iz, 0], results[:, iz, 1], color=col, label="z="+str(zs[iz]), s=3)
    ax[1].scatter(results[:, iz, 1], results[:, iz, 2], color=col, label="z="+str(zs[iz]), s=3)
    ax[2].scatter(results[:, iz, 0], results[:, iz, 2], color=col, label="z="+str(zs[iz]), s=3)

ax[0].legend()

ax[0].set_xlabel(lab_pars[0])
ax[1].set_xlabel(lab_pars[2])
ax[2].set_xlabel(lab_pars[0])

ax[0].set_ylabel(lab_pars[1])
ax[1].set_ylabel(lab_pars[2])
ax[2].set_ylabel(lab_pars[2])

plt.tight_layout()
plt.savefig("camels_nz.png")
# -



# ## Plot with all suites

mpg_cosmos = set_cosmo(cosmo_label="mpg_central", return_all=True)
nyx_cosmos = set_cosmo(cosmo_label="nyx_central", return_all=True)

# #### IDs of closest simulations

# +
nsim_mpg = 30
nz_mpg = 11
nsim_nyx = 18
nz_nyx = 16
lab_pars = ['Delta2_p', 'n_p', 'alpha_p']

mpg_cosmo = np.zeros((nsim_mpg, nz_mpg, 3))
nyx_cosmo = np.zeros((nsim_nyx, nz_nyx, 3))

for isim in range(nsim_mpg):
    for iz in range(nz_mpg):
        for kk, lab in enumerate(lab_pars):
            mpg_cosmo[isim, iz, kk] = mpg_cosmos["mpg_"+str(isim)]["linP_params"][lab][iz]


for isim in range(nsim_nyx):
    if isim != 14:
        for iz in range(nz_nyx):
            for kk, lab in enumerate(lab_pars):
                nyx_cosmo[isim, iz, kk] = nyx_cosmos["nyx_"+str(isim)]["linP_params"][lab][iz]
# -


nyx_cen = nyx_cosmos["nyx_central"]["linP_params"].copy()
idznyx = nyx_cen["z"] == 3
params = np.array([nyx_cen['Delta2_p'][_], nyx_cen['n_p'][_], nyx_cen['alpha_p'][_]])[:, 0]
params

camels_params = results[:, idz, :]

nyx_cosmo.shape

metric = np.zeros(3)
for ii in range(3):
    _ = nyx_cosmo[:, :, ii] != 0
    metric[ii] = np.max(nyx_cosmo[_, ii]) - np.min(nyx_cosmo[_, ii])
metric

dist = np.zeros(camels_params.shape[0])
for ii in range(3):
    dist += (camels_params[:,ii] - params[ii])**2/metric[ii]**2
np.argmin(dist)

camels_params[890]



# +
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
col = "C0"
ax[0].scatter(results[:, :, 0].reshape(-1), results[:, :, 1].reshape(-1), color=col, s=3, label="camels")
ax[1].scatter(results[:, :, 1].reshape(-1), results[:, :, 2].reshape(-1), color=col, s=3)
ax[2].scatter(results[:, :, 0].reshape(-1), results[:, :, 2].reshape(-1), color=col, s=3)


col = "C3"
x = nyx_cosmo[:, :, 0].reshape(-1)
y = nyx_cosmo[:, :, 1].reshape(-1)
z = nyx_cosmo[:, :, 2].reshape(-1)
_ = (x != 0)
ax[0].scatter(x[_], y[_], color=col, s=3, label="nyx")
ax[1].scatter(y[_], z[_], color=col, s=3)
ax[2].scatter(x[_], z[_], color=col, s=3)

col = "C1"
ax[0].scatter(mpg_cosmo[:, :, 0].reshape(-1), mpg_cosmo[:, :, 1].reshape(-1), color=col, s=3, label="mpg")
ax[1].scatter(mpg_cosmo[:, :, 1].reshape(-1), mpg_cosmo[:, :, 2].reshape(-1), color=col, s=3)
ax[2].scatter(mpg_cosmo[:, :, 0].reshape(-1), mpg_cosmo[:, :, 2].reshape(-1), color=col, s=3)


ax[0].legend()

ax[0].set_xlabel(lab_pars[0])
ax[1].set_xlabel(lab_pars[2])
ax[2].set_xlabel(lab_pars[0])

ax[0].set_ylabel(lab_pars[1])
ax[1].set_ylabel(lab_pars[2])
ax[2].set_ylabel(lab_pars[2])


plt.tight_layout()
plt.savefig("camels_nyx_mpg.png")

# +
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
col = "C0"
ax[0].scatter(results[:, :, 0].reshape(-1), results[:, :, 1].reshape(-1), color=col, s=3, label="camels")
ax[1].scatter(results[:, :, 1].reshape(-1), results[:, :, 2].reshape(-1), color=col, s=3)
ax[2].scatter(results[:, :, 0].reshape(-1), results[:, :, 2].reshape(-1), color=col, s=3)


col = "C3"
x = nyx_cosmo[:, :, 0].reshape(-1)
y = nyx_cosmo[:, :, 1].reshape(-1)
z = nyx_cosmo[:, :, 2].reshape(-1)
_ = (x != 0)
ax[0].scatter(x[_], y[_], color=col, s=3, label="nyx")
ax[1].scatter(y[_], z[_], color=col, s=3)
ax[2].scatter(x[_], z[_], color=col, s=3)

col = "C1"
ax[0].scatter(mpg_cosmo[:, :, 0].reshape(-1), mpg_cosmo[:, :, 1].reshape(-1), color=col, s=3, label="mpg")
ax[1].scatter(mpg_cosmo[:, :, 1].reshape(-1), mpg_cosmo[:, :, 2].reshape(-1), color=col, s=3)
ax[2].scatter(mpg_cosmo[:, :, 0].reshape(-1), mpg_cosmo[:, :, 2].reshape(-1), color=col, s=3)


ax[0].legend()

ax[0].set_xlabel(lab_pars[0])
ax[1].set_xlabel(lab_pars[2])
ax[2].set_xlabel(lab_pars[0])

ax[0].set_ylabel(lab_pars[1])
ax[1].set_ylabel(lab_pars[2])
ax[2].set_ylabel(lab_pars[2])

ax[0].set_xlim(0, 1)
ax[1].set_xlim(-2.5, -2.1)
ax[2].set_xlim(0, 1)

ax[0].set_ylim(-2.5, -2.1)
ax[1].set_ylim(-0.26, -0.16)
ax[2].set_ylim(-0.26, -0.16)


plt.tight_layout()
plt.savefig("camels_nyx_mpg_zoom.png")
# +
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
col = "C0"
ax[0].scatter(results[:, :, 0].reshape(-1), results[:, :, 1].reshape(-1), color=col, s=3, label="camels")
ax[1].scatter(results[:, :, 1].reshape(-1), results[:, :, 2].reshape(-1), color=col, s=3)
ax[2].scatter(results[:, :, 0].reshape(-1), results[:, :, 2].reshape(-1), color=col, s=3)


col = "C3"
x = nyx_cosmo[:, :, 0].reshape(-1)
y = nyx_cosmo[:, :, 1].reshape(-1)
z = nyx_cosmo[:, :, 2].reshape(-1)
_ = (x != 0)
ax[0].scatter(x[_], y[_], color=col, s=3, label="nyx")
ax[1].scatter(y[_], z[_], color=col, s=3)
ax[2].scatter(x[_], z[_], color=col, s=3)

col = "C1"
ax[0].scatter(mpg_cosmo[:, :, 0].reshape(-1), mpg_cosmo[:, :, 1].reshape(-1), color=col, s=3, label="mpg")
ax[1].scatter(mpg_cosmo[:, :, 1].reshape(-1), mpg_cosmo[:, :, 2].reshape(-1), color=col, s=3)
ax[2].scatter(mpg_cosmo[:, :, 0].reshape(-1), mpg_cosmo[:, :, 2].reshape(-1), color=col, s=3)


ax[0].legend()

ax[0].set_xlabel(lab_pars[0])
ax[1].set_xlabel(lab_pars[2])
ax[2].set_xlabel(lab_pars[0])

ax[0].set_ylabel(lab_pars[1])
ax[1].set_ylabel(lab_pars[2])
ax[2].set_ylabel(lab_pars[2])

ax[0].set_xlim(0.05, 0.85)
ax[1].set_xlim(-2.37, -2.23)
ax[2].set_xlim(0.05, 0.85)

ax[0].set_ylim(-2.37, -2.23)
ax[1].set_ylim(-0.235, -0.19)
ax[2].set_ylim(-0.235, -0.19)


plt.tight_layout()
plt.savefig("camels_nyx_mpg_zoom2.png")
# -



