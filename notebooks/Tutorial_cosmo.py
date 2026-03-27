# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: lace
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tutorial explaining how to use the cosmology class

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
from matplotlib import pyplot as plt
from lace.cosmo import cosmology, rescale_cosmology
from camb import model

# %% [markdown]
# ### Define three different cosmologies

# %%
cosmos = []
# Planck18 by default
cosmos.append(cosmology.Cosmology())
# Planck18
cosmos.append(cosmology.Cosmology(cosmo_label="Planck18"))
# Planck18 with H0=74
cosmos.append(cosmology.Cosmology(cosmo_params_dict={'H0':74.0}))
# Planck18 with ns=1
cosmos.append(cosmology.Cosmology(cosmo_params_dict={'ns':1.}))

# %% [markdown]
# Get parameters changing the background

# %%
for cosmo in cosmos:
    print(cosmo.get_background_params())

# %% [markdown]
# Compute a few interesting things
#
# - sigma8
# - fsigma8
# - linear power spectrum

# %%
z = 2.33
for cosmo in cosmos:
    print(cosmo.get_sigma8(z), cosmo.get_growth_rate(z))


# %%
k_Mpc = np.logspace(-3, 1, 100)
for ii, cosmo in enumerate(cosmos):
    pk_Mpc = cosmo.get_linP_Mpc(z=z, k_Mpc=k_Mpc)
    if ii == 0:
        pk0_Mpc = pk_Mpc.copy()
    plt.plot(k_Mpc, pk_Mpc/pk0_Mpc-1, label="cosmo {}".format(ii))
plt.xlabel(r"$k [\mathrm{Mpc}^{-1}]$")
plt.ylabel(r"$\frac{P(k)}{P_0(k)}-1$")
plt.legend()

# %% [markdown]
# Change units

# %%
z = 2.33
lambda_AA = 4000.0
for cosmo in cosmos:
    print(cosmo.get_dkms_dMpc(z), cosmo.get_dAA_dMpc(z, lambda_AA))

# %% [markdown]
# Compute linP params

# %%
kp_Mpc = 0.7
for cosmo in cosmos:
    print(cosmo.get_linP_Mpc_params(z=z,kp_Mpc=kp_Mpc))

# %% [markdown]
# ### Test RescaledCosmology

# %%
test = rescale_cosmology.RescaledCosmology(fid_cosmo=cosmos[0], new_params_dict={'ns':0.96})

# %%
try:
    bad_test = rescale_cosmology.RescaledCosmology(fid_cosmo=cosmos[0], new_params_dict={'H0':74})
except AssertionError as error:
    print(error)

# %%
print(test.get_dkms_dMpc(z), test.get_dAA_dMpc(z, lambda_AA))

# %%
print(test.get_linP_Mpc_params(z=z,kp_Mpc=kp_Mpc))

# %% [markdown]
# ### Check out Transfer function and expansion history

# %%
cosmos = []
# Planck18
cosmos.append(cosmology.Cosmology(cosmo_label="Planck18"))
params = cosmos[0].get_background_params()

# Planck18 with H0=74 while holding fixed Omh2
hnew = 0.74
cosmos.append(cosmology.Cosmology(cosmo_params_dict={'H0':hnew * 100}))

# Planck18 changing Omh2 while holding fixed H0
omch2new = params['omch2'] + 0.086/100
cosmos.append(cosmology.Cosmology(cosmo_params_dict={'omch2':omch2new}))


# %%
for ii in range(len(cosmos)):
    print(cosmos[ii].get_background_params())
    # compute CAMB object
    cosmos[ii].get_CAMBdata()

# %% [markdown]
# Plot transfer functions

# %%
plot_ratio = True

zz = np.array(cosmos[0].CAMBdata.transfer_redshifts)

ls = ["-", "--", "-.", ":"]
for kk, zuse in enumerate([2.2, 10.]):
    ii = np.argmin(np.abs(zz-zuse))
    zuse = zz[ii]

    for jj, cosmo in enumerate(cosmos):
        trans = cosmo.CAMBdata.get_matter_transfer_data()
        kh = trans.transfer_data[0, :, 0]
        k = kh * cosmo.CAMBdata.Params.h
        delta = trans.transfer_data[model.Transfer_cdm - 1, :, ii]
        if jj == 0:
            delta0 = delta.copy()
            k0 = k.copy()
            

        if plot_ratio:
            if jj != 0:
                yy = np.exp(np.interp(np.log(k0), np.log(k), np.log(delta)))
                plt.plot(k0, yy/delta0, label="Cosmo {}, z={}".format(jj, np.round(zuse,2)), ls=ls[jj], color="C"+str(kk))
        else:
            plt.plot(k, delta, label="Cosmo {}, z={}".format(jj, np.round(zuse,2)), ls=ls[jj], color="C"+str(kk))
    

plt.xscale("log")
plt.xlim([1e-3, 1])
plt.legend()

# %% [markdown]
# Plot expansion history

# %%
plot_ratio = True

zz = np.linspace(2, 10, 100)
for jj, cosmo in enumerate(cosmos):
    hz = cosmo.compute_hubble_parameter(zz)
    if jj == 0:
        hz0 = hz.copy()

    if plot_ratio:
        plt.plot(zz, hz/hz0, label="Cosmo {}".format(jj))
    else:
        plt.plot(zz, hz, label="Cosmo {}".format(jj))
plt.legend()

# %%
