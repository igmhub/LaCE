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

# %%
