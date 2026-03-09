# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: cupix
#     language: python
#     name: cupix
# ---

# %% [markdown]
# # Tutorial explaining how to use emulators

# %% [markdown]
# ## For normal users

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import numpy as np
from matplotlib import pyplot as plt
from lace.cosmo import cosmology, rescale_cosmology

# %%
verbose = True
cosmos = []
cosmos.append(cosmology.Cosmology(verbose=verbose))
cosmos.append(cosmology.Cosmology(cosmo_label="Planck18", verbose=verbose))
cosmos.append(cosmology.Cosmology(cosmo_params_dict={'H0':74.0}, verbose=verbose))

# %%
for cosmo in cosmos:
    cosmo.get_background_params()

# %%
z = 2.33
lambda_AA = 4000.0
for cosmo in cosmos:
    print(cosmo.get_dkms_dMpc(z), cosmo.get_dAA_dMpc(z, lambda_AA))

# %%
z = 2.0
k_Mpc = np.linspace(0.01, 1.0, 100)
for cosmo in cosmos:
    linP_Mpc=cosmo.get_linP_Mpc(z, k_Mpc)
    linP_hMpc=cosmo.get_linP_hMpc(z, k_Mpc/0.7)
    linP_kms=cosmo.get_linP_kms(z, k_Mpc/100)
    print(linP_Mpc[0], linP_hMpc[0], linP_kms[0])

# %%

# %%
test = rescale_cosmology.RescaledCosmology(fid_cosmo=cosmos[0], new_params_dict={'ns':0.96}, verbose=True)

# %%
try:
    bad_test = rescale_cosmology.RescaledCosmology(fid_cosmo=cosmos[0], new_params_dict={'H0':74})
except AssertionError as error:
    print(error)

# %%
print(test.dkms_dMpc(z), test.dAA_dMpc(z, lambda_AA))

# %%
test.get_linP_Mpc(z, k_Mpc)

# %%
