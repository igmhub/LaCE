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
    print(cosmo.get_background_params())

# %%
z = 2.33
lambda_AA = 4000.0
for cosmo in cosmos:
    print(cosmo.get_dkms_dMpc(z), cosmo.get_dAA_dMpc(z, lambda_AA))

# %%
kp_Mpc = 0.7
for cosmo in cosmos:
    print(cosmo.get_linP_Mpc_params(z=z,kp_Mpc=kp_Mpc))

# %% [markdown]
# ### Test RescaledCosmology

# %%
test = rescale_cosmology.RescaledCosmology(fid_cosmo=cosmos[0], new_params_dict={'ns':0.96}, verbose=True)

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
