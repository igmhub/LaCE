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
cosmos = []
cosmos.append(cosmology.Cosmology())
cosmos.append(cosmology.Cosmology(cosmo_label="Planck18"))
cosmos.append(cosmology.Cosmology(cosmo_params_dict={'H0':74.0}))

# %%
for cosmo in cosmos:
    cosmo.get_background_params()

# %%
z = 2.33
lambda_AA = 4000.0
for cosmo in cosmos:
    print(cosmo.dkms_dMpc(z), cosmo.dAA_dMpc(z, lambda_AA))

# %%
for cosmo in cosmos:
    print(cosmo.dkms_dMpc(z=3.0))

# %%
zs = [2.0, 0.0]
k_Mpc = np.linspace(0.01, 1.0, 100)
for cosmo in cosmos:
    _=cosmo.get_linP_Mpc(zs, k_Mpc)

# %%
test = rescale_cosmology.RescaledCosmology(fid_cosmo=cosmos[0], new_params={'ns':0.96})

# %%
test = rescale_cosmology.RescaledCosmology(fid_cosmo=cosmos[0], new_params={'H0':74})

# %%

# %%
