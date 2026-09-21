# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Emulate P1D given a cosmological and IGM model

# %%
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2
import numpy as np
import matplotlib.pyplot as plt

# %%
# our modules
from lace.cosmo import camb_cosmo
from lace.cosmo import fit_linP
from lace.archive import gadget_archive
from lace.emulator import gp_emulator

# %% [markdown]
# ### Load LaCE emulator

# %%
# specify simulation suite and P1D mesurements
archive = gadget_archive.GadgetArchive(postproc='Pedersen21')

# %%
emu_params=['Delta2_p', 'n_p','mF', 'sigT_Mpc', 'gamma', 'kF_Mpc']
training_data=archive.get_training_data(emu_params=emu_params)
Na = len(training_data)
print('P1D archive contains {} entries'.format(Na))

# %%
# maximum wavenumber to use in emulator
emu_kmax_Mpc=8
# setup GPy emulator
emulator=gp_emulator.GPEmulator(archive=archive,emu_params=emu_params,kmax_Mpc=emu_kmax_Mpc)

# %% [markdown]
# ### Specify cosmological model
#
# cosmo object will wrap a CAMB results object, and offer useful functionality.

# %%
cosmo=camb_cosmo.get_cosmology(H0=67,ns=0.96)

# %% [markdown]
# ### Compute linear power parameters at the redshift of interest

# %%
z=3.0
test_params=fit_linP.get_linP_Mpc_zs(cosmo,zs=[z],kp_Mpc=archive.kp_Mpc)[0]
for key,value in test_params.items():
    print(key,'=',value)

# %% [markdown]
# ### Specify IGM parameters at the redshift
#
# We need to choose a value of mean flux (mF), thermal broadening scale (sigT_Mpc), TDR slope gamma and filtering length (kF_Mpc).
#
# We will choose values that are well sampled in the archive.

# %%
dz=0.1
zmask=[ (archive.data[i]['z']<z+dz) & (archive.data[i]['z']>z-dz) for i in range(Na)]

# %%
test_params['mF']=np.mean([ archive.data[i]['mF'] for i in range(Na) if zmask[i] ])
print('mean flux = {:.3f}'.format(test_params['mF']))
test_params['sigT_Mpc']=np.mean([ archive.data[i]['sigT_Mpc'] for i in range(Na) if zmask[i] ])
print('thermal broadening sig_T = {:.3f} Mpc'.format(test_params['sigT_Mpc']))
test_params['gamma']=np.mean([ archive.data[i]['gamma'] for i in range(Na) if zmask[i] ])
print('TDR slope gamma = {:.3f}'.format(test_params['gamma']))
test_params['kF_Mpc']=np.mean([ archive.data[i]['kF_Mpc'] for i in range(Na) if zmask[i] ])
print('Filtering length k_F = {:.3f} 1/Mpc'.format(test_params['kF_Mpc']))

# %% [markdown]
# ### Ask emulator to predict P1D

# %%
# specify wavenumbers to emulate (in velocity units)
k_kms=np.logspace(np.log10(0.002),np.log10(0.02),num=20)
# use test cosmology to translate to comoving units
dkms_dMpc=camb_cosmo.dkms_dMpc(cosmo,z)
print('1 Mpc = {:.2f} km/s at z = {}'.format(dkms_dMpc,z))
k_Mpc=k_kms*dkms_dMpc

# %%
# emulate P1D in comoving units
p1d_Mpc=emulator.emulate_p1d_Mpc(model=test_params,k_Mpc=k_Mpc)
# use test cosmology to translate back to velocity units
p1d_kms=p1d_Mpc*dkms_dMpc

# %%
plt.loglog(k_kms,k_kms*p1d_kms)
plt.xlabel('k [s/km]')
plt.ylabel('k P(k)')

# %%
