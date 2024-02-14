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
#     display_name: emulators
#     language: python
#     name: emulators
# ---

# %% [markdown]
# # Load archive and plot P1D as a function of parameters

# %%
from lace.archive.gadget_archive import GadgetArchive
from lace.archive.nyx_archive import NyxArchive

# %%
from lace.utils.check_paramSpace import check_convex_hull

# %%
import os
import numpy as np

# %% [markdown]
# ### Load Gadget and Nyx archive

# %%
cabayol23_archive = GadgetArchive(postproc='Cabayol23')

# %%
os.environ["NYX_PATH"]='/pscratch/sd/l/lcabayol/P1D/Nyx_files/'
nyx_version = "Oct2023"
nyx_archive = NyxArchive(nyx_version=nyx_version, verbose=True)

# %% [markdown]
# ## Check simulation

# %%
sim_lab = 'mpg_23'
z = 2

test_sim = cabayol23_archive.get_testing_data(sim_label=sim_lab,emu_params=emu_params)
test_point = [d for d in test_sim if d['z'] == z][0]
test_params = {param: test_point[param] for param in emu_params}

# %%
check_convex_hull(cabayol23_archive, test_point =test_params, drop_sim= sim_lab)

# %%
sim_lab = 'nyx_2'
z = 4.6
emu_params=["Delta2_p", "n_p", "mF", "sigT_Mpc", "gamma", "kF_Mpc"]


test_sim = nyx_archive.get_testing_data(sim_label=sim_lab,emu_params=emu_params)
test_point = [d for d in test_sim if d['z'] == z][0]
test_params = {param: test_point[param] for param in emu_params}

# %%
check_convex_hull(nyx_archive, test_point =test_params, drop_sim= sim_lab)

# %% [markdown]
# ## Loop over the archive

# %%
emu_params=["Delta2_p", "n_p", "mF", "sigT_Mpc", "gamma", "kF_Mpc"]
for ss in range(14):
    sim_lab = f'nyx_{ss}'
    for z in np.arange(2,4.8,0.2):
        z = np.round(z,2)
  
        test_sim = nyx_archive.get_testing_data(sim_label=sim_lab,emu_params=emu_params)
        test_point = [d for d in test_sim if d['z'] == z]
        if len(test_point) == 0:
            print(f'Snapshot nyx_{ss} at z={z} does not exist')
            continue
        test_point = test_point[0]
        test_params = {param: test_point[param] for param in emu_params}

        print(f'sim nyx_{ss} at z={z}')    
        _ = check_convex_hull(nyx_archive, test_point =test_params, drop_sim= sim_lab, verbose=False)

            

# %%
test_sim[0]

# %%
