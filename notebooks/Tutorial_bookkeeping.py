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
# # Tutorial bookkeeping
#
# In this notebook, we explain how to use the LaCE bookkeeping

# %%
from lace.archive.gadget_archive import GadgetArchive
from lace.archive.nyx_archive import NyxArchive

# %% [markdown]
# ### Read full post-processing for each archive

# %% [markdown]
# We load all archives here, to avoid redoing these multiple times in the notebook. Then we test them one by one.

# %%
# %%time
pedersen21_archive = GadgetArchive(postproc='Pedersen21')

# %%
# %%time
cabayol23_archive = GadgetArchive(postproc='Cabayol23')

# %%
# %%time
# os.environ["NYX_PATH"] = # path to the Nyx files in your local computer
nyx_version = "Oct2023"
nyx_archive = NyxArchive(nyx_version=nyx_version, verbose=True)

# %% [markdown]
# ### Access training data

# %% [markdown]
# #### Pedersen21

# %%
archive = pedersen21_archive
# (hypercube + special pairs) * nphases * naxes * nscalings * nsnaps
# some special pairs have only 1 scaling
nelem = (30 + 1) * 2 * 1 * 3 * 11 + 6 * 2 * 1 * 1 * 11
print(len(archive.data), nelem)

emu_params = [
    "Delta2_p",
    "n_p",
    "mF",
    "sigT_Mpc",
    "gamma",
    "kF_Mpc",
]
archive_training = archive.get_training_data(emu_params)
# hypercube * nsnaps
nelem = 30 * 11
print(len(archive_training), nelem)

# %% [markdown]
# #### Cabayol23

# %%
archive = cabayol23_archive
# (hypercube + special pairs) * nphases * naxes * nscalings * nsnaps
nelem = (30 + 7) * 2 * 3 * 5 * 11
print(len(archive.data), nelem)

emu_params = [
    "Delta2_p",
    "n_p",
    "mF",
    "sigT_Mpc",
    "gamma",
    "kF_Mpc",
]
archive_training = archive.get_training_data(emu_params)
# hypercube * nphases * naxes * nscalings * nsnaps
n_av_phases = 30 * 1 * 3 * 5 * 11
n_av_axes   = 30 * 2 * 1 * 5 * 11
n_av_all    = 30 * 1 * 1 * 5 * 11
print(len(archive_training), n_av_phases + n_av_axes + n_av_all)

# %% [markdown]
# #### Nyx

# %%
archive = nyx_archive
print(len(archive.data))

# parameters that must be available for each training element
emu_params = [
    "Delta2_p",
    "n_p",
    "mF",
    "sigT_Mpc",
    "gamma",
    "kF_Mpc",
]

archive_training = archive.get_training_data(emu_params)
print(len(archive_training))

# %% [markdown]
# ### Leave-one-out dropping of sims (for first 3 sims)

# %% [markdown]
# #### Pedersen21

# %%
archive = pedersen21_archive
emu_params = [
    "Delta2_p",
    "n_p",
    "mF",
    "sigT_Mpc",
    "gamma",
    "kF_Mpc",
]
for ii in range(2):
    archive_training = archive.get_training_data(emu_params, drop_sim='mpg_'+str(ii))
    # hypercube * nphases * naxes * nscalings * nsnaps
    nelem = (30-1) * 1 * 1 * 1 * 11
    print(len(archive_training), nelem)
    
    archive_testing = archive.get_testing_data('mpg_'+str(ii))
    nelem = 1 * 1 * 1 * 1 * 11
    print(len(archive_testing), nelem)

# %% [markdown]
# #### Multiple sims

# %%
archive_training = archive.get_training_data(emu_params, drop_sim=['mpg_0', 'mpg_1'])
# hypercube * nphases * naxes * nscalings * nsnaps
nelem = (30-2) * 1 * 1 * 1 * 11
print(len(archive_training), nelem)

# %% [markdown]
# #### Or redshifts

# %%
archive_training = archive.get_training_data(emu_params, drop_z=[2., 3.])
# hypercube * nphases * naxes * nscalings * nsnaps
nelem = (30) * 1 * 1 * 1 * (11-2)
print(len(archive_training), nelem)

# %% [markdown]
# #### Or snapshots

# %%
archive_training = archive.get_training_data(
    emu_params, 
    drop_snap=["mpg_0_2.0", "mpg_1_2.25"]
)
# hypercube * nphases * naxes * nscalings * nsnaps
nelem = (30) * 1 * 1 * 1 * 11 - 2
print(len(archive_training), nelem)

# %% [markdown]
# #### Cabayol23

# %%
archive = cabayol23_archive
emu_params = [
    "Delta2_p",
    "n_p",
    "mF",
    "sigT_Mpc",
    "gamma",
    "kF_Mpc",
]
for ii in range(3):
    archive_training = archive.get_training_data(emu_params, drop_sim='mpg_'+str(ii))
    # hypercube * nphases * naxes * nscalings * nsnaps
    n_av_phases = (30-1) * 1 * 3 * 5 * 11
    n_av_axes   = (30-1) * 2 * 1 * 5 * 11
    n_av_all    = (30-1) * 1 * 1 * 5 * 11
    print(len(archive_training), n_av_phases + n_av_axes + n_av_all)
    
    archive_testing = archive.get_testing_data('mpg_'+str(ii))
    nelem = 1 * 1 * 1 * 1 * 11
    print(len(archive_testing), nelem)

# %% [markdown]
# #### Nyx

# %%
archive = nyx_archive
# parameters that must be available for each training element
emu_params = [
    "Delta2_p",
    "n_p",
    "mF",
    "sigT_Mpc",
    "gamma",
    "kF_Mpc",
]
for ii in range(3):
    archive_training = archive.get_training_data(
        emu_params=emu_params, 
        drop_sim='nyx_'+str(ii)
    )
    print(len(archive_training))
    
    archive_testing = archive.get_testing_data(
        'nyx_'+str(ii), 
        emu_params=emu_params
    )
    print(len(archive_testing))    

# %% [markdown]
# ### Read special simulations for testing

# %% [markdown]
# #### Cabayol23

# %%
archive = cabayol23_archive
for sim in archive.list_sim_test:
    archive_testing = archive.get_testing_data(sim)
    nelem = 1 * 1 * 1 * 1 * 11
    print(len(archive_testing), nelem, archive_testing[0]['sim_label'])

# %% [markdown]
# #### Pedersen21

# %%
archive = pedersen21_archive
for sim in archive.list_sim_test:
    archive_testing = archive.get_testing_data(sim)
    nelem = 1 * 1 * 1 * 1 * 11
    print(len(archive_testing), nelem, archive_testing[0]['sim_label'])

# %% [markdown]
# #### Nyx

# %%
archive = nyx_archive
for sim in archive.list_sim_test:
    archive_testing = archive.get_testing_data(sim)
    if archive_testing:
        print(archive_testing[0]['sim_label'], len(archive_testing))
    else:
        print('empty sim:',sim)

# %%
