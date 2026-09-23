# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: lace
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Train LaCE emulators

# %%
# %load_ext autoreload
# %autoreload 2

from lace.archive import gadget_archive, nyx_archive
from lace.emulator.gp_emulator_multi import GPEmulator

# %% [markdown]
# ## Train the full emulator
#
# Select the MP-Gadget or Nyx emulator below.  The archive is selected to
# match the emulator label.

# %%
emulator_label = "CH24_mpgcen_gpr"
# emulator_label = "CH24_nyxcen_gpr"

if emulator_label == "CH24_mpgcen_gpr":
    archive = gadget_archive.GadgetArchive()
elif emulator_label == "CH24_nyxcen_gpr":
    archive = nyx_archive.NyxArchive()
else:
    raise ValueError(f"Unsupported emulator label: {emulator_label}")

emulator = GPEmulator(
    emulator_label=emulator_label,
    archive=archive,
    train=True,
    drop_sim=None,
)

# %% [markdown]
# ## Train leave-one-out (L1O) emulators
#
# This trains one emulator per simulation after removing that simulation from
# the training set.  Nyx training is limited to its first 14 simulations.

# %%
for ii, isim in enumerate(archive.list_sim_cube):
    if (ii >= 14) & ("nyx" in emulator_label):
        continue
    print(ii, isim)
    emulator = GPEmulator(
        emulator_label=emulator_label,
        archive=archive,
        train=True,
        drop_sim=isim,
    )
