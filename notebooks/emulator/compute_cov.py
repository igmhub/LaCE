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
# # Compute the LaCE emulator covariance

# %%
# %load_ext autoreload
# %autoreload 2

from pathlib import Path

import numpy as np

import lace
from lace.archive import gadget_archive, nyx_archive
from lace.emulator.covariance import data_for_l10_lace
from lace.emulator.gp_emulator_multi import GPEmulator
from lace.plotting import plot_l1o_correlation, plot_l1o_errors

# %% [markdown]
# ## Load the emulator and matching archive

# %%
emulator_label = "CH24_mpgcen_gpr"
# emulator_label = "CH24_nyxcen_gpr"

if emulator_label == "CH24_mpgcen_gpr":
    suite = "mpg"
    archive = gadget_archive.GadgetArchive()
elif emulator_label == "CH24_nyxcen_gpr":
    suite = "nyx"
    archive = nyx_archive.NyxArchive()
else:
    raise ValueError("Use CH24_mpgcen_gpr or CH24_nyxcen_gpr.")

emulator = GPEmulator(emulator_label=emulator_label)

# %% [markdown]
# ## Run the leave-one-out calculation

# %%
zz, k_Mpc, p1d_Mpc_orig, p1d_Mpc_sm, p1d_Mpc_emu, mask = data_for_l10_lace(
    archive,
    emulator_label,
    suite=suite,
)

rel_diff = p1d_Mpc_emu / p1d_Mpc_sm - 1
rel_diff[~mask] = 0
rel_diff[~np.isfinite(rel_diff)] = 0

rel_diff_zk = rel_diff.reshape(rel_diff.shape[0], -1)
rel_diff_k = rel_diff.reshape(-1, rel_diff.shape[-1])
cov_zk = np.cov(rel_diff_zk.T)
cov_k = np.cov(rel_diff_k.T)

# %% [markdown]
# ## Plot the covariance diagnostics

# %%
plot_l1o_correlation(cov_zk)

# %%
plot_l1o_errors(zz, k_Mpc, rel_diff, cov_zk)

# %% [markdown]
# ## Optionally store the covariance data

# %%
save_data = False

if save_data:
    output_path = (
        Path(lace.__file__).resolve().parents[1]
        / "data"
        / "covariance"
        / f"l1O_cov_{emulator_label}.npz"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        emulator_label=emulator_label,
        zz=zz,
        k_Mpc=k_Mpc,
        cov_k=cov_k,
        cov_zk=cov_zk,
        p1d_Mpc_orig=p1d_Mpc_orig,
        p1d_Mpc_sm=p1d_Mpc_sm,
        p1d_Mpc_emu=p1d_Mpc_emu,
        rel_diff=rel_diff,
        mask=mask,
    )
    print(f"Saved {output_path}")
