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
# # Reading hydro-simulation post-processing output
#
# This example reads one JSON file produced during hydro-simulation
# post-processing.  It shows the snapshot metadata, the available mean-flux
# rescalings, and the corresponding one-dimensional flux-power spectra.

# %%
import json
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

from lace.configuration import get_data_path

# %% [markdown]
# The default location is the bundled LaCE data directory.  If the simulation
# suite is stored elsewhere, configure it with `lace.configuration.set_data_path`
# or replace `postprocessing_file` with an explicit `Path`.

# %%
relative_path = Path(
    "sim_suites/post_768/sim_pair_0/sim_minus/"
    # "p1d_stau_0_Ns768_wM0.05_axis1.json"
    "p1d_reshaped_0_Ns768_wM0.05_axis1.json"
)
postprocessing_file = get_data_path() / relative_path

if not postprocessing_file.is_file():
    raise FileNotFoundError(
        "Post-processing file not found: "
        f"{postprocessing_file}. Configure the LaCE data path or provide "
        "an explicit path."
    )

print(postprocessing_file)

# %%
with postprocessing_file.open(encoding="utf-8") as file:
    postprocessing = json.load(file)

print("Top-level fields:", list(postprocessing))

# %% [markdown]
# `snapshot_data` records the simulation and snapshot settings used to
# construct the spectra.

# %%
pprint(postprocessing["snapshot_data"], sort_dicts=False)

# %%
print("Optical-depth rescalings:", postprocessing["scales_tau"])
print("Number of P1D records:", len(postprocessing["p1d_data"]))

summary = []
for record in postprocessing["p1d_data"]:
    summary.append(
        {
            "scale_tau": record["scale_tau"],
            "mean_flux": record["mF"],
            "n_k": len(record["k_Mpc"]),
            "k_min_Mpc": min(record["k_Mpc"]),
            "k_max_Mpc": max(record["k_Mpc"]),
            "skewer_file": record["sk_file"],
        }
    )

for row in summary:
    print(row)

# %% [markdown]
# Each record contains the 1D spectrum (`k_Mpc`, `p1d_Mpc`) and associated
# metadata.  The embedded `p3d_data` holds the corresponding binned 3D
# measurement.

# %%
first_record = postprocessing["p1d_data"][0]
print("Fields in one P1D record:", list(first_record))
print("First five k values [1/Mpc]:", np.asarray(first_record["k_Mpc"])[:5])
print("First five P1D values [Mpc]:", np.asarray(first_record["p1d_Mpc"])[:5])
print("Fields in its p3d_data:", list(first_record["p3d_data"]))

# %%
fig, ax = plt.subplots()
for record in postprocessing["p1d_data"]:
    k_mpc = np.asarray(record["k_Mpc"])
    p1d_mpc = np.asarray(record["p1d_Mpc"])
    positive_k = k_mpc > 0
    ax.loglog(
        k_mpc[positive_k],
        k_mpc[positive_k] * p1d_mpc[positive_k] / np.pi,
        label=f"scale_tau = {record['scale_tau']}",
    )

ax.set_xlabel(r"$k$ [Mpc$^{-1}$]")
ax.set_ylabel(r"$k P_{1D}(k) / \pi$")
ax.legend()
ax.grid(alpha=0.25)
plt.show()
