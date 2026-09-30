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
# The default location is the bundled LaCE data directory. This comparison
# uses the original `p1d_stau` output and the corrected `p1d_reshaped_stau`
# output for the same training snapshot.  These files share τ-rescalings, so
# their P1D and P3D measurements can be compared directly.

# %%
relative_directory = Path("sim_suites/post_768/sim_pair_0/sim_minus")
postprocessing_files = {
    "original": get_data_path()
    / relative_directory
    / "p1d_stau_0_Ns768_wM0.05_axis1.json",
    "reshaped": get_data_path()
    / relative_directory
    / "p1d_reshaped_stau_0_Ns768_wM0.05_axis1.json",
}

for label, filename in postprocessing_files.items():
    if not filename.is_file():
        raise FileNotFoundError(
            f"{label} post-processing file not found: {filename}. "
            "Configure the LaCE data path or provide explicit paths."
        )
    print(f"{label:8s}: {filename}")

# %%
postprocessing = {}
for label, filename in postprocessing_files.items():
    with filename.open(encoding="utf-8") as file:
        postprocessing[label] = json.load(file)

for label, payload in postprocessing.items():
    print(f"{label:8s} top-level fields: {list(payload)}")
    print(
        f"{label:8s} τ rescalings: {payload['scales_tau']}; "
        f"P1D records: {len(payload['p1d_data'])}"
    )

# %% [markdown]
# `snapshot_data` records the simulation and snapshot settings used to
# construct the spectra. They should agree between the two files.

# %%
pprint(postprocessing["original"]["snapshot_data"], sort_dicts=False)
assert (
    postprocessing["original"]["snapshot_data"]
    == postprocessing["reshaped"]["snapshot_data"]
)

# %%
for label, payload in postprocessing.items():
    print(f"\n{label}")
    for record in payload["p1d_data"]:
        print(
            {
                "scale_tau": record["scale_tau"],
                "mean_flux": record["mF"],
                "n_k": len(record["k_Mpc"]),
                "k_min_Mpc": min(record["k_Mpc"]),
                "k_max_Mpc": max(record["k_Mpc"]),
                "skewer_file": record["sk_file"],
            }
        )

# %% [markdown]
# Select a τ-rescaling that occurs in both files. The original and reshaped
# `stau` products share 0.9 and 1.1; change this value to compare the other
# common rescaling.

# %%
scale_tau = 0.9


def record_at_scale_tau(payload, scale_tau):
    matches = [
        record
        for record in payload["p1d_data"]
        if np.isclose(record["scale_tau"], scale_tau)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one record at scale_tau={scale_tau}; "
            f"found {len(matches)}"
        )
    return matches[0]


records = {
    label: record_at_scale_tau(payload, scale_tau)
    for label, payload in postprocessing.items()
}
print("Comparing scale_tau =", scale_tau)
for label, record in records.items():
    print(f"{label:8s} mean flux = {record['mF']:.8f}")

# %% [markdown]
# Each selected record contains a P1D measurement (`k_Mpc`, `p1d_Mpc`) and an
# embedded binned P3D measurement (`p3d_data`). Both panels below therefore
# compare the two post-processing methods at identical simulation conditions.

# %%
fig, ax = plt.subplots(2, 1, figsize=(6, 4))
for label, record in records.items():
    k_Mpc = np.asarray(record["k_Mpc"])
    P1D_Mpc = np.asarray(record["p1d_Mpc"])
    if label == "original":
        P1D_Mpc_orig = P1D_Mpc.copy()
    positive = (k_Mpc > 0) & (P1D_Mpc > 0)
    ax[0].loglog(
        k_Mpc[positive],
        k_Mpc[positive] * P1D_Mpc[positive] / np.pi,
        label=label,
    )
ax[1].plot(
    k_Mpc[positive],
    P1D_Mpc[positive] / P1D_Mpc_orig[positive]-1,
)

ax[1].set(xscale="log", ylim=[-0.01, 0.01])
ax[1].set_xlabel(r"$k$ [Mpc$^{-1}$]")
ax[0].set_ylabel(r"$k P_{1D}(k) / \pi$")
ax[0].set_title(rf"P1D comparison, $\tau$ scale = {scale_tau}")
ax[0].legend()

# %%
fig, ax = plt.subplots(
    2, 1, figsize=(7, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
)
# Show representative transverse-angle bins; colours identify μ and line
# styles identify the post-processing method. The lower panel is reshaped /
# original - 1, directly analogous to the P1D comparison above.
mu_indices = (0, 4, 8, 12)
for mu_index in mu_indices:
    spectra = {}
    for label, record in records.items():
        P3D = record["p3d_data"]
        k_Mpc = np.asarray(P3D["k_Mpc"])[:, mu_index]
        mu = np.asarray(P3D["mu"])[:, mu_index]
        P3D_Mpc = np.asarray(P3D["p3d_Mpc"])[:, mu_index]
        spectra[label] = (k_Mpc, P3D_Mpc)
        positive = (k_Mpc > 0) & (P3D_Mpc > 0)
        ax[0].loglog(
            k_Mpc[positive],
            k_Mpc[positive] ** 3 * P3D_Mpc[positive] / (2 * np.pi**2),
            color=f"C{mu_index // 4}",
            linestyle="-" if label == "original" else "--",
            label=(
                rf"$\mu={mu[positive][0]:.2f}$, {label}"
                if np.any(positive)
                else None
            ),
        )

    k_Mpc, P3D_original_Mpc = spectra["original"]
    _, P3D_reshaped_Mpc = spectra["reshaped"]
    positive = (k_Mpc > 0) & (P3D_original_Mpc > 0)
    ax[1].plot(
        k_Mpc[positive],
        P3D_reshaped_Mpc[positive] / P3D_original_Mpc[positive] - 1,
        color=f"C{mu_index // 4}",
        label=rf"$\mu={mu[positive][0]:.2f}$" if np.any(positive) else None,
    )

ax[0].set_ylabel(r"$k^3 P_{3D}(k,\mu) / (2\pi^2)$")
ax[0].set_title(rf"P3D comparison, $\tau$ scale = {scale_tau}")
ax[0].legend(ncol=2, fontsize=9)
ax[0].grid(alpha=0.25)
ax[1].axhline(0.0, color="k", lw=0.8)
ax[1].set(xscale="log", ylim=[-0.05, 0.05], xlim=[0.07, 7])
ax[1].set_xlabel(r"$k$ [Mpc$^{-1}$]")
ax[1].set_ylabel("reshaped /\noriginal - 1")
ax[1].legend(ncol=2, fontsize=9)
ax[1].grid(alpha=0.25)

# %%
