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
# # Explore LaCE simulation archives
#
# This tutorial loads the Gadget and Nyx simulation archives, visualizes their
# emulator training domains, and compares their saved IGM histories. Reusable
# figures are implemented in `lace.plotting.ArchivePlotter`; this notebook only
# selects the data and interprets the resulting plots.

# %% [markdown]
# ## Imports and plotting API
#
# The archive classes load simulation entries. `ArchivePlotter` converts their
# scalar parameters into consistently labelled Matplotlib figures without
# writing files unless an explicit `save_path` is supplied.

# %%
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

import numpy as np

from lace.archive.gadget_archive import GadgetArchive
from lace.archive.nyx_archive import NyxArchive
from lace.configuration import get_nyx_path, get_path_repo
from lace.plotting import ArchivePlotter, PARAMETER_LABELS


# %% [markdown]
# ## Load the Gadget archive
#
# The Cabayol23 post-processing supplies the MPG simulations. We construct the
# archive and its plotter once and reuse both throughout the notebook.

# %%
gadget_archive = GadgetArchive(postproc="Cabayol23")
gadget_plotter = ArchivePlotter(gadget_archive)


# %% [markdown]
# ## Inspect simulations and construct the training sample
#
# Each archive simulation contributes snapshots and post-processing variants.
# The training sample combines the archive entries according to the archive's
# existing selection rules. These seven parameters span the emulator domain.

# %%
print("Test simulations:", gadget_archive.list_sim_test)
training_parameters = [
    "Delta2_p",
    "n_p",
    "alpha_p",
    "mF",
    "sigT_Mpc",
    "gamma",
    "kF_Mpc",
]
gadget_training_data = gadget_plotter.get_training_data(
    training_parameters, average="both"
)
print(f"Averaged training entries: {len(gadget_training_data)}")


# %% [markdown]
# ## Individual relationships in the selected training domain
#
# These projections show how the linear-power and IGM parameters populate the
# selected archive. Points are coloured by redshift, so redshift evolution can
# be distinguished from variation across simulations.

# %%
parameter_pairs = [
    ("n_p", "Delta2_p"),
    ("z", "f_p"),
    ("z", "mF"),
    ("T0", "gamma"),
    ("T0", "sigT_Mpc"),
    ("sigT_Mpc", "kF_Mpc"),
]
for x_parameter, y_parameter in parameter_pairs:
    gadget_plotter.plot_parameter_pair(
        x_parameter, y_parameter, data=gadget_training_data
    )


# %% [markdown]
# ## Adjacent projections through the full emulator domain
#
# The panels join neighbouring parameters in the emulator input order. They
# provide a compact view of correlations between linear power, mean flux,
# thermal broadening, the temperature-density slope, and filtering length.

# %%
parameter_labels = {
    parameter: PARAMETER_LABELS[parameter] for parameter in training_parameters
}
gadget_plotter.plot_parameter_sequence(
    training_parameters, data=gadget_training_data, labels=parameter_labels
)


# %% [markdown]
# ## Dependence of the one-dimensional power spectrum on IGM parameters
#
# Each figure uses training data averaged over phase and line-of-sight-axis
# repetitions. Those repetitions are useful for emulator training but obscure
# this diagnostic plot. The averaged spectra expose how mean flux, thermal
# broadening, the temperature-density slope, and filtering length change the
# scale dependence of $k_\parallel P_{\rm 1D}$. Only
# $0 < k_\parallel < 10\,\mathrm{Mpc}^{-1}$ is displayed.

# %%
for parameter in ["mF", "sigT_Mpc", "gamma", "kF_Mpc"]:
    gadget_plotter.plot_p1d_dependence(
        parameter, data=gadget_training_data, max_curves=None
    )


# %% [markdown]
# ## Load the Nyx archive for a training-domain comparison
#
# The comparison uses the same emulator parameters for both data sets. This
# makes differences in their coverage visible without changing the axes or the
# archive selection rules.

# %%
nyx_archive = NyxArchive()
nyx_plotter = ArchivePlotter(nyx_archive)
nyx_training_data = nyx_plotter.get_training_data(
    training_parameters, average="both"
)


# %% [markdown]
# ## Compare Gadget and Nyx emulator domains
#
# Each panel overlays the same adjacent parameter projection for both archives.
# The colours identify the simulation suite; they do not encode redshift in
# this figure.

# %%
gadget_plotter.compare_parameter_sequences(
    {"Gadget": gadget_training_data, "Nyx": nyx_training_data},
    training_parameters,
    parameter_labels=parameter_labels,
    colors=["salmon", "goldenrod"],
    alphas=[0.5, 0.3],
)


# %% [markdown]
# ## Load saved IGM histories
#
# The Gadget histories are versioned with the LaCE repository. Nyx histories
# are read from the configured Nyx directory. A zero entry in these files marks
# an unavailable quantity and is masked by the plotting routines.

# %%
gadget_history_path = (
    get_path_repo() / "data" / "sim_suites" / "Australia20" / "IGM_histories.npy"
)
nyx_history_path = get_nyx_path() / "IGM_histories.npy"
gadget_histories = np.load(gadget_history_path, allow_pickle=True).item()
nyx_histories = np.load(nyx_history_path, allow_pickle=True).item()


# %% [markdown]
# ## Plot Gadget IGM histories
#
# The figure shows redshift evolution of effective optical depth, the
# temperature-density slope, thermal broadening, and filtering length. The
# `nyx_central` and `mpg_reio` histories are highlighted when they are present;
# `nyx_14` is excluded to retain the original notebook selection.

# %%
ArchivePlotter.plot_igm_histories(
    gadget_histories,
    highlighted_simulations=["nyx_central", "mpg_reio"],
    excluded_simulations=["nyx_14"],
)


# %% [markdown]
# ## Compare Gadget and Nyx IGM histories
#
# This overlays the available histories from each suite. The same physical
# quantities and redshift axes are used in every panel, allowing direct visual
# comparison of the thermal and ionization histories.

# %%
ArchivePlotter.compare_igm_histories(
    {"Gadget": gadget_histories, "Nyx": nyx_histories},
    colors=["black", "red"],
    alphas=[0.2, 0.2],
    excluded_simulations=["nyx_14"],
)

# %%

# %%
