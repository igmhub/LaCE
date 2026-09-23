# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: lace_test
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

# # COSMOPOWER vs CAMB COMPARISON

# +
import numpy as np
import cosmopower as cp
import pandas as pd
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from lace.archive import gadget_archive
from lace.cosmo.fit_linP_cosmopower import linPCosmologyCosmopower
from lace.emulator.constants import PROJ_ROOT

# -

# # LOAD DATA

test_sim = "mpg_central"
z_star=3


archive = gadget_archive.GadgetArchive(postproc="Cabayol23")

testing_data = archive.get_testing_data(sim_label=test_sim)

cosmo_params = testing_data[0]['cosmo_params']
cosmo_params["omnuh2"] = cosmo_params["mnu"] / 94.07
star_params = testing_data[0]['star_params']

# ### 1.2. P(K) WITH COSMOPOWER
#

# Load the emulator
emu_path = (PROJ_ROOT / "data" / "cosmopower_models" / "Pk_cp_NN_nrun").as_posix()
cp_nn = cp.cosmopower_NN(restore=True, 
                         restore_filename=emu_path)

# Access the emulator parameters and the __k__ modes
k_Mpc = cp_nn.modes
logger.info(f"Emulator parameters: {cp_nn.parameters}")

# Define the cosmology dictionary. Some parameters are used by the emulator (e.g. parameters from the previous cell) and others are used to convert to kms
cosmo_cp = {'H0': [cosmo_params["H0"]],
         'h': [cosmo_params["H0"]/100],
         'mnu': [cosmo_params["mnu"]],
         'Omega_m': [(cosmo_params["omch2"] + cosmo_params["ombh2"]) / (cosmo_params["H0"]/100)**2],
         'Omega_Lambda': [1- (cosmo_params["omch2"] + cosmo_params["ombh2"]) / (cosmo_params["H0"]/100)**2],
         'omch2': [cosmo_params["omch2"]],
         'ombh2': [cosmo_params["ombh2"]],
         'omnuh2': [cosmo_params["omnuh2"]],
         'As': [cosmo_params["As"]],
         'ns': [cosmo_params["ns"]],
         'nrun': [cosmo_params["nrun"]]}


# Emulate the power spectrum with cosmopower
Pk_Mpc_cp = cp_nn.ten_to_predictions_np(cosmo_cp)
k_Mpc_cp = cp_nn.modes.reshape(1,len(cp_nn.modes))

# Convert to kms
k_kms_cp, Pk_kms_cp = linPCosmologyCosmopower.convert_to_kms(cosmo_cp, k_Mpc_cp, Pk_Mpc_cp, z_star = z_star)

# ### 1.3. P(K) WITH CAMB
#

from lace.cosmo import camb_cosmo
from cup1d.likelihood import CAMB_model

cosmo_camb = camb_cosmo.get_cosmology(
    H0=cosmo_params["H0"],
    mnu=cosmo_params["mnu"],
    omch2=cosmo_params["omch2"],
    ombh2=cosmo_params["ombh2"],
    omk=cosmo_params["omk"],
    As=cosmo_params["As"],
    ns=cosmo_params["ns"],
    nrun=cosmo_params["nrun"]
)

fun_cosmo = CAMB_model.CAMBModel(
    zs= [3],
    cosmo=cosmo_camb,
    z_star=z_star,
    kp_kms=0.009,
)

k_Mpc_camb, zs, linP_Mpc_camb = fun_cosmo.get_linP_Mpc()

k_kms_camb, Pk_kms_camb = linPCosmologyCosmopower.convert_to_kms(cosmo_cp, k_Mpc_camb, linP_Mpc_camb, z_star = z_star)

# +
import matplotlib.pyplot as plt
import logging
import numpy as np

# Disable matplotlib debug logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'axes.titlesize': 16,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': [8, 8],
    'axes.prop_cycle': plt.cycler(color=['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
})

fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

# Main panel
ax1.set_title(f"Simulation: {test_sim}")
ax1.loglog(k_Mpc_camb, linP_Mpc_camb[0], label='CAMB')
ax1.loglog(k_Mpc_cp[0], Pk_Mpc_cp[0], label='CP', ls=':')
ax1.set_ylabel(r"$P(k)$ [Mpc]")
ax1.legend()

# Ratio panel
ratio = np.interp(k_Mpc_camb, k_Mpc_cp[0], Pk_Mpc_cp[0]) / linP_Mpc_camb[0]
ax2.semilogx(k_Mpc_camb, ratio-1)
ax2.set_ylabel('CP / CAMB - 1')
ax2.set_xlabel(r"$k$ [1/Mpc]")
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# +
import matplotlib.pyplot as plt
import logging
import numpy as np

# Disable matplotlib debug logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'axes.titlesize': 16,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': [8, 8],
    'axes.prop_cycle': plt.cycler(color=['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
})

fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

# Main panel
ax1.set_title(f"Simulation: {test_sim}")
ax1.loglog(k_kms_camb[0], Pk_kms_camb[0], label='CAMB')
ax1.loglog(k_kms_cp[0], Pk_kms_cp[0], label='CP', ls=':')
ax1.set_ylabel(r"$P(k)$ [kms]")
ax1.legend()

# Ratio panel
ratio = np.interp(k_kms_camb[0], k_kms_cp[0], Pk_kms_cp[0]) / Pk_kms_camb[0]
ax2.semilogx(k_kms_camb[0], ratio)
ax2.set_ylabel('CP / CAMB')
ax2.set_xlabel(r"$k$ [s/km]")
ax2.axhline(y=1, color='k', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
# -


