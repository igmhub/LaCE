# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: emulators2
#     language: python
#     name: emulators2
# ---

# # THIS NOTEBOOK PLOTS COSMOLOGICAL PARAMETERS FROM MCMC CHAINS

# %load_ext autoreload
# %autoreload 2

# +
import corner
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from lace.utils.plotting_functions import create_corner_plot


import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams["font.family"] = "serif"
# -

# ## DEFINE PATHS TO CHAINS TO BE PLOTTED

file_dirs = [
    "/pscratch/sd/l/lcabayol/P1D/mockChallengeLym1d/mock_challenge_v0/chains_H067/mock_v2.1.txt",
    "/pscratch/sd/l/lcabayol/P1D/mockChallengeLym1d/mock_challenge_v0/chains_H074/mock_v2.1.txt"]
    

# ## OPEN AND SAVE CHAINS IN DATAFRAMES

with open(file_dirs[0], 'r') as file:
    for line in file:
        if line.strip().startswith("#"):  # Check for a commented line
            first_comment = line.strip()
            break
    parameters_list = first_comment.lstrip("#").split()

dfs = []
for file in file_dirs:
    chains = np.loadtxt(file)
    df = pd.DataFrame(chains, columns = parameters_list)
    dfs.append(df)

# ## PLOT COSMOLOGICAL PARAMETERS

# ### DEFINE PARAMETERS TO BE PLOTTED

parameters_of_interest = ["sigma8", "Omega_m", "h"]#omega_cdm
labels = [r"$\sigma_8$", r"$\Omega_m$", r"$h$"]#\Omega_{\rm cdm} h^2

dirs_list = []
for df in dfs:
    dict_ = {}
    for param in parameters_of_interest:
        dict_[f"{param}"] = df.loc[:,f'{param}'].values
    dirs_list.append(dict_)

create_corner_plot(dirs_list[0], 
                   labels, 
                  additional_data_dicts = dirs_list[1:],
                  additional_colors = ['crimson'], 
                  legend_labels = [r"$H_0 = 67$ km s$^{−1}$ Mpc$^{−1}$",r"$H_0 = 74 $km s$^{−1}$ Mpc$^{−1}$"])
                  #legend_labels = [r"$\Omega_{\rm cdm} h^2$=0.117",r"$\Omega_{\rm cdm} h^2$=0.13"])

# ## COMPRESSED PARAMETERS WITH COSMOPOWER

from lace.cosmo.fit_linP_cosmopower import linPCosmologyCosmopower


fitter_compressed_params = linPCosmologyCosmopower()

# DEFINES THE MAPPING BETWEEN PARAMETER NAMES AND INTERNAL NAMING
param_mapping = {
    'h': 'h',
    'm_ncdm': 'm_ncdm',
    'omega_cdm': 'omega_cdm',
    'Omega_m': 'Omega_m',
    'ln_A_s_1e10': 'ln_A_s_1e10',
    'n_s': 'n_s'
}

for ii, df in enumerate(dfs):
    df['m_ncdm'] = np.zeros(shape = len(df))
    df['ln_A_s_1e10'] = np.log(df.A_s*1e10)
    linP_cosmology_results = fitter_compressed_params.fit_linP_cosmology(chains_df = df, 
                                                                     param_mapping = param_mapping)
    df_star_params = pd.DataFrame(linP_cosmology_results)
    df = pd.concat((df, df_star_params),axis=1)
    dfs[ii] = df

# ## PLOT COMPRESSED PARAMETERS

parameters_of_interest = ["Delta2_star", "n_star", "alpha_star"]
labels = [r"$\Delta^2_*$", r"$n_*$", r"$\alpha_*$"]
true_vals = [0.36,-2.3, -0.21]

dirs_list = []
for df in dfs:
    dict_ = {}
    for param in parameters_of_interest:
        dict_[f"{param}"] = df.loc[:,f'{param}'].values
    dirs_list.append(dict_)

create_corner_plot(dirs_list[0], 
                   labels, 
                  additional_data_dicts = dirs_list[1:],
                  additional_colors = ['crimson'],
                  truth_values = true_vals,
                  legend_labels = [r"$H_0 = 67$ km s$^{−1}$ Mpc$^{−1}$",r"$H_0 = 74$km s$^{−1}$ Mpc$^{−1}$"])
