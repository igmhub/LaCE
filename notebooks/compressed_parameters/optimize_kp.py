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

# # OPTIMIZATION OF k*

# %load_ext autoreload
# %autoreload 2

# +
import corner
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cosmopower as cp

from lace.utils.plotting_functions import create_corner_plot
from lace.cosmo.fit_linP_cosmopower import linPCosmologyCosmopower
from lace.emulator.constants import PROJ_ROOT

from lace.cosmo import camb_cosmo
from cup1d.likelihood import CAMB_model
from cup1d.utils.utils import purge_chains


import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams["font.family"] = "serif"


# -

def open_Chain(fname, source_code):
    #access parameters
    if source_code == "lym1d":
        with open(fname, 'r') as file:
            for line in file:
                if line.strip().startswith("#"):  
                    first_comment = line.strip()
                    break
            parameters_list = first_comment.lstrip("#").split()  
        
        chains = np.loadtxt(fname)
        df = pd.DataFrame(chains, columns = parameters_list)
        df['m_ncdm'] = 0
        df['m_ncdm'] = 3* df['m_ncdm'] #the parameter is the sum of the masses of the neutrinos
        df['nrun'] = 0
        df['ln_A_s_1e10'] = np.log(df.A_s*1e10)
        df['omnuh2'] = df.m_ncdm / 94.07

        
    elif source_code=='cup1d_old':


        data = np.load(fname, allow_pickle=True).item()
        keep, discard = purge_chains(data["posterior"]["lnprob"], abs_diff=2, nsplit=8)
        posterior_dict = data['posterior']
        param_not = ['param_names', 'param_names_latex','param_percen', 'param_mle', 'lnprob_mle']
        df_dict = {key: posterior_dict[key][:, keep].flatten() for key in posterior_dict.keys() if key not in param_not}
        df = pd.DataFrame(df_dict)

        df['ln_A_s_1e10'] = np.log(df.As*1e10)
        df["h"] = data["like"]["cosmo_fid_label"]["cosmo"]["H0"]/100
        df['mnu'] = data["like"]["cosmo_fid_label"]["cosmo"]["mnu"]
        df['omnuh2'] = df.mnu / 94.07
        df["ombh2"] = data["like"]["cosmo_fid_label"]["cosmo"]["ombh2"]
        df["omch2"] = data["like"]["cosmo_fid_label"]["cosmo"]["omch2"] 
        df["Omega_m"] = (df.omch2.values + df.ombh2.values + df.omnuh2.values) / df.h**2
        df["Omega_Lambda"] = 1 - df.Omega_m.values 
        df["As_fid"] = data["like"]["cosmo_fid_label"]["cosmo"]["As"]
        df["ns_fid"] = data["like"]["cosmo_fid_label"]["cosmo"]["ns"]
        try:
            df["nrun"] = df["nrun"]
        except:
            df["nrun"] = 0
    
    elif source_code == 'cup1d':
        data = np.load(fname, allow_pickle=True).item()
        sampling_params = data["fitter"]["chain_names"]  # to chain
        star_params = data["fitter"]["blobs_names"]  # to blob
        _chain = data["fitter"]["chain"].reshape(
            -1, data["fitter"]["chain"].shape[-1]
        )
        _blobs = data["fitter"]["blobs"].reshape(-1)
        if "nrun" in sampling_params:
            nstar = 3
        else:
            nstar = 2
        all_params = np.zeros((_chain.shape[0], _chain.shape[1] + nstar))
        all_params_names = []
        for ii in range(_chain.shape[-1]):
            prange = data["fitter"]["chain_from_cube"][sampling_params[ii]]
            # print(sampling_params[ii], prange)
            all_params[:, ii] = _chain[:, ii] * (prange[1] - prange[0]) + prange[0]
            all_params_names.append(sampling_params[ii])

        for ii in range(nstar):
            all_params[:, -nstar + ii] = _blobs[star_params[ii]]
            all_params_names.append(star_params[ii])

        df = pd.DataFrame(all_params, columns=all_params_names)
        h = data["like"]["cosmo_fid_label"]["cosmo"]["H0"] / 100
        omch2 = data["like"]["cosmo_fid_label"]["cosmo"]["omch2"]
        ombh2 = data["like"]["cosmo_fid_label"]["cosmo"]["ombh2"]
        mnu = data["like"]["cosmo_fid_label"]["cosmo"]["mnu"]
        As = data["like"]["cosmo_fid_label"]["cosmo"]["As"]
        ns = data["like"]["cosmo_fid_label"]["cosmo"]["ns"]
        # next two lines to be updated when using with neutrinos
        omnuh2 = mnu / 94.07  # this is more complicated, need CAMB or CLASS
        Omega_m = (omch2 + ombh2) / h**2  # should I include omnuh2 here?

        if "nrun" not in sampling_params:
            df["nrun"] = 0

        df["ln_A_s_1e10"] = np.log(df.As * 1e10)
        df["h"] = h
        df["m_ncdm"] = mnu
        df["omch2"] = omch2
        df["ombh2"] = ombh2
        df["omnuh2"] = omnuh2
        df["Omega_m"] = Omega_m
        df["Omega_Lambda"] = 1 - Omega_m
        df["ns_fid"] = ns
        df["As_fid"] = As
    return df
    

file_dirs = [["/Users/lauracabayol/Documents/DESI/cup1d_chains/sampler_results_ca.npy", "cup1d_old"]]

# ## OPEN AND SAVE CHAINS IN DATAFRAMES

dfs = []
for file in file_dirs:
    df = open_Chain(file[0], file[1])   
    dfs.append(df)

#

# DEFINES THE MAPPING BETWEEN PARAMETER NAMES AND INTERNAL NAMING
param_mapping = {
    'h': 'h',
    'mnu': 'mnu',
    'omch2': 'omch2',
    'Omega_m': 'Omega_m',
    'Omega_Lambda': 'Omega_Lambda',
    'ln_A_s_1e10': 'ln_A_s_1e10',
    'ns': 'ns',
    'omnuh2': 'omnuh2',
    'nrun': 'nrun'
}

# ## LOOP OVER Kp

# +
#kps = np.arange(0.003,0.02,0.001)
kps = np.arange(0.003,0.011,0.001)

parameters_of_interest = ["Delta2_star_cp", "n_star_cp", "alpha_star_cp"]
labels = [r"$\Delta^2_*$", r"$n_*$", r"$\alpha_*$"]


# +
# to estimate the true value of the compressed parameters
from lace.cosmo import camb_cosmo
from cup1d.likelihood import CAMB_model


z_star = 3
cosmo_camb = camb_cosmo.get_cosmology(
    H0=df.h[0]*100,
    mnu=df.mnu[0],
    omch2=df.omch2[0],
    ombh2=df.ombh2[0],
    omk=0,
    As=df.As_fid[0],
    ns=df.ns_fid[0],
    nrun=df.nrun[0]
)

# +
df_dict = {}
correlations = {}
sigma_68_cp = {}
for ii, kp in enumerate(kps):
    dict_ = {}
    fitter_compressed_params =linPCosmologyCosmopower(cosmopower_model = "Pk_cp_NN_nrun",
                                                        fit_min_kms = 0.5,
                                                        fit_max_kms = 2,
                                                        kp_kms = kp)
    linP_cosmology_results = fitter_compressed_params.fit_linP_cosmology(chains_df = df, 
                                                                 param_mapping = param_mapping)
    
    df_star_params = pd.DataFrame(linP_cosmology_results)
    
    corrs =np.corrcoef(df_star_params.values, rowvar=False)
    corr_delta_n = corrs[0,1]
    corr_delta_alpha = corrs[0,2]
    corr_n_alpha = corrs[1,2]
    
    correlations[f"{np.round(kp,4)}"] = np.c_[corr_delta_n,corr_delta_alpha,corr_n_alpha]
    df_dict[f"{np.round(kp,4)}"] = df_star_params

    fun_cosmo = CAMB_model.CAMBModel(
    zs= [3],
    cosmo=cosmo_camb,
        z_star=z_star,
        kp_kms=kp,
    )
    results_camb = fun_cosmo.get_linP_params()
    q16, q50, q84 = np.percentile(df_star_params["Delta2_star_cp"], [16, 50, 84])
    sigma_68 = 0.5*(q84 - q16)

    sigma_68_cp[f"{np.round(kp,4)}"] = sigma_68 / results_camb["Delta2_star"]
    
    if np.round(kp,4) == 0.009:
        q16, q50, q84 = np.percentile(df["Delta2_star"], [16, 50, 84])
        sigma_68_chain = 0.5*(q84 - q16) / results_camb["Delta2_star"]


    

# +
keys = sorted(correlations.keys(), key=lambda x: float(x))
keys_float = [float(k) for k in keys]

values = [correlations[k][0] for k in keys]
columns = np.array(values).T

labels = [r"corr($\Delta^2_* - n_*$)", r"corr($\Delta^2_* - \alpha_*$)", r"corr($\alpha_* - n_*$)"]
colors = ["crimson", "navy", "goldenrod"]

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 1]})

# Plot correlations
for i, column in enumerate(columns[:1]):

    ax1.plot(keys_float, np.abs(column), label=f'{labels[i]}', marker='.', color=colors[i])

# Customize the first subplot
ax1.set_xlabel('kp [s/km]', fontsize=12)
ax1.set_ylabel('abs(Correlation)', fontsize=12)
ax1.legend()
ax1.grid(True)
ax1.set_title('Correlations between parameters', fontsize=14)

# Plot errors
error_keys = sorted(sigma_68_cp.keys(), key=lambda x: float(x))
error_values = [sigma_68_cp[k] for k in error_keys]
ax2.plot([float(k) for k in error_keys], error_values, marker='.', color='green')

# Add sigma68 from chain as black cross
#ax2.scatter(float(0.009), sigma_68_chain, marker='x', color='black', label='Chain', s=80)
#ax2.legend()

# Customize the second subplot
ax2.set_xlabel('kp [s/km]', fontsize=12)
ax2.set_ylabel(r'$\sigma_{68}[\Delta^2_{\star, \rm CP}]$ / $\Delta^2_{\star, \rm truth}$', fontsize=12)
ax2.grid(True)

plt.tight_layout()
plt.show()
# -

parameters_of_interest = ["Delta2_star", "n_star"]
labels = [r"$\Delta^2_*$", r"$n_*$"]
create_corner_plot(list_of_dfs = [df_dict["0.009"].rename(columns = {"Delta2_star_cp":"Delta2_star",
                                                                     "n_star_cp":"n_star",
                                                                     "alpha_star_cp":"alpha_star"}),
                                df], 
                   params_to_plot = parameters_of_interest,
                   labels = labels, 
                   colors = ['steelblue', 'crimson'], 
                  legend_labels = [r"$k_p=0.009$, CP", r"$k_p=0.009$, CAMB"])

# ### PLOT ELIPLSES

parameters_of_interest = ["Delta2_star_cp", "n_star_cp", "alpha_star_cp"]
labels = [r"$\Delta^2_*$", r"$n_*$", r"$\alpha_*$"]

create_corner_plot(list_of_dfs = [df_dict["0.006"]], 
                   params_to_plot = parameters_of_interest,
                   labels = labels, 
                   colors = ['steelblue'], 
                  legend_labels = [r"$k_p=0.006$"])

# ## LOOP OVER REDSHIFT

df_dict = {}
correlations = {}
sigma_68_cp = {}
z_stars = np.arange(2.5,3.6,0.1)
for ii, z_star in enumerate(z_stars):
    dict_ = {}
    fitter_compressed_params =linPCosmologyCosmopower(cosmopower_model = "Pk_cp_NN_nrun",
                                                        fit_min_kms = 0.5,
                                                        fit_max_kms = 2,
                                                        kp_kms = 0.009,
                                                        z_star = z_star)
    linP_cosmology_results = fitter_compressed_params.fit_linP_cosmology(chains_df = df, 
                                                                 param_mapping = param_mapping)
    
    df_star_params = pd.DataFrame(linP_cosmology_results)
    
    corrs =np.corrcoef(df_star_params.values, rowvar=False)
    corr_delta_n = corrs[0,1]
    corr_delta_alpha = corrs[0,2]
    corr_n_alpha = corrs[1,2]
    
    correlations[f"{np.round(kp,4)}"] = np.c_[corr_delta_n,corr_delta_alpha,corr_n_alpha]
    df_dict[f"{np.round(kp,4)}"] = df_star_params

    fun_cosmo = CAMB_model.CAMBModel(
        zs= [z_star],
    cosmo=cosmo_camb,
        z_star=z_star,
        kp_kms=0.009,
    )
    results_camb = fun_cosmo.get_linP_params()
    q16, q50, q84 = np.percentile(df_star_params["Delta2_star_cp"], [16, 50, 84])
    sigma_68 = 0.5*(q84 - q16)

    sigma_68_cp[f"{np.round(z_star,1)}"] = sigma_68 / results_camb["Delta2_star"]


columns.shape

# +
keys = sorted(correlations.keys(), key=lambda x: float(x))
keys_float = [float(k) for k in keys]

values = [correlations[k][0] for k in keys]
columns = np.array(values).T

labels = [r"corr($\Delta^2_* - n_*$)", r"corr($\Delta^2_* - \alpha_*$)", r"corr($\alpha_* - n_*$)"]
colors = ["crimson", "navy", "goldenrod"]

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 1]})

# Plot correlations
for i, column in enumerate(columns[:1]):
    ax1.plot(z_stars, np.abs(column), label=f'{labels[i]}', marker='.', color=colors[i])

# Customize the first subplot
ax1.set_xlabel('z*', fontsize=12)
ax1.set_ylabel('abs(Correlation)', fontsize=12)
ax1.legend()
ax1.grid(True)
ax1.set_title('Correlations between parameters', fontsize=14)

# Plot errors
error_keys = sorted(sigma_68_cp.keys(), key=lambda x: float(x))
error_values = [sigma_68_cp[k] for k in error_keys]
ax2.plot([float(k) for k in error_keys], error_values, marker='.', color='green')

# Add sigma68 from chain as black cross
#ax2.scatter(float(0.009), sigma_68_chain, marker='x', color='black', label='Chain', s=80)
#ax2.legend()

# Customize the second subplot
ax2.set_xlabel('z_*', fontsize=12)
ax2.set_ylabel(r'$\sigma_{68}[\Delta^2_{\star, \rm CP}]$ / $\Delta^2_{\star, \rm truth}$', fontsize=12)
ax2.grid(True)

plt.tight_layout()
plt.show()
# -












