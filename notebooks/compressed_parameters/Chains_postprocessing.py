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

# # THIS NOTEBOOK PLOTS COSMOLOGICAL PARAMETERS FROM MCMC CHAINS

# %load_ext autoreload
# %autoreload 2

# +
import corner
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from lace.utils.plotting_functions import create_corner_plot
from lace.cosmo.fit_linP_cosmopower import linPCosmologyCosmopower

from cup1d.utils.utils import purge_chains

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
    

# ## DEFINE PATHS TO CHAINS TO BE PLOTTED

file_dirs = [
    ["/Users/lauracabayol/Documents/DESI/cup1d_chains/sampler_results_nyxcentral.npy","cup1d"]
    ]
    

# ## OPEN AND SAVE CHAINS IN DATAFRAMES

dfs = []
for file in file_dirs:
    df = open_Chain(file[0], file[1])    
    dfs.append(df)

# +
#The model expects the following parameter: h, m_ncdm, omch2, Omega_m, Omega_Lambda, ln_A_s_1e10, n_s, nrun.

#Use the following mapping to convert the parameter names in the dataframe to the expected parameter names.
#If you are using an emulator with other parameters, modify the dictionary and add only the parameters used by the model (e.g. if your emulator is **not** using neutrino masses, you should not include mnu in the dictionary).

#parameter expected by the module:parameter name in your dataframe  


param_mapping = {
    'h': 'h',
    'mnu': 'm_ncdm',
    'omch2': 'omch2',
    'Omega_m': 'Omega_m',
    'Omega_Lambda': 'Omega_Lambda',
    'ln_A_s_1e10': 'ln_A_s_1e10',
    'ns': 'ns',
    'nrun': 'nrun',
    'omnuh2': 'omnuh2'
}
# -

for ii, df in enumerate(dfs):
    linP_cosmology_results = fitter_compressed_params.fit_linP_cosmology(chains_df = df, 
                                                                     param_mapping = param_mapping)
    df_star_params = pd.DataFrame(linP_cosmology_results)
    df = pd.concat((df, df_star_params),axis=1)
    dfs[ii] = df

# ## PLOT COSMOLOGICAL PARAMETERS

# ### DEFINE PARAMETERS TO BE PLOTTED

#parameters_of_interest = ["sigma8", "Omega_m", "h", "omega_cdm"]
parameters_of_interest = ["As", "ns"]
#labels = [r"$\sigma_8$", r"$\Omega_m$", r"$h$",r"$\Omega_{\rm cdm} h^2$"]
labels = ["$A_s$", "$n_s$"]


# +
create_corner_plot(list_of_dfs = dfs, 
                   params_to_plot = parameters_of_interest,
                   labels = labels, 
                   colors = ['steelblue','crimson', 'goldenrod'], 
                  legend_labels = [r"LaCE-Nyx (Planck)", r"LaCE-Nyx (Nyx)"])



# +
create_corner_plot(list_of_dfs = dfs, 
                   params_to_plot = ["Delta2_star_cp", "n_star_cp"],
                   labels = [r"$\Delta^2_*$", r"$n_*$"], 
                   colors = ['steelblue','crimson', 'goldenrod'], 
                  legend_labels = [r"LaCE-Nyx (Planck)", r"LaCE-Nyx (Nyx)"])



# +
#checking that CP is deriving the same parameters as those in the chain.
ii = 0
parameters_of_interest = ["Delta2_star", "n_star"]#, "alpha_star"]
labels = [r"$\Delta^2_*$", r"$n_*$"]#, r"$\alpha_*$"]

df_star_params = dfs[ii][["Delta2_star_cp", "n_star_cp", "alpha_star_cp"]].rename(columns = {"Delta2_star_cp": "Delta2_star", "n_star_cp": "n_star", "alpha_star_cp": "alpha_star"})

create_corner_plot(list_of_dfs = [df_star_params, dfs[ii]], 
                   params_to_plot = parameters_of_interest,
                   labels = labels, 
                   colors = ['steelblue','crimson'], 
                   legend_labels = [r"CP", r"Scaling CAMB"])
# -




