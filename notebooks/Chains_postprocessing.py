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
        df['nrun'] = 0
        df['ln_A_s_1e10'] = np.log(df.A_s*1e10)
        
    elif source_code=='cup1d':
        f = np.load(fname, allow_pickle=True)
        df = pd.DataFrame(f.tolist())    
        
        mask = df[df.lnprob>-14]
        df['ln_A_s_1e10'] = np.log(df.As*1e10)
        df["Omega_m"] = 0
        df["Omega_Lambda"] = 1 - df.Omega_m.values
        df["h"] = df.H0.values / 100 
        df["ombh2"] = 0.049009 / df.h**2
        df["omch2"] = (df.Omega_m - 0.049009) / df.h**2
        df['m_ncdm'] = 0
        df['nrun'] = 0        
    return df
    


# ## DEFINE PATHS TO CHAINS TO BE PLOTTED

file_dirs = [
    #["/pscratch/sd/l/lcabayol/P1D/mockChallengeLym1d/mock_challenge_v0/chains_fiducialMockChallenge/mock_v2.1.txt","lym1d"],
    #"/pscratch/sd/l/lcabayol/P1D/mockChallengeLym1d/mock_challenge_v0/chains_Omh2_0.132/mock_v2.1.txt",
    #"/pscratch/sd/l/lcabayol/P1D/mockChallengeLym1d/mock_challenge_v0/chains/mock_v2.1.txt"]
    #["/pscratch/sd/l/lcabayol/P1D/mockChallengeLym1d/mock_challenge_v0/chains_H067/mock_v2.1.txt","lym1d"],
    #["/pscratch/sd/l/lcabayol/P1D/mockChallengeLym1d/mock_challenge_v0/chains_H074/mock_v2.1.txt","lym1d"]]
    ["/pscratch/sd/l/lcabayol/P1D/cup1d_chains/chain.npy","cup1d"]]
    

# ## OPEN AND SAVE CHAINS IN DATAFRAMES

dfs = []
for file in file_dirs:
    df = open_Chain(file[0], file[1])    
    dfs.append(df)

fitter_compressed_params = linPCosmologyCosmopower(cosmopower_model = "Pk_cp_NN_nrun")

# +
#The model expects the following parameter: h, m_ncdm, omch2, Omega_m, Omega_Lambda, ln_A_s_1e10, n_s, nrun.

#Use the following mapping to convert the parameter names in the dataframe to the expected parameter names.
#If you are using an emulator with other parameters, modify the dictionary and add only the parameters used by the model (e.g. if your emulator is **not** using neutrino masses, you should not include mnu in the dictionary).

#parameter expected by the module:parameter name in your dataframe  


param_mapping = {
    'h': 'h',
    'm_ncdm': 'm_ncdm',
    'omch2': 'omch2',
    'Omega_m': 'Omega_m',
    'Omega_Lambda': 'Omega_Lambda',
    'ln_A_s_1e10': 'ln_A_s_1e10',
    'ns': 'ns',
    'nrun': 'nrun'
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


create_corner_plot(list_of_dfs = dfs, 
                   params_to_plot = parameters_of_interest,
                   labels = labels, 
                   colors = ['steelblue','crimson', 'goldenrod'], 
                   #legend_labels = [r"Fiducial mock challenge", "$H_0 = 67$ km s$^{−1}$ Mpc$^{−1}$",r"$H_0 = 74 $km s$^{−1}$ Mpc$^{−1}$"])
                  #legend_labels = [r"Fiducial mock challenge", r"$\Omega_{\rm cdm} h^2$=0.132", r"$\Omega_{\rm cdm} h^2$=0.1204"])
                  legend_labels = [r"cup1d chain"])

parameters_of_interest = ["Delta2_star_cp", "n_star_cp", "alpha_star_cp"]
labels = [r"$\Delta^2_*$", r"$n_*$", r"$\alpha_*$"]
true_vals = [0.36,-2.3, -0.21]

create_corner_plot(list_of_dfs = dfs, 
                   params_to_plot = parameters_of_interest,
                   labels = labels, 
                   colors = ['steelblue','crimson', 'goldenrod'], 
                   #legend_labels = [r"Fiducial mock challenge", "$H_0 = 67$ km s$^{−1}$ Mpc$^{−1}$",r"$H_0 = 74 $km s$^{−1}$ Mpc$^{−1}$"])
                  #legend_labels = [r"Fiducial mock challenge", r"$\Omega_{\rm cdm} h^2$=0.132", r"$\Omega_{\rm cdm} h^2$=0.1204"])
                  legend_labels = [r"cup1d chain"])
