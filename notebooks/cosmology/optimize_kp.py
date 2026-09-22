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
from lace.archive import gadget_archive
from lace.emulator.constants import PROJ_ROOT

from lace.cosmo import camb_cosmo
from cup1d.likelihood import CAMB_model

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
        df['nrun'] = 0
        df['ln_A_s_1e10'] = np.log(df.A_s*1e10)
        
    elif source_code=='cup1d':
        f = np.load(fname, allow_pickle=True)
        df = pd.DataFrame(f.tolist())    
        
        df = df[df.lnprob>-14]
        df['ln_A_s_1e10'] = np.log(df.As*1e10)
        df["Omega_m"] = 0.316329
        df["Omega_Lambda"] = 1 - df.Omega_m.values
        df["h"] = df.H0.values / 100 
        df["ombh2"] = 0.049009 * df.h**2
        df["omch2"] = (df.Omega_m - 0.049009) * df.h**2
        df['mnu'] = 0
        df["omnuh2"] = df.mnu /  94.07
        df['nrun'] = 0        
    return df
    


file_dirs = [["/Users/lauracabayol/Documents/DESI/cup1d_chains/chain.npy", "cup1d"]]

# ## OPEN AND SAVE CHAINS IN DATAFRAMES

dfs = []
for file in file_dirs:
    df = open_Chain(file[0], file[1])   
    #df = df.sample(100_000) 
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
    H0=0.6732117*100,
    mnu=0,
    omch2=0.12027891481623526,
    ombh2=0.02221156458376476,
    omk=0,
    As=2.1e-09,
    ns=0.966,
    nrun=0
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
for i, column in enumerate(columns):
    ax1.plot(keys_float, np.abs(column), label=f'{labels[i]}', marker='.', color=colors[i])

# Customize the first subplot
ax1.set_xlabel('kp [s/km]', fontsize=12)
ax1.set_ylabel('abs(Correlation)', fontsize=12)
ax1.legend()
ax1.grid(True)
ax1.set_title('Correlations between parameters', fontsize=14)

# Plot errors
error_keys = sorted(sigma_68_dict.keys(), key=lambda x: float(x))
error_values = [sigma_68_dict[k] for k in error_keys]
ax2.plot([float(k) for k in error_keys], error_values, marker='.', color='green')

# Add sigma68 from chain as black cross
ax2.scatter(float(0.009), sigma_68_chain, marker='x', color='black', label='Chain', s=80)
ax2.legend()

# Customize the second subplot
ax2.set_xlabel('kp [s/km]', fontsize=12)
ax2.set_ylabel(r'$\sigma_{68}[\Delta^2_{\star, \rm CP}]$ / $\Delta^2_{\star, \rm CAMB}$', fontsize=12)
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

# ## PLOT ELIPLSES

parameters_of_interest = ["Delta2_star_cp", "n_star_cp", "alpha_star_cp"]
labels = [r"$\Delta^2_*$", r"$n_*$", r"$\alpha_*$"]

create_corner_plot(list_of_dfs = [df_dict["0.004"]], 
                   params_to_plot = parameters_of_interest,
                   labels = labels, 
                   colors = ['steelblue'], 
                  legend_labels = [r"$k_p=0.004$"])

create_corner_plot(list_of_dfs = [df_dict["0.009"]], 
                   params_to_plot = parameters_of_interest,
                   labels = labels, 
                   colors = ['steelblue'], 
                  legend_labels = [r"$k_p=0.009$"])

create_corner_plot(list_of_dfs = [df_dict["0.01"]], 
                   params_to_plot = parameters_of_interest,
                   labels = labels, 
                   colors = ['steelblue'], 
                  legend_labels = [r"$k_p=0.01$"])

# ## DELTA^2* PRECISION VS k*

# ### In the mpg central simulation

test_sim = "mpg_neutrinos"
archive = gadget_archive.GadgetArchive(postproc="Cabayol23")
testing_data = archive.get_testing_data(sim_label=test_sim)

z_star = 3
fit_min=0.5
fit_max=2

cosmo_params = testing_data[0]['cosmo_params']
cosmo_params["omnuh2"] = cosmo_params["mnu"] / 94.07
star_params = testing_data[0]['star_params']

# Load cosmopower emulator
emu_path = (PROJ_ROOT / "data" / "cosmopower_models" / "Pk_cp_NN_nrun").as_posix()
cp_nn = cp.cosmopower_NN(restore=True, 
                         restore_filename=emu_path)
logger.info(f"Emulator parameters: {cp_nn.parameters}")

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

# +
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


# -

# Emulate the power spectrum with cosmopower
Pk_Mpc_cp = cp_nn.ten_to_predictions_np(cosmo_cp)
k_Mpc_cp = cp_nn.modes.reshape(1,len(cp_nn.modes))
k_kms_cp, Pk_kms_cp = linPCosmologyCosmopower.convert_to_kms(cosmo_cp, k_Mpc_cp, Pk_Mpc_cp , z_star = z_star)


# +
kps = np.arange(0.003,0.011,0.001)
#kps = np.arange(0.008,0.015,0.001)

errors = {}
#kps = [0.003]

for ii, kp in enumerate(kps):
    
    kmin_kms = fit_min * kp
    kmax_kms = fit_max * kp  
    
    poly_linP = linPCosmologyCosmopower.fit_polynomial(
                        xmin = kmin_kms / kp, 
                        xmax= kmax_kms / kp, 
                        x = k_kms_cp / kp, 
                        y = Pk_kms_cp, 
                        deg=2
                        )
    
    results = linPCosmologyCosmopower.get_star_params(linP_kms = poly_linP, 
                                                        kp_kms = kp)
    
    fun_cosmo = CAMB_model.CAMBModel(
        zs= [3],
        cosmo=cosmo_camb,
        z_star=z_star,
        kp_kms=kp,
    )
    results_camb = fun_cosmo.get_linP_params()

    err_delta2 = np.abs(results["Delta2_star_cp"] - results_camb["Delta2_star"]) / results_camb["Delta2_star"] 

    errors[f"{np.round(kp,4)}"] = err_delta2
    
    

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
for i, column in enumerate(columns):
    ax1.plot(keys_float, np.abs(column), label=f'{labels[i]}', marker='o', color=colors[i])

# Customize the first subplot
ax1.set_xlabel('kp [s/km]', fontsize=12)
ax1.set_ylabel('abs(Correlation)', fontsize=12)
ax1.legend()
ax1.grid(True)
ax1.set_title('Correlations between parameters', fontsize=14)

# Plot errors
error_keys = sorted(errors.keys(), key=lambda x: float(x))
error_values = [errors[k] for k in error_keys]
ax2.plot(error_keys, error_values, marker='o', color='green')

# Customize the second subplot
ax2.set_xlabel('kp [s/km]', fontsize=12)
ax2.set_ylabel(r'$\Delta^2_{\star, \rm CP}\ /\ \Delta^2_{\star, \rm CAMB}$ -1', fontsize=12)
ax2.grid(True)
ax2.set_title('Relative Error in $\Delta^2_*$ for the MPG neutrinos simulation', fontsize=14)

plt.tight_layout()
plt.show()
# -

#
