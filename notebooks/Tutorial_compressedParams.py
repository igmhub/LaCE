# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: cosmopower
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

# # TUTORIAL ON HOW TO ESTIMATE THE COMPRESSED PARAMETERS WITH COSMOPOWER
#
#

# + [markdown] vscode={"languageId": "plaintext"}
# ##### DESCLAIMER: Cosmopower is not installed by default in the LaCE package. You can install it as pip install cosmopower. It requires the explicit installation of the dependencies (see [installation instructions](https://igmhub.github.io/LaCE/installation/))
#

# + [markdown] vscode={"languageId": "plaintext"}
# Additional documentation on how to use cosmopower can be found [here](https://igmhub.github.io/LaCE/users/compressedParameters/)
#

# +
import numpy as np
import cosmopower as cp
import pandas as pd
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

from lace.archive import gadget_archive
from lace.cosmo.fit_linP_cosmopower import linPCosmologyCosmopower
from lace.emulator.constants import PROJ_ROOT
from lace.cosmo.train_linP_cosmopower import (create_LH_sample, 
                                              generate_training_spectra, 
                                              cosmopower_prepare_training, 
                                              cosmopower_train_model)

# -

# ## 1. ESTIMATE THE COMPRESSED PARAMETERS OF THE GADGET TEST SIMULATIONS

# ### 1.1. LOAD THE GADGET TEST SIMULATION

test_sim = "mpg_neutrino"
emulator_mnu = True

archive = gadget_archive.GadgetArchive(postproc="Cabayol23")

testing_data = archive.get_testing_data(sim_label=test_sim)

cosmo_params = testing_data[0]['cosmo_params']
star_params = testing_data[0]['star_params']

cosmo_params

# ### 1.2. FIT THE COMPRESSED PARAMETERS WITH COSMOPOWER
#

# +
# Define fitting parameters
kp_kms = 0.009
z_star = 3
fit_min=0.5
fit_max=2

kmin_kms = fit_min * kp_kms
kmax_kms = fit_max * kp_kms
# -

# Load the emulator
emu_path = (PROJ_ROOT / "data" / "cosmopower_models" / "Pk_cp_NN_nrun").as_posix()
cp_nn = cp.cosmopower_NN(restore=True, 
                         restore_filename=emu_path)

# Access the emulator parameters and the __k__ modes
k_Mpc = cp_nn.modes
logger.info(f"Emulator parameters: {cp_nn.parameters}")

# Define the cosmology dictionary. Some parameters are used by the emulator (e.g. parameters from the previous cell) and others are used to convert to kms
cosmo = {'H0': [cosmo_params["H0"]],
         'h': [cosmo_params["H0"]/100],
         'mnu': [cosmo_params["mnu"]],
         'Omega_m': [(cosmo_params["omch2"] + cosmo_params["ombh2"]) / (cosmo_params["H0"]/100)**2],
         'Omega_Lambda': [1- (cosmo_params["omch2"] + cosmo_params["ombh2"]) / (cosmo_params["H0"]/100)**2],
         'omch2': [cosmo_params["omch2"]],
         'ombh2': [cosmo_params["ombh2"]],
         'As': [cosmo_params["As"]],
         'ns': [cosmo_params["ns"]],
         'nrun': [cosmo_params["nrun"]]}


# Emulate the power spectrum with cosmopower
Pk_Mpc = cp_nn.ten_to_predictions_np(cosmo)
k_Mpc = cp_nn.modes.reshape(1,len(cp_nn.modes))

# +
# Call the class to fit the compressed parameters
linP_Cosmology_Cosmopower = linPCosmologyCosmopower()

# Convert to kms
k_kms, Pk_kms = linP_Cosmology_Cosmopower.convert_to_kms(cosmo, k_Mpc, Pk_Mpc, z_star = z_star)
# Fit the polynomial
linP_kms = linP_Cosmology_Cosmopower.fit_polynomial(
    xmin = kmin_kms / kp_kms, 
    xmax= kmax_kms / kp_kms, 
    x = k_kms / kp_kms, 
    y = Pk_kms, 
    deg=2
)

starparams_CP = linP_Cosmology_Cosmopower.get_star_params(linP_kms = linP_kms, 
                                                          kp_kms = kp_kms)
# -

logger.info(f"Star parameters of: {test_sim} with CP emulator with neutrino masses {emulator_mnu}")
logger.info(f"Percent relative error [%] on Delta2_star: {(starparams_CP['Delta2_star'] / star_params['Delta2_star'] - 1)*100:.2f}%")
logger.info(f"Percent relative error [%] on n_star: {(starparams_CP['n_star'] / star_params['n_star'] - 1)*100:.2f}%") 
logger.info(f"Percent relative error [%] on alpha_star: {(starparams_CP['alpha_star'] / star_params['alpha_star'] - 1)*100:.2f}%")

# ## 2. MEASURING COMPRESSED PARAMETERS FROM COSMOLOGICAL CHAINS

# The previous section shows how to fit the compressed parameters from a single cosmology. Let's see how to do it for a chain.

#open the chain and define the parameters used by the emulator and not present in the chain
df = pd.read_csv(PROJ_ROOT / "data/utils" / "mini_chain_test.csv")
df['nrun'] = 0

# We need to provide a dictionary that maps the parameter names expected by the emulator to the column names of the dataframe.
#
# These parameters must be in the dataframe. They are used either to emulate P(k) or to convert to kms.

# +
#The model expects the following parameter: h, m_ncdm, omch2, Omega_m, Omega_Lambda, ln_A_s_1e10, n_s, nrun.

#Use the following mapping to convert the parameter names in the dataframe to the expected parameter names.
#If you are using an emulator with other parameters, modify the dictionary and add only the parameters used by the model (e.g. if your emulator is **not** using neutrino masses, you should not include mnu in the dictionary).

#parameter expected by the module:parameter name in your dataframe  
param_mapping = {
    'h': 'h',
    'm_ncdm': 'm_ncdm',
    'omch2': 'omega_cdm',
    'Omega_m': 'Omega_m',
    'Omega_Lambda': 'Omega_Lambda',
    'ln_A_s_1e10': 'ln_A_s_1e10',
    'ns': 'n_s',
    'nrun': 'nrun'
}
# -

fitter_compressed_params = linPCosmologyCosmopower(cosmopower_model = "Pk_cp_NN_nrun")
linP_cosmology_results = fitter_compressed_params.fit_linP_cosmology(chains_df = df, 
                                                                     param_mapping = param_mapping)

linP_cosmology_results

# ## 3. TRAIN COSMOPOWER MODELS

# ### 2.1. CREATE THE LATIN HYPERCUBE OF PARAMETERS
#
#

#Define the parameters used by the emulator and the ranges
params = ["ombh2", "omch2", "H0", "ns", "As", "mnu", "nrun"]
ranges = [
    [0.015, 0.03],
    [0.05, 0.16], 
    [60, 80],
    [0.8, 1.2],
    [5e-10, 4e-9],
    [0, 2],
    [0, 0.05]
]
dict_params_ranges = dict(zip(params, ranges))

#Create the Latin Hypercube sample
create_LH_sample(dict_params_ranges = dict_params_ranges,
                     nsamples = 10,
                     filename = "LHS_params_test.npz")

# ### 2.2 GENERATE THE TRAINING SPECTRA

#Generate the training power spectra with CAMB
generate_training_spectra(input_LH_filename = 'LHS_params_test.npz',
                          output_filename = "linear_test.dat")

# ### 2.3 PREPARE THE TRAINING FILES FOR COSMOPOWER

#Prepare the training files for cosmopower. The input Pk file must be that generated in the previous step.
cosmopower_prepare_training(params = params,
                            Pk_filename = "linear_test.dat")

# ### 2.4 TRAIN THE COSMOPOWER MODEL
#

# Explain that it will always take the files generated in the previous step.

cosmopower_train_model(
                        model_save_filename = "Pk_cp_NN_test",
                        model_params = params
)

#
