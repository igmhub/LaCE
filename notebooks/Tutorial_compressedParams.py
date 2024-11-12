# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: lace
#     language: python
#     name: python3
# ---

# # TUTORIAL ON HOW TO ESTIMATE THE COMPRESSED PARAMETERS WITH COSMOPOWER
#
#

# ##### DESCLAIMER: Cosmopower is not installed by default in the LaCE package. You can install it as pip install cosmopower. 
#

# +
import numpy as np
import cosmopower as cp
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

# ## TRAIN COSMOPOWER
#

# ## 1. ESTIMATE THE COMPRESSED PARAMETERS OF THE GADGET TEST SIMULATIONS

# ### 1.1. LOAD THE GADGET TEST SIMULATION

test_sim = "mpg_neutrinos"
emulator_mnu = True

archive = gadget_archive.GadgetArchive(postproc="Cabayol23")

testing_data = archive.get_testing_data(sim_label=test_sim)

cosmo_params = testing_data[0]['cosmo_params']
star_params = testing_data[0]['star_params']

# ### 1.2. FIT THE COMPRESSED PARAMETERS WITH COSMOPOWER
#

# +
kp_Mpc = 0.7
z_star = 3
fit_min=0.5
fit_max=2

kmin_Mpc = fit_min * kp_Mpc
kmax_Mpc = fit_max * kp_Mpc
# -

emu_path = (PROJ_ROOT / "data" / "cosmopower_models" / "Pk_cp_NN").as_posix()
cp_nn = cp.cosmopower_NN(restore=True, 
                         restore_filename=emu_path)

k_Mpc = cp_nn.modes
logger.info(f"Emulator parameters: {cp_nn.parameters}")

# Define the cosmology dictionary
cosmo = {'H0': [cosmo_params["H0"]],
         'mnu': [cosmo_params["mnu"]],
         'omega_cdm': [cosmo_params["omch2"]],
         'omega_b': [cosmo_params["ombh2"]],
         'As': [cosmo_params["As"]],
         'ns': [cosmo_params["ns"]]}

# Compute the power spectrum
Pk_Mpc = cp_nn.ten_to_predictions_np(cosmo)
k_Mpc = cp_nn.modes.reshape(1,len(cp_nn.modes))

# Call the class to fit the compressed parameters
linP_Cosmology_Cosmopower = linPCosmologyCosmopower()


# Fit the power spectrum with a polynomial
linP_Mpc = linP_Cosmology_Cosmopower.fit_polynomial(
    xmin = kmin_Mpc / kp_Mpc, 
    xmax= kmax_Mpc / kp_Mpc, 
    x = k_Mpc / kp_Mpc, 
    y = Pk_Mpc, 
    deg=2
)

starparams_CP = linP_Cosmology_Cosmopower.get_star_params(linP = linP_Mpc, 
                                       kp = kp_Mpc)

logger.info(f"Star parameters of: {test_sim} with CP emulator with neutrino masses {emulator_mnu}")
logger.info(f"Percent relative error [%] on Delta2_star: {(starparams_CP['Delta2_star'] / star_params['Delta2_star'] - 1)*100:.2f}%")
logger.info(f"Percent relative error [%] on n_star: {(starparams_CP['n_star'] / star_params['n_star'] - 1)*100:.2f}%") 
logger.info(f"Percent relative error [%] on alpha_star: {(starparams_CP['alpha_star'] / star_params['alpha_star'] - 1)*100:.2f}%")

# ## 2. TRAIN COSMOPOWER MODELS

# ### 2.1. CREATE THE LATIN HYPERCUBE OF PARAMETERS
#
#

dict_params_ranges = {
    'ombh2': [0.015, 0.03],
    'omch2': [0.05, 0.3],
    'H0': [60, 80],
    'ns': [0.8, 1.2],
    'As': [5e-10, 4e-9],
    'mnu': [0, 0.5],}

create_LH_sample(dict_params_ranges = dict_params_ranges,
                     nsamples = 100,
                     filename = "LHS_params_test.npz")

# ### 2.2 GENERATE THE TRAINING SPECTRA

generate_training_spectra(input_LH_filename = 'LHS_params_test.npz',
                          output_filename = "linear_test.dat")

# ### 2.3 PREPARE THE TRAINING FILES FOR COSMOPOWER

cosmopower_prepare_training(params = ["H0", "mnu", "omega_cdm", "omega_b", "As", "ns"],
    Pk_filename = "linear_test.dat")

# ### 2.4 TRAIN THE COSMOPOWER MODEL
#

cosmopower_train_model(model_save_filename = "Pk_cp_NN_test")


