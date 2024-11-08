import numpy as np
import pyDOE as pyDOE
from pathlib import Path
from tqdm import tqdm
from typing import List
from cosmopower import cosmopower_NN
import tensorflow as tf


from lace.cosmo import camb_cosmo
from lace.emulator.constants import PROJ_ROOT
from cup1d.likelihood import CAMB_model


import logging
logger = logging.getLogger()

logger = logging.getLogger(__name__)

def create_LH_sample(dict_params_ranges: dict,
                     nsamples: int=400000):
    
    ombh2 =      np.linspace(dict_params_ranges['ombh2'][0], dict_params_ranges['ombh2'][1], nsamples)
    omch2 =     np.linspace(dict_params_ranges['omch2'][0], dict_params_ranges['omch2'][1], nsamples)
    H0 =         np.linspace(dict_params_ranges['H0'][0], dict_params_ranges['H0'][1], nsamples)
    ns =        np.linspace(dict_params_ranges['ns'][0], dict_params_ranges['ns'][1], nsamples)
    As =      np.linspace(dict_params_ranges['As'][0], dict_params_ranges['As'][1], nsamples)
    mnu =       np.linspace(dict_params_ranges['mnu'][0], dict_params_ranges['mnu'][1], nsamples)
    nrun =      np.linspace(dict_params_ranges['nrun'][0], dict_params_ranges['nrun'][1], nsamples)
    

    AllParams = np.vstack([ombh2, omch2, H0, ns, As, mnu, nrun])
    n_params = len(AllParams)

    lhd = pyDOE.lhs(n_params, samples=nsamples, criterion=None)
    idx = (lhd * nsamples).astype(int)

    AllCombinations = np.zeros((nsamples, n_params))
    for i in range(n_params):
        AllCombinations[:, i] = AllParams[i][idx[:, i]]

    params = {'omega_b': AllCombinations[:, 0],
            'omega_cdm': AllCombinations[:, 1],
            'H0': AllCombinations[:, 2],
            'ns': AllCombinations[:, 3],
            'As': AllCombinations[:, 4],
            'mnu': AllCombinations[:, 5],
            'nrun': AllCombinations[:, 6],
            }
    
    np.savez(PROJ_ROOT / 'data' / 'cosmopower_models' / 'LHS_params.npz', **params)
        
def generate_training_spectra(input_LH: Path):
    logger.info(f"Opening {input_LH}")
    LH_params = np.load(input_LH)
    logger.info(f"Generating {len(LH_params['H0'])} training spectra")

    for ii in tqdm(range(len(LH_params["H0"]))):
        cosmo = camb_cosmo.get_cosmology(
            H0=LH_params["H0"][ii],
            mnu=LH_params["mnu"][ii],
            omch2=LH_params["omega_cdm"][ii],
            ombh2=LH_params["omega_b"][ii],
            omk=0,
            As=LH_params["As"][ii],
            ns=LH_params["ns"][ii],
            nrun=LH_params["nrun"][ii]
        )

        fun_cosmo = CAMB_model.CAMBModel(
            zs=[3],
            cosmo=cosmo,
            z_star=3,
            kp_kms=0.009,
        )

        k_Mpc, _, linP_Mpc = fun_cosmo.get_linP_Mpc()

        params_lhs = np.array([
            LH_params[param][ii] for param in ["H0", "mnu", "omega_cdm", "omega_b", "As", "ns", "nrun"] ])
        #params_lhs = np.insert(params_lhs, 4, 0)  # Insert omk=0 at index 4

        cosmo_array = np.hstack((params_lhs, linP_Mpc.flatten()))

        f=open(PROJ_ROOT / 'data' / 'cosmopower_models' / 'linear.dat','ab')
        np.savetxt(f, [cosmo_array])
        f.close()
    np.savetxt(PROJ_ROOT / "data" / "cosmopower_models" / "k_modes.txt", k_Mpc)

    return

def cosmopower_prepare_training(params : List = ["H0", "mnu", "omega_cdm", "omega_b", "As", "ns", "nrun"]):
    k_modes = np.loadtxt(PROJ_ROOT / "data" / "cosmopower_models" / "k_modes.txt")
    linear_spectra_and_params = np.loadtxt(PROJ_ROOT / "data" / "cosmopower_models" / "linear.dat")


    n_params = len(params)

    # separate parameters from spectra, take log
    # Remove rows with NaN or infinite values from both arrays
    valid_rows = ~(np.isnan(linear_spectra_and_params).any(axis=1) | 
                  np.isinf(linear_spectra_and_params).any(axis=1))
    
    # Split into parameters and spectra
    linear_parameters = linear_spectra_and_params[valid_rows, :n_params]
    spectra = linear_spectra_and_params[valid_rows, n_params:]

    
    logger.info(f"Removing non-positive values before taking log")
    # Remove any non-positive values before taking log
    valid_spectra = (spectra > 0).all(axis=1)
    linear_parameters = linear_parameters[valid_spectra]
    linear_log_spectra = np.log10(spectra[valid_spectra])

    logger.info(f"Number of valid spectra: {len(linear_log_spectra)}")
    
    linear_parameters_dict = {params[i]: linear_parameters[:, i] for i in range(len(params))}
    linear_log_spectra_dict = {'modes': k_modes,
                            'features': linear_log_spectra}
    np.savez(PROJ_ROOT / "data" / "cosmopower_models" / "camb_linear_params.npz", **linear_parameters_dict)
    np.savez(PROJ_ROOT / "data" / "cosmopower_models" / "camb_linear_logpower.npz", **linear_log_spectra_dict)


def cosmopower_train_model(model_params: List = ["H0", "mnu", "omega_cdm", "omega_b", "As", "ns", "nrun"]):
    training_parameters = np.load(PROJ_ROOT / "data" / "cosmopower_models" / "camb_linear_params.npz")
    training_features = np.load(PROJ_ROOT / "data" / "cosmopower_models" / "camb_linear_logpower.npz")
    training_parameters = {model_params[i]: training_parameters[model_params[i]] for i in range(len(model_params))}
    training_log_spectra = training_features['features']
    device = "CPU"

    cp_nn = cosmopower_NN(parameters=model_params, 
                        modes=training_features['modes'], 
                        n_hidden = [512, 512, 512, 512], # 4 hidden layers, each with 512 nodes
                        verbose=True, # useful to understand the different steps in initialisation and training
                        )
    
    with tf.device(device):
        cp_nn.train(training_parameters=training_parameters,
                training_features=np.array(training_log_spectra),
                filename_saved_model=(PROJ_ROOT / 'data' / 'cosmopower_models' / 'Pk_cp_NN').as_posix(),
                # cooling schedule
                validation_split=0.1,
                learning_rates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                batch_sizes=[1024, 1024, 1024, 1024, 1024],
                gradient_accumulation_steps = [1, 1, 1, 1, 1],
                # early stopping set up
                patience_values = [100,100,100,100,100],
                #max_epochs = [1000,1000,1000,1000,1000],
                max_epochs = [1000,1000,1000,1000,1000]
                )
