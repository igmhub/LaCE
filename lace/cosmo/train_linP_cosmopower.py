import numpy as np
import pyDOE as pyDOE
from pathlib import Path
from tqdm import tqdm
from typing import List
from cosmopower import cosmopower_NN
import tensorflow as tf
import pandas as pd
import os
import camb

import logging
logger = logging.getLogger(__name__)
PROJ_ROOT = Path(__file__).resolve().parents[2]

def get_cosmology(
    H0=67.66,
    mnu=0.0,
    omch2=0.119,
    ombh2=0.0224,
    omk=0.0,
    As=2.105e-09,
    ns=0.9665,
    nrun=0.0,
    pivot_scalar=0.05,
    w=-1,
):
    """Given set of cosmological parameters, return CAMB cosmology object.

    Fiducial values for Planck 2018
    """

    pars = camb.CAMBparams()
    # set background cosmology
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, omk=omk, mnu=mnu)
    # set DE
    pars.set_dark_energy(w=w)
    # set primordial power
    pars.InitPower.set_params(
        As=As, ns=ns, nrun=nrun, pivot_scalar=pivot_scalar
    )

    return pars

def get_camb_results(pars: camb.CAMBparams, 
                     zs: list[float] | None = None, 
                     camb_kmax_Mpc: float = 30.0, 
                     camb_extra_kmax: float = 1.001, 
                     camb_fit_kmax_Mpc: float = 1.5) -> camb.CAMBdata:
    """Call camb.get_results, the slowest function in CAMB calls.
    - pars: CAMBparams object describing the cosmology
    - zs (optional): redshifts where we want to evaluate the linear power
    - camb_kmax_Mpc (optional): maximum k to compute power spectrum
    - camb_extra_kmax (optional): maximum k to compute power later on
    - camb_fit_kmax_Mpc (optional): maximum k to compute power later on"""

    if zs is not None:
        set_fast_camb_options(pars)
        kmax_Mpc = camb_fit_kmax_Mpc

        # camb_extra_kmax will allow to evaluate the power later on at kmax
        pars.set_matter_power(
            redshifts=zs,
            kmax=camb_extra_kmax * kmax_Mpc,
            nonlinear=False,
            silent=True,
        )
    return camb.get_results(pars)

def get_linP_Mpc(
    pars: camb.CAMBparams,
    zs: list[float],
    camb_results: camb.CAMBdata | None = None,
    camb_kmin_Mpc: float = 1.0e-4,
    camb_extra_kmax: float = 1.001,
    camb_npoints: int = 1000,
    fluid: int = 8,
) -> tuple[np.ndarray, list[float], np.ndarray]:
    """
    Given a CAMB cosmology and a set of redshifts, compute the linear power spectrum.
    
    Parameters:
    - pars: CAMB cosmology object
    - zs: list of redshifts
    - camb_results: pre-computed CAMB results (optional)
    - units: 'h/Mpc' or 'Mpc' (default: 'h/Mpc')
    - camb_kmin_Mpc: minimum k value in Mpc^-1 (default: 1.0e-4)
    - camb_extra_kmax: factor to extend kmax (default: 1.001)
    - camb_npoints: number of k points (default: 1000)
    - fluid: specify transfer function to use (default: 8 for CDM+baryons)
    
    Returns:
    - k: wavenumbers
    - zs: input redshifts
    - P: power spectrum values
    """
    
    camb_results = get_camb_results(pars, zs=zs)
    
    h = pars.H0 / 100.0
    kmin_hMpc = camb_kmin_Mpc / h
    kmax_hMpc = pars.Transfer.kmax / h / camb_extra_kmax
    
    kh, zs_out, Ph = camb_results.get_matter_power_spectrum(
        var1=fluid,
        var2=fluid,
        npoints=camb_npoints,
        minkh=kmin_hMpc,
        maxkh=kmax_hMpc,
    )
    
    assert all(z in zs_out for z in zs), "Requested redshifts not available in camb_results"
    
    P_out = np.array([Ph[zs_out.index(z)] for z in zs])
    
    k_Mpc = kh * h
    P_Mpc = P_out / h**3
    
    return k_Mpc, zs, P_Mpc

def set_fast_camb_options(pars: camb.CAMBparams) -> None:
    """Tune options in CAMB to speed-up evaluations"""

    fast_options = {
        'Want_CMB': False,
        'WantDerivedParameters': False,
        'WantCls': False,
        'Want_CMB_lensing': False,
        'DoLensing': False,
        'Want_cl_2D_array': False,
        'max_l_tensor': 0,
        'max_l': 2,
        'max_eta_k_tensor': 0,
    }

    reion_options = {
        'Reionization': False,
        'ReionizationModel': False,
        'include_helium_fullreion': False,
    }

    accuracy_options = {
        'AccuratePolarization': False,
        'AccurateReionization': False,
    }

    source_terms_options = {
        'counts_density': False,
        'limber_windows': False,
        'counts_timedelay': False,
        'counts_potential': False,
        'counts_ISW': False,
        'line_basic': False,
        'line_distortions': False,
        'use_21cm_mK': False,
    }

    for attr, value in fast_options.items():
        setattr(pars, attr, value)

    for attr, value in reion_options.items():
        setattr(pars.Reion, attr, value)

    for attr, value in accuracy_options.items():
        setattr(pars.Accuracy, attr, value)

    for attr, value in source_terms_options.items():
        setattr(pars.SourceTerms, attr, value)

def create_LH_sample(dict_params_ranges: dict,
                     nsamples: int=400000,
                     filename: str="LHS_params.npz"):
    
    params = {}
    for param, range_values in dict_params_ranges.items():
        params[param] = np.linspace(range_values[0], range_values[1], nsamples)

    AllParams = np.vstack(list(params.values()))
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
            #'nrun': AllCombinations[:, 6],
            }
    
    np.savez(PROJ_ROOT / 'data' / 'cosmopower_models' / filename, **params)
        
def generate_training_spectra(input_LH_filename: str,
                              output_filename: str = "linear.dat"):
    logger.info(f"Opening {PROJ_ROOT / 'data' / 'cosmopower_models' / input_LH_filename}")
    LH_params = np.load(PROJ_ROOT / 'data' / 'cosmopower_models' / input_LH_filename)
    logger.info(f"Generating {len(LH_params['H0'])} training spectra")

    ## Remove the existing linear.dat file if it exists
    if os.path.exists(PROJ_ROOT / 'data' / 'cosmopower_models' / output_filename):
        os.remove(PROJ_ROOT / 'data' / 'cosmopower_models' / output_filename)

    for ii in tqdm(range(len(LH_params["H0"]))):
        cosmo = get_cosmology(
            H0=LH_params["H0"][ii],
            mnu=LH_params["mnu"][ii],
            omch2=LH_params["omega_cdm"][ii],
            ombh2=LH_params["omega_b"][ii],
            omk=0,
            As=LH_params["As"][ii],
            ns=LH_params["ns"][ii],
            #nrun=LH_params["nrun"][ii]
                        )

        camb_results = get_camb_results(cosmo, 
                                        zs=[3])


        k_Mpc, _, linP_Mpc = get_linP_Mpc(
            pars=cosmo, 
            zs=[3], 
            camb_results=camb_results
        )

        params_lhs = np.array([
            LH_params[param][ii] for param in ["H0", "mnu", "omega_cdm", "omega_b", "As", "ns"] ])
        
        cosmo_array = np.hstack((params_lhs, linP_Mpc.flatten()))

        f=open(PROJ_ROOT / 'data' / 'cosmopower_models' / output_filename,'ab')
        np.savetxt(f, [cosmo_array])
        f.close()
    np.savetxt(PROJ_ROOT / "data" / "cosmopower_models" / "k_modes.txt", k_Mpc)

    return

def cosmopower_prepare_training(params : List = ["H0", "mnu", "omega_cdm", "omega_b", "As", "ns"],
                                Pk_filename: str = "linear.dat"):
    k_modes = np.loadtxt(PROJ_ROOT / "data" / "cosmopower_models" / "k_modes.txt")
    linear_spectra_and_params = np.loadtxt(PROJ_ROOT / "data" / "cosmopower_models" / Pk_filename)

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


def cosmopower_train_model(model_params: List = ["H0", "mnu", "omega_cdm", "omega_b", "As", "ns"],
                            model_save_filename: str = "Pk_cp_NN"):
    
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
                filename_saved_model=(PROJ_ROOT / 'data' / 'cosmopower_models' / model_save_filename).as_posix(),
                # cooling schedule
                validation_split=0.1,
                learning_rates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                batch_sizes=[1024, 1024, 1024, 1024, 1024],
                gradient_accumulation_steps = [1, 1, 1, 1, 1],
                # early stopping set up
                patience_values = [100,100,100,100,100],
                max_epochs = [1000,1000,1000,1000,1000]
                )

