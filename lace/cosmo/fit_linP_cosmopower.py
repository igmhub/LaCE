import cosmopower as cp
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJ_ROOT = Path(__file__).resolve().parents[2]

@dataclass
class linPCosmologyCosmopower:
    kp_kms : float = 0.009
    z_star : float = 3
    fit_min_kms : float = 0.5
    fit_max_kms : float = 2
    cosmopower_model : str = "Pk_cp_NN_sumnu"

    def __post_init__(self):
        self.cp_emulator = cp.cosmopower_NN(restore=True, 
                         restore_filename=PROJ_ROOT.as_posix()+f"/data/cosmopower_models/{self.cosmopower_model}")
        
        
    @staticmethod
    def fit_polynomial(xmin: float, 
                       xmax: float, 
                       x: np.ndarray, 
                       y: np.ndarray, 
                       deg: int = 2):
        """Fit a polynomial on the log of the function, within range"""
        x_fit = (x > xmin) & (x < xmax)
        poly = np.polyfit(np.log(x[x_fit]), np.log(y[x_fit]), deg=deg)
        return np.poly1d(poly)
    
    @staticmethod
    def measure_Hz(params: dict, 
                   zstar: float):
        Omega_bc = np.array(params["Omega_m"])
        Omega_lambda = np.array(params["Omega_Lambda"])
        #Omega_nu = 3*np.array(params["omega_ncdm"])
        H0 = np.array(params["h"])*100
        Omega_r = 1 - Omega_bc - Omega_lambda
        Hz = H0 * np.sqrt((Omega_bc*(1+zstar)**3 + Omega_r*(1+zstar)**4 + Omega_lambda))
        return Hz
    
    @staticmethod
    def convert_to_hMpc(Pk_Mpc: np.ndarray, 
                         k_Mpc: np.ndarray, 
                         h: np.ndarray):
        Pk_hMpc = Pk_Mpc / h[:,None]**3
        k_hMpc = k_Mpc * h[:,None]
        return Pk_hMpc, k_hMpc
    
    @staticmethod
    def convert_to_kms(params: dict, 
                        k_Mpc: np.ndarray, 
                        Pk_Mpc: np.ndarray, 
                        z_star: float):
        H_z = linPCosmologyCosmopower.measure_Hz(params, zstar = z_star)
        dvdX = H_z / (1 + z_star) 
        k_kms = k_Mpc / dvdX[:,None]
        Pk_kms = Pk_Mpc * (dvdX[:,None]**3)
        return k_kms, Pk_kms
    
    @staticmethod
    def get_star_params(linP_kms: np.poly1d, 
                        kp_kms: float):
        # translate the polynomial to our parameters
        ln_A_star = linP_kms[0]
        Delta2_star = np.exp(ln_A_star) * kp_kms**3 / (2 * np.pi**2)
        n_star = linP_kms[1]
        # note that the curvature is alpha/2
        alpha_star = 2.0 * linP_kms[2]

        results = {
            "Delta2_star": Delta2_star,
            "n_star": n_star,
            "alpha_star": alpha_star,
        }
        return results
    
    def fit_linP_cosmology(self, chains_df: pd.DataFrame, 
                           param_mapping: dict):   
        linP_cosmology_results = []
        # Calculate number of chunks needed
        chunk_size = 100_000
        n_chunks = len(chains_df) // chunk_size + (1 if len(chains_df) % chunk_size != 0 else 0)
        
        # Split DataFrame into chunks
        chunks = [chains_df.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(n_chunks)]
        logger.info(f"Fitting linear power to cosmology in {n_chunks} chunks")
        param_descriptions = {
            'h': 'Reduced Hubble parameter',
            'm_ncdm': 'Sum of neutrino masses',
            'omega_cdm': 'Total CDM density / h^2',
            'Omega_m': 'Total matter density',
            'ln_A_s_1e10': 'log(As/1e10)',
            'n_s': 'spectral index'
        }
        
        param_info = [f"{value}: ({param_descriptions.get(key, 'The emulator does not use this parameter.')})" for key, value in param_mapping.items()]
        logger.info(f"Using parameter mapping: {param_mapping}\n    " + "\n    ".join(param_info))

        for chunk in chunks:    # create a dict of cosmological parameters
            params = {
                'H0': [chunk[param_mapping['h']].values[ii]*100 for ii in range(len(chunk))],
                'h': [chunk[param_mapping['h']].values[ii] for ii in range(len(chunk))],
                'mnu': [3*chunk[param_mapping['m_ncdm']].values[ii] for ii in range(len(chunk))],
                'omega_cdm': [chunk[param_mapping['omega_cdm']].values[ii] for ii in range(len(chunk))],
                'omega_b': [(chunk[param_mapping['Omega_m']].values[ii] - chunk[param_mapping['omega_cdm']].values[ii]/chunk[param_mapping['h']].values[ii]**2) * chunk[param_mapping['h']].values[ii]**2 for ii in range(len(chunk))],
                'Omega_m': [chunk[param_mapping['Omega_m']].values[ii] for ii in range(len(chunk))],
                'Omega_Lambda': [chunk[param_mapping['Omega_Lambda']].values[ii] for ii in range(len(chunk))],
                'As': [np.exp(chunk[param_mapping['ln_A_s_1e10']].values[ii])/1e10 for ii in range(len(chunk))],
                'ns': [chunk[param_mapping['n_s']].values[ii] for ii in range(len(chunk))]
            }            
            Pk_Mpc = self.cp_emulator.ten_to_predictions_np(params)
            k_Mpc = self.cp_emulator.modes

            k_kms, Pk_kms = linPCosmologyCosmopower.convert_to_kms(params, k_Mpc, Pk_Mpc, z_star = self.z_star)

            kmin_kms = self.fit_min_kms * self.kp_kms
            kmax_kms = self.fit_max_kms * self.kp_kms  

            for ii in tqdm(range(len(chunk)), desc="Processing samples"):
                poly_linP = linPCosmologyCosmopower.fit_polynomial(
                                    xmin = kmin_kms / self.kp_kms, 
                                    xmax= kmax_kms / self.kp_kms, 
                                    x = k_kms[ii] / self.kp_kms, 
                                    y = Pk_kms[ii], 
                                    deg=2
                                    )
                
                results = linPCosmologyCosmopower.get_star_params(linP_kms = poly_linP, 
                                                                  kp_kms = self.kp_kms)

                linP_cosmology_results.append(results)

        return linP_cosmology_results


