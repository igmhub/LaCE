import cosmopower as cp
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from lace.emulator.constants import PROJ_ROOT


@dataclass
class linP_cosmology_cosmopower:
    chains_df : pd.DataFrame
    kp_kms : float = 0.01
    z_star : float = 0.0
    fit_min_Mpc : float = 0.1
    fit_max_Mpc : float = 3

    def __post_init__(self):
        self.cp_emulator = cp.cosmopower_NN(restore=True, 
                         restore_filename=PROJ_ROOT/"data/cosmopower_models/PKLIN_NN")
        
        self.kmin_kms = self.fit_min_Mpc * self.kp_kms
        self.kmax_kms = self.fit_max_Mpc * self.kp_kms

        self.chains_df = chains_df
        
    @staticmethod
    def fit_polynomial(xmin, xmax, x, y, deg=2):
        """Fit a polynomial on the log of the function, within range"""
        x_fit = (x > xmin) & (x < xmax)
        poly = np.polyfit(np.log(x[x_fit]), np.log(y[x_fit]), deg=deg)
        return np.poly1d(poly)
    
    @staticmethod
    def measure_Hz(params, zstar):
        Omega_m = np.array(params["Omega_m"])
        Omega_lambda = np.array(params["Omega_Lambda"])
        H0 = np.array(params["h"])*100
        Omega_r = 1 - Omega_m - Omega_lambda
        Hz = np.sqrt(H0**2 * (Omega_m*(1+zstar)**3 + Omega_r*(1+zstar)**4 + params["Omega_Lambda"]))
        return Hz
    
    @staticmethod
    def convert_to_hMpc(Pk_Mpc, k_Mpc, h):
        Pk_hMpc = Pk_Mpc / h[:,None]**3
        k_hMpc = k_Mpc * h[:,None]
        return Pk_hMpc, k_hMpc
    
    @staticmethod
    def convert_to_kms(params, k_hMpc, Pk_hMpc, zstar):
        H_z = linP_cosmology_cosmopower.measure_Hz(params, zstar = z_star)
        dvdX = H_z / (1 + z_star) / params['h']
        k_kms = k_hMpc / dvdX[:,None]
        Pk_kms = Pk_hMpc * dvdX[:,None]**3
        return k_kms, Pk_kms
    
    def fit_linP_cosmology(self):        
        linP_cosmology_results = []
        # Calculate number of chunks needed
        chunk_size = 100_000
        n_chunks = len(self.chains_df) // chunk_size + (1 if len(self.chains_df) % chunk_size != 0 else 0)
        
        # Split DataFrame into chunks
        chunks = [self.chains_df.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(n_chunks)]
        
        for chunk in chunks:    # # create a dict of cosmological parameters
            params = {'omega_b': [chunk.Omega_m.values[ii]  - chunk.omega_cdm.values[ii]/chunk.h.values[ii] ** 2 for ii in range(len(chunk))],
                    'omega_cdm': [chunk.omega_cdm.values[ii] for ii in range(len(chunk))],
                    'Omega_m': [chunk.Omega_m.values[ii] for ii in range(len(chunk))],
                    'Omega_Lambda': [chunk.Omega_Lambda.values[ii] for ii in range(len(chunk))],
                    'h': [chunk.h.values[ii] for ii in range(len(chunk))],
                    'n_s': [chunk.n_s.values[ii] for ii in range(len(chunk))],
                    'ln10^{10}A_s': [chunk.ln_A_s_1e10.values[ii] for ii in range(len(chunk))],
                    'z': [self.z_star for ii in range(N)]
                    }
            Pk_Mpc = self.cp_emulator.ten_to_predictions_np(params)
            k_Mpc = self.cp_emulator.modes

            Pk_hMpc, k_hMpc = linP_cosmology_cosmopower.convert_to_hMpc(Pk_Mpc, k_Mpc, params['h'])
            k_kms, Pk_kms = linP_cosmology_cosmopower.convert_to_kms(params, k_hMpc, Pk_hMpc, self.z_star)

            for ii in range(len(chunk)):
                poly_linP_kms = linP_cosmology_cosmopower.fit_polynomial(
                                    xmin = self.kmin_kms / self.kp_kms, 
                                    xmax= self.kmax_kms / self.kp_kms, 
                                    x = k_kms / self.kp_kms, 
                                    y = Pk_kms, 
                                    deg=5
                                    )
                # translate the polynomial to our parameters
                ln_A_star = poly_linP_kms[0]
                Delta2_star = np.exp(ln_A_star) * self.kp_kms**3 / (2 * np.pi**2)
                n_star = poly_linP_kms[1]
                # note that the curvature is alpha/2
                alpha_star = 2.0 * poly_linP_kms[2]

                results = {
                    "Delta2_star": Delta2_star,
                    "n_star": n_star,
                    "alpha_star": alpha_star,
                }
                linP_cosmology_results.append(results)

        return linP_cosmology_results


