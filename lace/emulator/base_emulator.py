import numpy as np
from abc import ABC, abstractmethod


class BaseEmulator(ABC):
    """Minimum amount of documentation"""

    @abstractmethod
    def emulate_p1d_Mpc(self, model, k_Mpc, return_covar=False, z=None):
        """
        Emulate P1D values for a given set of k values in Mpc units.

        Args:
            model (dict): Dictionary containing the model parameters.
            k_Mpc (np.ndarray): Wavenumbers in Mpc^-1 (legacy argument name).
            return_covar (bool, optional): Whether to return covariance. Defaults to False.
            z (float, optional): Redshift value. Defaults to None.

        Returns:
            np.ndarray: Emulated P1D values.
        """
        pass

    def emulate_P1D_Mpc(self, model, k_iMpc, return_covar=False, z=None):
        """Emulate P1D in Mpc using wavenumbers in inverse Mpc."""
        return self.emulate_p1d_Mpc(model, k_iMpc, return_covar, z)
