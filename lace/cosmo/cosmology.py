import os
import numpy as np
from lace.cosmo import camb_cosmo
import lace.cosmo.labeled_cosmologies as lab_cosmo


class Cosmology(object):
    """
    Class to interact with CAMB parameters, results, and others. 
    """

    def __init__(self, cosmo_params_dict=None, cosmo_label=None):

        if (cosmo_params_dict is not None) and (cosmo_label is not None):
            raise ValueError("You cannot provide both cosmo params and label")

        # store these just in case (note this is really just the input)
        self.input_cosmo_label = cosmo_label
        self.input_cosmo_params_dict = cosmo_params_dict

        if cosmo_params_dict is None:
            # if no cosmo params provided, use label or Planck18
            if cosmo_label is None:
                cosmo_label = "Planck18"
            cosmo_params_dict = lab_cosmo.get_cosmo_params_dict_from_label(cosmo_label)

        # get CAMBparams object
        self.CAMBparams = camb_cosmo.get_cosmology_from_dictionary(cosmo_params_dict)

        # for now this will be empty
        self.CAMBdata = None
        self.zs_camb_linP = []

        return 


    def print_info(self):
        """Print relevant parameters"""

        camb_cosmo.print_info(self.CAMBparams)
        return


    def _call_camb_results(self, zs=None):
        """Call camb.get_results, the slowest function in CAMB calls.
            - zs (optional): redshifts where we want to evaluate the linear power
        """

        self.CAMBdata = camb_cosmo.get_camb_results(self.CAMBparams,zs)
        self.zs_camb_linP = zs

        return


    def _call_camb_results_if_needed(self, zs=None):
        """Check if linear power at zs were computed"""

        if self.CAMBdata is None:
            self._call_camb_results(zs)
        elif zs is not None:
            if self.zs_camb_linP is None:
                self._call_camb_results(zs)
            else:
                #check if we already have all zs
                if not set(zs).issubset(self.zs_camb_linP):
                    self._call_camb_results(zs)

        return


    def get_linP_hMpc(self, zs, kmax_Mpc=None):
        """Compute linear power (will call CAMB if needed)"""

        # make sure that you have computed CAMB results
        self._call_camb_results_if_needed(zs)

        return camb_cosmo.get_linP_hMpc(self.CAMBparams,
                                        zs,
                                        self.CAMBdata, 
                                        kmax_Mpc=kmax_Mpc)


    def get_linP_Mpc(self, zs, kmax_Mpc=None):
        """Compute linear power (will call CAMB if needed)"""

        # make sure that you have computed CAMB results
        self._call_camb_results_if_needed(zs)

        return self.get_linP_Mpc(self.CAMBparams,
                                 zs,
                                 self.CAMBdata,
                                 kmax_Mpc=kmax_Mpc)


    def get_linP_kms(self, zs, kmax_Mpc=None):
        """Compute linear power in velocity units (will call CAMB if needed)"""
    
        # make sure that you have computed CAMB results
        self._call_camb_results_if_needed(zs)

        return self.get_linP_kms(self.CAMBparams,
                                 zs,
                                 self.CAMBdata,
                                 kmax_Mpc=kmax_Mpc)


    def dkms_dMpc(self, z):
        """Compute factor to translate velocity separations (in km/s) to comoving
            separations (in Mpc). At z=3 it should return roughly 70. """

        # make sure that you have computed CAMB results
        self._call_camb_results_if_needed()

        return camb_cosmo.dkms_dMpc(self.CAMBparams, z, self.CAMBdata)

    
    def dkms_dhMpc(self, z):
        """Compute factor to translate velocity separations (in km/s) to comoving
            separations (in Mpc/h). At z=3 it should return roughly 100. """

        # make sure that you have computed CAMB results
        self._call_camb_results_if_needed()

        return camb_cosmo.dkms_dhMpc(self.CAMBparams, z, self.CAMBdata)


    def dAA_dMpc(self, z, lambda_AA):
        """Compute factor to translate wavelength separations (in AA) to comoving
            separations (in Mpc). """
            
        # make sure that you have computed CAMB results
        self._call_camb_results_if_needed()

        return camb_cosmo.dAA_dMpc(self.CAMBparams, z, lambda_AA, self.CAMBdata)


