import os
import numpy as np
from lace.cosmo import camb_cosmo
import lace.cosmo.labeled_cosmologies as lab_cosmo


class Cosmology(object):
    """
    Class to interact with CAMB parameters, results, and others.
    """

    def __init__(self, cosmo_params_dict=None, cosmo_label=None, camb_kmax_Mpc=200.0):

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

        # store background parameters (dictionary)
        self.background_params = self.get_background_params()

        # call CAMB only later when needed
        self.CAMBdata = None
        self.linP_Mpc_interp = None
        self.camb_kmax_Mpc=camb_kmax_Mpc

        return


    def get_background_params(self):

        # collect parameters that would change the background expansion
        params = {}
        params["H0"] = self.CAMBparams.H0
        params["ombh2"] = self.CAMBparams.ombh2
        params["omch2"] = self.CAMBparams.omch2
        params["omk"] = self.CAMBparams.omk
        params["omnuh2"] = self.CAMBparams.omnuh2
        params["mnu"] = camb_cosmo.get_mnu(self.CAMBparams)
        params["w"] = self.CAMBparams.DarkEnergy.w
        params["wa"] = self.CAMBparams.DarkEnergy.wa

        return params


    def same_background(self, cosmo_params):
        """Check if any of the input cosmological parameters would change
        the background expansion of the cosmology"""

        # look for parameters that would change background
        back_params = self.get_background_params()
        print('back params', back_params)
        for name, value in back_params.items():
            if name in cosmo_params:
                if value != cosmo_params[name]:
                    print('background parameter differ', value, cosmo_params[name])
                    return False

        return True


    def print_info(self):
        """Print relevant parameters"""

        camb_cosmo.print_info(self.CAMBparams)
        return


    def _call_camb_results_background(self):
        """Call camb.get_results, but only for background expansion"""

        # zs = None to not compute linear power
        self.CAMBdata = camb_cosmo.get_camb_results(self.CAMBparams, zs=None)

        return


    def _call_camb_results_background_if_needed(self):
        """Call camb.get_results, but only for background expansion, if needed"""

        if self.CAMBdata is None:
            self._call_camb_results_background()

        return


    def _call_camb_results_full(self):
        """Call camb.get_results, the slowest function in CAMB calls."""

        zs = np.linspace(0, 10, 256)
        self.CAMBdata = camb_cosmo.get_camb_results(
            self.CAMBparams, zs=zs, camb_kmax_Mpc=self.camb_kmax_Mpc
        )
        self.linP_Mpc_interp = self.CAMBdata.get_matter_power_interpolator(
            nonlinear=False,
            var1=8,
            var2=8,
            hubble_units=False,
            k_hunit=False,
            log_interp=True
        )

        return


    def _call_camb_results_full_if_needed(self):
        """Check if linear power is already computed"""

        if self.linP_Mpc_interp is None:
            self._call_camb_results_full()

        return


    def get_linP_Mpc(self, z, k_Mpc):
        """Compute linear power (will call CAMB if needed)"""

        # make sure that you have set the interpolator
        self._call_camb_results_full_if_needed()

        if (z < self.linP_Mpc_interp.zmin) or (z > self.linP_Mpc_interp.zmax):
            raise ValueError(
                f"Requested z={z} is outside interpolation range [{self._linP_interp.zmin}, {self.linP_Mpc_interp.zmax}]"
            )
        elif k_Mpc.max() > self.linP_Mpc_interp.kmax:
            raise ValueError(
                f"Requested k_Mpc={k_Mpc.max()} exceeds interpolation range kmax_Mpc={self.linP_Mpc_interp.kmax}"
            )

        return self.linP_Mpc_interp.P(z, k_Mpc)


    def get_linP_hMpc(self, z, k_hMpc):
        """Compute linear power (will call CAMB if needed)"""

        h = self.CAMBparams.H0 / 100
        k_Mpc = k_hMpc * h
        pk_Mpc = self.get_linP_Mpc(z, k_Mpc)
        pk_hMpc = pk_Mpc * h**3

        return pk_hMpc


    def get_linP_kms(self, z, k_kms):
        """Compute linear power in velocity units (will call CAMB if needed)"""

        k_Mpc = k_kms * self.dkms_dMpc(z)
        pk_Mpc = self.get_linP_Mpc(z, k_Mpc)
        pk_kms = pk_Mpc * self.dkms_dMpc(z)**3

        return pk_kms


    def dkms_dMpc(self, z):
        """Compute factor to translate velocity separations (in km/s) to comoving
        separations (in Mpc). At z=3 it should return roughly 70."""

        # make sure that you have computed CAMB results
        self._call_camb_results_background_if_needed()

        return camb_cosmo.dkms_dMpc(self.CAMBparams, z, self.CAMBdata)


    def dkms_dhMpc(self, z):
        """Compute factor to translate velocity separations (in km/s) to comoving
        separations (in Mpc/h). At z=3 it should return roughly 100."""

        # make sure that you have computed CAMB results
        self._call_camb_results_background_if_needed()

        return camb_cosmo.dkms_dhMpc(self.CAMBparams, z, self.CAMBdata)


    def dAA_dMpc(self, z, lambda_AA):
        """Compute factor to translate wavelength separations (in AA) to comoving
        separations (in Mpc)."""

        # make sure that you have computed CAMB results
        self._call_camb_results_background_if_needed()

        return camb_cosmo.dAA_dMpc(self.CAMBparams, z, lambda_AA, self.CAMBdata)
