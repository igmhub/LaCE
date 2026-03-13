import os
import numpy as np
from lace.cosmo import base_cosmology, camb_cosmo
import lace.cosmo.labeled_cosmologies as lab_cosmo


class Cosmology(base_cosmology.BaseCosmology):
    """
    Class to interact with CAMB parameters, results, and others.
    """

    def __init__(
        self,
        cosmo_params_dict=None,
        cosmo_label=None,
        camb_kmax_Mpc=200.0,
        verbose=False,
    ):

        if verbose:
            print("inside Cosmology.__ini__")

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
        self.camb_kmax_Mpc = camb_kmax_Mpc

        # initialize BaseClass cosmo (should be a formality)
        super().__init__(verbose)

        return

    # overwrite virtual functions in base class

    def compute_hubble_parameter(self, z):
        """Return H(z) in units of km/s/Mpc"""

        # make sure that you have computed CAMB results
        self._call_camb_results_background_if_needed()
        return self.CAMBdata.hubble_parameter(z)

    def compute_angular_diameter_distance(self, z):
        """Return angular diameter distance (not comoving) in Mpc"""

        # make sure that you have computed CAMB results
        self._call_camb_results_background_if_needed()
        return self.CAMBdata.angular_diameter_distance(z)

    def compute_linP_Mpc(self, z, k_Mpc):
        """Return linear power at (z, k_Mpc) (will call CAMB if needed)"""

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

    def compute_growth_rate(self, z):
        """Return logarithmic growth rate (f) at z"""

        self._call_camb_results_full_if_needed()

        z_transfer = np.array(self.CAMBdata.transfer_redshifts)
        fsig8 = np.array(self.CAMBdata.get_fsigma8())
        sig8 = np.array(self.CAMBdata.get_sigma8())
        f = fsig8 / sig8
        # need to sort to interpolate
        ind_sort = np.argsort(z_transfer)

        return np.interp(z, z_transfer[ind_sort], f[ind_sort])

    def compute_sigma8(self, z):
        """Return sigma8 at z"""

        self._call_camb_results_full_if_needed()

        z_transfer = np.array(self.CAMBdata.transfer_redshifts)
        sig8 = np.array(self.CAMBdata.get_sigma8())
        # need to sort to interpolate
        ind_sort = np.argsort(z_transfer)

        return np.interp(z, z_transfer[ind_sort], sig8[ind_sort])

    # other functions specific to this class below

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

        if cosmo_params is None:
            return True

        # look for parameters that would change background
        back_params = self.get_background_params()
        if self.verbose:
            print("back params", back_params)
        for name, value in back_params.items():
            if name in cosmo_params:
                if value != cosmo_params[name]:
                    if self.verbose:
                        print("background parameter differ", value, cosmo_params[name])
                    return False

        return True

    def print_info(self):
        """Print relevant parameters"""

        camb_cosmo.print_info(self.CAMBparams)
        return

    def get_CAMBdata(self):
        """Set CAMBdata object"""
        self._call_camb_results_full_if_needed()
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
            log_interp=True,
        )

        return

    def _call_camb_results_full_if_needed(self):
        """Check if linear power is already computed"""

        if self.linP_Mpc_interp is None:
            self._call_camb_results_full()

        return
