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

        # call CAMB to get the background results
        self.CAMBdata = self._call_camb_results_background()
        # for now this will be empty
        self.zs_camb_linP = None
        self._get_linP_Mpc = None

        return

    def print_info(self):
        """Print relevant parameters"""

        camb_cosmo.print_info(self.CAMBparams)
        return

    def _call_camb_results_background(self):
        """Call camb.get_results, the slowest function in CAMB calls.
        - zs (optional): redshifts where we want to evaluate the linear power
        """

        # set to None so these are not accidentally defined for other
        self.CAMBdata = camb_cosmo.get_camb_results(self.CAMBparams)

        return

    def _call_camb_results_full(self, zs=None, camb_kmax_Mpc=200.0):
        """Call camb.get_results, the slowest function in CAMB calls.
        - zs (optional): redshifts where we want to evaluate the linear power
        """

        if zs is None:
            zmin = 0
            zmax = 10
            nz = 256
            zs = np.linspace(zmin, zmax, nz)

        self.zs_camb_linP = zs
        self.CAMBdata = camb_cosmo.get_camb_results(
            self.CAMBparams, zs=self.zs_camb_linP, camb_kmax_Mpc=camb_kmax_Mpc
        )

        return

    def _call_camb_results_full_if_needed(self, zs=None, camb_kmax_Mpc=200.0):
        """Check if linear power at zs were computed"""

        if self.zs_camb_linP is None:
            self._call_camb_results_full(zs, camb_kmax_Mpc)
        else:
            # check if we already have all zs
            if not set(zs).issubset(self.zs_camb_linP):
                self._call_camb_results_full(zs, camb_kmax_Mpc)

        return

    def set_matter_power_interpolator(
        self,
        nonlinear=False,
        transfer_var1=8,
        transfer_var2=8,
        hubble_units=False,
        k_hunit=False,
        log_interp=True,
    ):
        """Sets an interpolator of the linear power spectrum from CAMB.

        See https://camb.readthedocs.io/en/latest/results.html

        Parameters:
            nonlinear (bool, optional): Whether to use the nonlinear power spectrum. Defaults to False.
            transfer_var1 (int, optional): Transfer function variable 1 for the power spectrum. Defaults to 8.
            transfer_var2 (int, optional): Transfer function variable 2 for the power spectrum. Defaults to 8.
            hubble_units (bool, optional): Whether to use Hubble units for the power spectrum. Defaults to False.
            k_hunit (bool, optional): Whether to use h/Mpc units for the wavenumber. Defaults to False.
            log_interp (bool, optional): Whether to use logarithmic interpolation for the power spectrum. Defaults to True.

        Returns:
            None: This method sets the linear power spectrum interpolator as an attribute of the class.
            It takes redshift (z) and wavenumber (k_Mpc) as inputs and returns the corresponding linear power spectrum.
        """

        self._call_camb_results_full_if_needed()

        self._linP_interp = self.CAMBdata.get_matter_power_interpolator(
            nonlinear=False,
            var1=transfer_var1,
            var2=transfer_var2,
            hubble_units=hubble_units,
            k_hunit=k_hunit,
            log_interp=log_interp,
        )

    def get_linP_Mpc(self, zs, k_Mpc):
        """Compute linear power (will call CAMB if needed)"""

        # make sure that you have set the interpolator
        if self._linP_interp is None:
            self.set_matter_power_interpolator()

        if (zs < self._linP_interp.zmin) or (zs > self._linP_interp.zmax):
            raise ValueError(
                f"Requested z={zs} is outside interpolation range [{self._linP_interp.zmin}, {self._linP_interp.zmax}]"
            )
        elif k_Mpc.max() > self._linP_interp.kmax:
            raise ValueError(
                f"Requested k_Mpc={k_Mpc.max()} exceeds interpolation range kmax_Mpc={self._linP_interp.kmax}"
            )

        return self._linP_interp.P(zs, k_Mpc)

    def get_linP_hMpc(self, zs, k_hMpc):
        """Compute linear power (will call CAMB if needed)"""

        if self.CAMBdata is None:
            self._call_camb_results()

        h = self.CAMBparams.H0 / 100
        k_Mpc = k_hMpc * h
        pk_Mpc = self.get_linP_Mpc(zs, k_Mpc)
        pk_hMpc = pk_Mpc * h**3

        return pk_hMpc

    def get_linP_kms(self, zs, k_kms):
        """Compute linear power in velocity units (will call CAMB if needed)"""

        # make sure that you have computed CAMB results
        self._call_camb_results_if_needed(zs)

        return self.get_linP_kms(self.CAMBparams, zs, self.CAMBdata, kmax_Mpc=kmax_Mpc)

    def dkms_dMpc(self, z):
        """Compute factor to translate velocity separations (in km/s) to comoving
        separations (in Mpc). At z=3 it should return roughly 70."""

        # make sure that you have computed CAMB results
        self._call_camb_results_if_needed()

        return camb_cosmo.dkms_dMpc(self.CAMBparams, z, self.CAMBdata)

    def dkms_dhMpc(self, z):
        """Compute factor to translate velocity separations (in km/s) to comoving
        separations (in Mpc/h). At z=3 it should return roughly 100."""

        # make sure that you have computed CAMB results
        self._call_camb_results_if_needed()

        return camb_cosmo.dkms_dhMpc(self.CAMBparams, z, self.CAMBdata)

    def dAA_dMpc(self, z, lambda_AA):
        """Compute factor to translate wavelength separations (in AA) to comoving
        separations (in Mpc)."""

        # make sure that you have computed CAMB results
        self._call_camb_results_if_needed()

        return camb_cosmo.dAA_dMpc(self.CAMBparams, z, lambda_AA, self.CAMBdata)
