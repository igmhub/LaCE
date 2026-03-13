import numpy as np
import scipy.constants
from scipy.integrate import simpson

# speed of light in km/s
c_kms = scipy.constants.c / 1e3


class BaseCosmology(object):
    """
    Abstract cosmology class, with virtual methods that need to be overwritten
    by other cosmology classes inheriting this one.
    """

    def __init__(self, verbose=False):

        self.verbose = verbose

        if self.verbose:
            print("inside BaseCosmology.__ini__")

        return

    # four functions that other cosmo classes should implement

    def compute_linP_Mpc(self, z, k_Mpc):
        """Return linear power at (z, k_Mpc)"""
        raise NotImplementedError()

    def compute_hubble_parameter(self, z):
        """Return H(z) in units of km/s/Mpc"""
        raise NotImplementedError()

    def compute_angular_diameter_distance(self, z):
        """Return angular diameter distance (not comoving) in Mpc"""
        raise NotImplementedError()

    def compute_growth_rate(self, z):
        """Return growth rate at z"""
        raise NotImplementedError()

    def compute_sigma8(self, z):
        """Return sigma8 at z"""
        raise NotImplementedError()

    # below here, no need to overwrite

    def get_H0(self):
        """Hubble constant in km/s/Mpc"""

        return self.compute_hubble_parameter(z=0)

    def get_h(self):
        """H0 / 100 km/s/Mpc"""

        return self.get_H0() / 100.0

    def get_growth_rate(self, z):
        """Return logarithmic growth rate (f) at z"""
        return self.compute_growth_rate(z)

    def get_sigma8(self, z):
        """Return sigma8 at z"""
        return self.compute_sigma8(z)

    def get_linP_Mpc(self, z, k_Mpc):
        """Return linear density power spectrum at (z, k_Mpc)"""

        return self.compute_linP_Mpc(z, k_Mpc)

    def get_linP_hMpc(self, z, k_hMpc):
        """Return linear density power spectrum at (z, k_Mpc)"""

        h = self.get_h()
        k_Mpc = k_hMpc * h
        pk_Mpc = self.compute_linP_Mpc(z, k_Mpc)
        pk_hMpc = pk_Mpc * h**3

        return pk_hMpc

    def get_sigma8_cb(self, z):
        """Return sigma8 at z

        It computes the integral of the linear power spectrum of CDM+baryons
        over a top-hat filter of radius 8 Mpc/h. Note that the result of this
        function is different from the one expected for a cosmology with massive
        neutrinos. Please use get_sigma8 right now

        """

        def fft_top_hat(k_hMpc, R_hMpc=8.0):
            """Top-hat filter"""
            x = k_hMpc * R_hMpc
            win = np.zeros_like(k_hMpc)
            # CAMB implementation https://github.com/cmbant/CAMB/blob/master/fortran/results.f90
            _ = np.argwhere(x < 1e-2)[:, 0]
            win[_] = 1 - x[_] ** 2 / 10
            _ = np.argwhere(x >= 1e-2)[:, 0]
            win[_] = 3 / x[_] ** 3 * (np.sin(x[_]) - x[_] * np.cos(x[_]))
            return win

        k_hMpc = np.logspace(-4, 2, 1000)
        linP_hMpc = self.get_linP_hMpc(z, k_hMpc)
        integrand = (k_hMpc**3 * linP_hMpc / 2 / np.pi**2) * fft_top_hat(k_hMpc) ** 2
        return np.sqrt(simpson(integrand, x=np.log(k_hMpc)))

    def get_linP_kms(self, z, k_kms):
        """Return linear power in velocity units"""

        k_Mpc = k_kms * self.get_dkms_dMpc(z)
        pk_Mpc = self.get_linP_Mpc(z, k_Mpc)
        pk_kms = pk_Mpc * self.get_dkms_dMpc(z) ** 3

        return pk_kms

    def get_dkms_dMpc(self, z):
        """Factor to translate velocity separations (in km/s) to comoving
        separations (in Mpc). At z=3 it should return roughly 70."""

        H_z = self.compute_hubble_parameter(z)
        dvdX = H_z / (1 + z)
        return dvdX

    def get_dkms_dhMpc(self, z):
        """Factor to translate velocity separations (in km/s) to comoving
        separations (in Mpc/h). At z=3 it should return roughly 100."""

        dvdX_Mpc = self.get_dkms_dMpc(z)
        h = self.get_h()
        dvdX_hMpc = dvdX_Mpc / h
        return dvdX_hMpc

    def get_dAA_dMpc(self, z, lambda_rest_AA=1215.67):
        """Factor to translate wavelength separations (in AA) to comoving
        separations (in Mpc)."""

        dkms_dMpc = self.get_dkms_dMpc(z)
        dAA_dkms = (1.0 + z) * lambda_rest_AA / c_kms
        return dkms_dMpc * dAA_dkms

    def get_drad_dMpc(self, z):
        """Factor to translate angular separations (in radians) to comoving
        transverse separations (in Mpc)."""

        # this should be defined in proper Mpc (not comoving)
        ang_dist = self.compute_angular_diameter_distance(z)
        D_M = ang_dist * (1 + z)
        return 1.0 / D_M

    def get_ddeg_dMpc(self, z):
        """Factor to translate angular separations (in deg) to comoving
        transverse separations (in Mpc)."""

        drad_dMpc = self.get_drad_dMpc(z)
        return 180.0 / np.pi * drad_dMpc

    def get_darc_dMpc(self, z):
        """Factor to translate angular separations (in arcmin) to comoving
        transverse separations (in Mpc)."""

        drad_dMpc = self.get_drad_dMpc(z)
        return 180.0 / np.pi * 60.0 * drad_dMpc

    def get_linP_Mpc_params(self, z, kp_Mpc):
        """Parameters describing the linear power around kp_Mpc"""

        # specify wavenumber range to fit
        kmin_over_kp = 0.5
        kmax_over_kp = 2.0
        k_over_kp = np.logspace(np.log10(kmin_over_kp), np.log10(kmax_over_kp), 100)
        k_Mpc = kp_Mpc * k_over_kp

        # get power spectrum in this range
        linP_Mpc = self.get_linP_Mpc(z, k_Mpc)

        # fit a 2nd-order polynomial to the log power
        poly_fit = np.polyfit(np.log(k_over_kp), np.log(linP_Mpc), deg=2)
        linP_Mpc_poly = np.poly1d(poly_fit)

        # translate the polynomial to linP params
        ln_A_p = linP_Mpc_poly[0]
        Delta2_p = np.exp(ln_A_p) * kp_Mpc**3 / (2 * np.pi**2)
        n_p = linP_Mpc_poly[1]
        # note that the curvature is alpha/2
        alpha_p = 2.0 * linP_Mpc_poly[2]

        linP_params = {"Delta2_p": Delta2_p, "n_p": n_p, "alpha_p": alpha_p}

        return linP_params

    def get_linP_kms_params(self, z, kp_kms):
        """Parameters describing the linear power around kp_kms"""

        # translate the pivot point to Mpc
        kp_Mpc = kp_kms * self.get_dkms_dMpc(z)

        # get the parameters in Mpc
        linP_Mpc_params = self.get_linP_Mpc_params(z, kp_Mpc)

        # modify the names
        linP_kms_params = {
            "Delta2_star": linP_Mpc_params["Delta2_p"],
            "n_star": linP_Mpc_params["n_p"],
            "alpha_star": linP_Mpc_params["alpha_p"],
        }

        return linP_kms_params
