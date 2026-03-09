import numpy as np
import scipy.constants 

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
            print('inside BaseCosmology.__ini__')

        return


    # three functions that other cosmo classes should implement

    def compute_linP_Mpc(self, z, k_Mpc):
        """Return linear power at (z, k_Mpc)"""
        raise NotImplementedError()

    def compute_hubble_parameter(self, z):
        """Return H(z) in units of km/s/Mpc"""
        raise NotImplementedError()

    def compute_angular_diameter_distance(self, z):
        """Return angular diameter distance (not comoving) in Mpc"""
        raise NotImplementedError()


    # below here, no need to overwrite


    def get_linP_Mpc(self, z, k_Mpc):
        """Return linear density power spectrum at (z, k_Mpc)"""

        return self.compute_linP_Mpc(z, k_Mpc)


    def get_linP_hMpc(self, z, k_hMpc):
        """Return linear density power spectrum at (z, k_Mpc)"""

        h = self.compute_hubble_parameter(z=0) / 100
        k_Mpc = k_hMpc * h
        pk_Mpc = self.compute_linP_Mpc(z, k_Mpc)
        pk_hMpc = pk_Mpc * h**3

        return pk_hMpc


    def get_linP_kms(self, z, k_kms):
        """Return linear power in velocity units"""

        k_Mpc = k_kms * self.get_dkms_dMpc(z)
        pk_Mpc = self.get_linP_Mpc(z, k_Mpc)
        pk_kms = pk_Mpc * self.get_dkms_dMpc(z)**3

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
        h = self.compute_hubble_parameter(z=0) / 100.0
        dvdX_hMpc = dvdX_Mpc / h
        return dvdX_hMpc


    def get_dAA_dMpc(self, z, lambda_AA):
        """Factor to translate wavelength separations (in AA) to comoving
        separations (in Mpc)."""

        dkms_dMpc = self.get_dkms_dMpc(z)
        dAA_dkms = (1.0 + z) * lambda_AA / c_kms
        return dkms_dMpc * dAA_dkms


    def get_drad_dMpc(self, z):
        """Factor to translate angular separations (in radians) to comoving
            transverse separations (in Mpc)."""

        # this should be defined in proper Mpc (not comoving)
        ang_dist = self.compute_angular_diameter_distance(z)
        D_M = ang_dist * (1+z)
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
        """Parameters describing the lienar power around kp_Mpc"""

        # specify wavenumber range to fit
        kmin_over_kp = 0.5
        kmax_over_kp = 2.0
        k_over_kp = np.logspace(np.log10(kmin_over_kp), 
                                np.log10(kmax_over_kp), 100)
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

        linP_params = {
            "Delta2_p": Delta2_p,
            "n_p": n_p,
            "alpha_p": alpha_p
        }

        return linP_params

