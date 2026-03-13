import numpy as np
from lace.cosmo import base_cosmology


class RescaledCosmology(base_cosmology.BaseCosmology):
    """
    Given a fiducial cosmology, make predictions for other cosmologies
    that do not modify the background expansion.
    """

    def __init__(self, fid_cosmo, new_params_dict=None, verbose=False):

        if verbose:
            print("inside RescaledCosmology.__ini__")

        # make sure that you are not modifying the background
        assert fid_cosmo.same_background(new_params_dict), "background not fixed"

        self.fid_cosmo = fid_cosmo
        if new_params_dict is None:
            self.new_params = {}
        else:
            self.new_params = new_params_dict

        # initialize BaseClass cosmo (should be a formality)
        super().__init__(verbose)

        return

    # overwrite virtual functions in base class

    def compute_hubble_parameter(self, z):
        """Return H(z) in units of km/s/Mpc"""

        return self.fid_cosmo.compute_hubble_parameter(z)

    def compute_angular_diameter_distance(self, z):
        """Return angular diameter distance (not comoving) in Mpc"""

        return self.fid_cosmo.compute_angular_diameter_distance(z)

    def compute_linP_Mpc(self, z, k_Mpc, species="bc"):
        """Return linear power at (z, k_Mpc) (will call CAMB if needed)"""

        linP_Mpc = self.fid_cosmo.compute_linP_Mpc(z, k_Mpc, species=species)
        scaling = self.get_linP_Mpc_scaling(k_Mpc)
        return linP_Mpc * scaling

    def compute_growth_rate(self, z):
        """Return logarithmic growth rate (f) at z"""

        return self.fid_cosmo.compute_growth_rate(z)

    # other functions specific to this class below

    def get_linP_Mpc_scaling(self, k_Mpc):
        """Multiplicative correction to fiducial primordial power"""

        # primordial power in fiducial cosmology
        fid_As = self.fid_cosmo.CAMBparams.InitPower.As
        fid_ns = self.fid_cosmo.CAMBparams.InitPower.ns
        fid_nrun = self.fid_cosmo.CAMBparams.InitPower.nrun

        # assume standard pivot point
        assert self.fid_cosmo.CAMBparams.InitPower.pivot_scalar == 0.05
        k_s = self.fid_cosmo.CAMBparams.InitPower.pivot_scalar
        k_over_k_s = k_Mpc / k_s

        # modifications in this cosmology
        new_As = self.new_params.get("As", fid_As)
        new_ns = self.new_params.get("ns", fid_ns)
        new_nrun = self.new_params.get("nrun", fid_nrun)

        # compute scaling
        ratio_As = new_As / fid_As
        delta_ns = new_ns - fid_ns
        delta_nrun = new_nrun - fid_nrun

        ln_scaling = np.log(ratio_As) + delta_ns * k_over_k_s
        ln_scaling += 0.5 * delta_nrun * k_over_k_s**2

        return np.exp(ln_scaling)
