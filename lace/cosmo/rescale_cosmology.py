import numpy as np
from lace.cosmo import base_cosmology


class IncompatibleBackgroundError(ValueError):
    """Raised when primordial rescaling cannot represent a cosmology."""

    def __init__(self, changes):
        self.changes = changes
        details = "; ".join(
            f"{name}: fiducial={old_value!r}, requested={new_value!r}"
            for name, (old_value, new_value) in changes.items()
        )
        super().__init__(
            "RescaledCosmology requires unchanged background parameters; "
            + details
        )


class RescaledCosmology(base_cosmology.BaseCosmology):
    """
    Given a fiducial cosmology, make predictions for other cosmologies
    that do not modify the background expansion.
    """

    def __init__(self, fid_cosmo, new_params_dict=None, verbose=False):

        if verbose:
            print("inside RescaledCosmology.__ini__")

        changes = self._get_background_changes(fid_cosmo, new_params_dict)
        if changes:
            raise IncompatibleBackgroundError(changes)

        pivot_scalar = fid_cosmo.CAMBparams.InitPower.pivot_scalar
        if not np.isclose(pivot_scalar, 0.05, rtol=0.0, atol=1e-12):
            raise ValueError(
                "RescaledCosmology requires pivot_scalar=0.05 1/Mpc; "
                f"fiducial pivot_scalar={pivot_scalar!r} 1/Mpc"
            )

        self.fid_cosmo = fid_cosmo
        if new_params_dict is None:
            self.new_params = {}
        else:
            self.new_params = new_params_dict

        # initialize BaseClass cosmo (should be a formality)
        super().__init__(verbose)

        return

    @staticmethod
    def _get_background_changes(fid_cosmo, new_params_dict):
        """Return requested background changes as old/new value pairs."""

        if new_params_dict is None:
            return {}
        changes = {}
        for name, old_value in fid_cosmo.get_background_params().items():
            if name not in new_params_dict:
                continue
            new_value = new_params_dict[name]
            tolerance = 1e-4 if name == "mnu" else 0.0
            if not np.isclose(old_value, new_value, rtol=0.0, atol=tolerance):
                changes[name] = (old_value, new_value)
        return changes


    # overwrite virtual functions in base class

    def get_kmax_linP_Mpc(self):
        """Return highest k for which we trust linear power"""
        return self.fid_cosmo.get_kmax_linP_Mpc()


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

    def get_mnu(self):
        """Return the fiducial cosmology's neutrino mass in eV."""

        return self.fid_cosmo.get_mnu()

    # other functions specific to this class below

    def get_primordial_params(self):
        """Return primordial parameters after applying the rescaling."""

        params = self.fid_cosmo.get_primordial_params()
        params.update(self.new_params)
        return params

    def get_linP_Mpc_scaling(self, k_Mpc):
        """Multiplicative correction to fiducial primordial power"""

        # primordial power in fiducial cosmology
        fid_params = self.fid_cosmo.get_primordial_params()
        fid_As = fid_params["As"]
        fid_ns = fid_params["ns"]
        fid_nrun = fid_params["nrun"]

        # assume standard pivot point
        k_s = self.fid_cosmo.CAMBparams.InitPower.pivot_scalar
        ln_k_over_k_s = np.log(k_Mpc / k_s)

        # modifications in this cosmology
        new_As = self.new_params.get("As", fid_As)
        new_ns = self.new_params.get("ns", fid_ns)
        new_nrun = self.new_params.get("nrun", fid_nrun)

        # compute scaling
        ratio_As = new_As / fid_As
        delta_ns = new_ns - fid_ns
        delta_nrun = new_nrun - fid_nrun

        ln_scaling = np.log(ratio_As) + delta_ns * ln_k_over_k_s
        ln_scaling += 0.5 * delta_nrun * ln_k_over_k_s**2

        return np.exp(ln_scaling)
