import numpy as np
from lace.cosmo import base_cosmology


class IncompatibleBackgroundError(ValueError):
    """Raised when primordial rescaling cannot represent a cosmology."""

    def __init__(self, changes):
        """Describe background changes that invalidate primordial rescaling.

        Parameters
        ----------
        changes : mapping
            Changed parameter names mapped to ``(fiducial, requested)`` pairs.
        """
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
        """Construct a primordial-spectrum rescaling of a fiducial cosmology.

        Parameters
        ----------
        fid_cosmo : BaseCosmology
            Cosmology supplying unchanged background and transfer functions.
        new_params_dict : mapping, optional
            Allowed primordial amplitude, tilt, and running updates.
        verbose : bool, default=False
            Print construction diagnostics.

        Raises
        ------
        IncompatibleBackgroundError
            If requested values change a background or transfer parameter.
        ValueError
            If the fiducial primordial pivot is not 0.05 1/Mpc.
        """

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
        """Identify requested changes incompatible with fixed transfers.

        Parameters
        ----------
        fid_cosmo : BaseCosmology
            Reference cosmology.
        new_params_dict : mapping or None
            Candidate parameter updates.

        Returns
        -------
        dict
            Changed names mapped to fiducial/requested value pairs.
        """

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
        if "pivot_scalar" in new_params_dict:
            old_value = fid_cosmo.CAMBparams.InitPower.pivot_scalar
            new_value = new_params_dict["pivot_scalar"]
            if not np.isclose(old_value, new_value, rtol=0.0, atol=1e-12):
                changes["pivot_scalar"] = (old_value, new_value)
        for name in ("theta", "cosmomc_theta", "theta_MC_100"):
            if name in new_params_dict:
                changes[name] = ("requires a CAMB angular-size solve", new_params_dict[name])
        return changes


    # overwrite virtual functions in base class

    def get_kmax_linP_Mpc(self):
        """Return the fiducial linear-power limit in 1/Mpc."""
        return self.fid_cosmo.get_kmax_linP_Mpc()


    def compute_hubble_parameter(self, z):
        """Delegate Hubble-rate evaluation to the unchanged fiducial background."""

        return self.fid_cosmo.compute_hubble_parameter(z)

    def compute_angular_diameter_distance(self, z):
        """Delegate angular-diameter distance evaluation to the fiducial background."""

        return self.fid_cosmo.compute_angular_diameter_distance(z)

    def compute_linP_Mpc(self, z, k_Mpc, species="bc"):
        """Return fiducial linear power times the primordial rescaling.

        Parameters
        ----------
        z, k_Mpc, species
            Arguments forwarded to ``fid_cosmo.compute_linP_Mpc``; ``k_Mpc``
            is in 1/Mpc.

        Returns
        -------
        ndarray
            Linear power in Mpc cubed.
        """

        linP_Mpc = self.fid_cosmo.compute_linP_Mpc(z, k_Mpc, species=species)
        scaling = self.get_linP_Mpc_scaling(k_Mpc)
        return linP_Mpc * scaling

    def compute_growth_rate(self, z):
        """Delegate the unchanged logarithmic growth rate to the fiducial model."""

        return self.fid_cosmo.compute_growth_rate(z)

    def get_mnu(self):
        """Return the fiducial cosmology's neutrino mass in eV."""

        return self.fid_cosmo.get_mnu()

    # other functions specific to this class below

    def get_primordial_params(self):
        """Return fiducial primordial parameters updated by requested values."""

        params = self.fid_cosmo.get_primordial_params()
        params.update(self.new_params)
        return params

    def get_linP_Mpc_params(self, z, kp_Mpc, species="bc"):
        """Return pivot summaries from cached fiducial fits plus exact shifts.

        Parameters
        ----------
        z : float
            Redshift.
        kp_Mpc : float
            Pivot wavenumber in 1/Mpc.
        species : {'bc', 'bcnu'}, default='bc'
            Transfer-function species.

        Returns
        -------
        dict
            ``Delta2_p``, ``n_p``, and ``alpha_p`` as returned by
            :meth:`BaseCosmology.get_linP_Mpc_params` for this cosmology.

        Notes
        -----
        The base method fits a quadratic in ``x = ln(k/kp)`` to ``ln P`` on a
        fixed grid; a least-squares fit is linear in ``ln P``. Rescaling adds
        ``ln S = a + b L + c L**2 + d L**3`` with ``L = x + ln(kp/k_s)``, so the
        rescaled fit is the fiducial fit (computed once per ``(z, kp, species)``
        and cached on the fiducial cosmology) plus the fit of ``ln S``: exact
        for the constant, linear and quadratic terms, and a fixed projection
        of ``x**3`` onto the quadratic for ``nrunrun``.
        """
        cache = self.fid_cosmo.__dict__.setdefault("_linP_Mpc_params_cache", {})
        key = (float(z), float(kp_Mpc), species)
        fiducial = cache.get(key)
        if fiducial is None:
            fiducial = self.fid_cosmo.get_linP_Mpc_params(z, kp_Mpc, species=species)
            cache[key] = fiducial

        fid_params = self.fid_cosmo.get_primordial_params()
        k_s = self.fid_cosmo.CAMBparams.InitPower.pivot_scalar
        ell = np.log(kp_Mpc / k_s)
        a = np.log(self.new_params.get("As", fid_params["As"]) / fid_params["As"])
        b = self.new_params.get("ns", fid_params["ns"]) - fid_params["ns"]
        c = 0.5 * (self.new_params.get("nrun", fid_params["nrun"]) - fid_params["nrun"])
        d = (self.new_params.get("nrunrun", fid_params["nrunrun"]) - fid_params["nrunrun"]) / 6.0

        # ln S expanded in x = ln(k/kp): const + lin x + quad x**2 + d x**3
        const = a + b * ell + c * ell**2 + d * ell**3
        lin = b + 2.0 * c * ell + 3.0 * d * ell**2
        quad = c + 3.0 * d * ell
        if d != 0.0:
            q2, q1, q0 = _cubic_projection()
            const += d * q0
            lin += d * q1
            quad += d * q2

        return {
            "Delta2_p": fiducial["Delta2_p"] * np.exp(const),
            "n_p": fiducial["n_p"] + lin,
            # the fitted curvature is alpha/2
            "alpha_p": fiducial["alpha_p"] + 2.0 * quad,
        }

    def get_linP_Mpc_scaling(self, k_Mpc):
        """Compute the multiplicative primordial-power correction.

        Parameters
        ----------
        k_Mpc : float or array-like
            Comoving wavenumber in 1/Mpc.

        Returns
        -------
        float or ndarray
            Dimensionless ratio of rescaled to fiducial linear power.
        """

        # primordial power in fiducial cosmology
        fid_params = self.fid_cosmo.get_primordial_params()
        fid_As = fid_params["As"]
        fid_ns = fid_params["ns"]
        fid_nrun = fid_params["nrun"]
        fid_nrunrun = fid_params["nrunrun"]

        # assume standard pivot point
        k_s = self.fid_cosmo.CAMBparams.InitPower.pivot_scalar
        ln_k_over_k_s = np.log(k_Mpc / k_s)

        # modifications in this cosmology
        new_As = self.new_params.get("As", fid_As)
        new_ns = self.new_params.get("ns", fid_ns)
        new_nrun = self.new_params.get("nrun", fid_nrun)
        new_nrunrun = self.new_params.get("nrunrun", fid_nrunrun)

        # compute scaling
        ratio_As = new_As / fid_As
        delta_ns = new_ns - fid_ns
        delta_nrun = new_nrun - fid_nrun
        delta_nrunrun = new_nrunrun - fid_nrunrun

        ln_scaling = np.log(ratio_As) + delta_ns * ln_k_over_k_s
        ln_scaling += 0.5 * delta_nrun * ln_k_over_k_s**2
        ln_scaling += (delta_nrunrun / 6.0) * ln_k_over_k_s**3

        return np.exp(ln_scaling)


def _cubic_projection():
    """Return the quadratic least-squares fit of ``x**3`` on the pivot grid.

    The grid (100 log-spaced ``k/kp`` in ``[0.5, 2]``) is that of
    :meth:`BaseCosmology.get_linP_Mpc_params`; coefficients are ordered
    highest power first, as in :func:`numpy.polyfit`.
    """
    x = np.log(base_cosmology.LINP_FIT_K_OVER_KP)
    return np.polyfit(x, x**3, deg=2)
