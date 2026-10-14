import numpy as np


class BaseCosmology(object):
    """Abstract cosmology interface with shared coordinate conversions.

    Subclasses provide linear power, background expansion, distances, and
    growth. This base class derives common Mpc, Mpc/h, and velocity-space
    quantities without changing the underlying transfer-function species.
    """

    def __init__(self, verbose=False):
        """Initialize base diagnostics.

        Parameters
        ----------
        verbose : bool, default=False
            Print a construction message.
        """

        self.verbose = verbose

        if self.verbose:
            print("inside BaseCosmology.__ini__")

        return


    # five functions that other cosmo classes should implement

    def get_kmax_linP_Mpc(self):
        """Return the maximum trusted linear wavenumber in 1/Mpc.

        Raises
        ------
        NotImplementedError
            Subclasses must define their linear-power validity limit.
        """
        raise NotImplementedError()

    def compute_linP_Mpc(self, z, k_Mpc, species="bc"):
        """Return three-dimensional linear power in Mpc cubed.

        Parameters
        ----------
        z : float or array-like
            Redshift.
        k_Mpc : float or array-like
            Comoving wavenumber in 1/Mpc.
        species : {'bc', 'bcnu'}, default='bc'
            Baryon-plus-CDM or total-matter transfer-function species.

        Raises
        ------
        NotImplementedError
            Subclasses must provide linear power.
        """
        raise NotImplementedError()

    def compute_hubble_parameter(self, z):
        """Return Hubble expansion rate in km/s/Mpc.

        Raises
        ------
        NotImplementedError
            Subclasses must provide background expansion.
        """
        raise NotImplementedError()

    def compute_angular_diameter_distance(self, z):
        """Return proper angular-diameter distance in Mpc.

        Raises
        ------
        NotImplementedError
            Subclasses must provide angular-diameter distance.
        """
        raise NotImplementedError()

    def compute_growth_rate(self, z):
        """Return logarithmic linear growth rate at redshift ``z``.

        Raises
        ------
        NotImplementedError
            Subclasses must provide the growth rate.
        """
        raise NotImplementedError()

    def get_mnu(self):
        """Return total neutrino mass in eV.

        Raises
        ------
        NotImplementedError
            Subclasses must provide neutrino mass.
        """
        raise NotImplementedError()

    def get_primordial_params(self):
        """Return public primordial-spectrum parameters.

        Raises
        ------
        NotImplementedError
            Subclasses must provide primordial parameters.
        """
        raise NotImplementedError()


    # below here, no need to overwrite

    def get_H0(self):
        """Return the Hubble constant in km/s/Mpc.

        Returns
        -------
        float
            ``H(z=0)`` from the subclass background expansion.
        """

        return self.compute_hubble_parameter(z=0)

    def get_h(self):
        """Return dimensionless reduced Hubble constant ``H0 / 100``.

        Returns
        -------
        float
            Dimensionless reduced Hubble constant.
        """

        return self.get_H0() / 100.0

    def get_growth_rate(self, z):
        """Return logarithmic growth rate at a redshift.

        Parameters
        ----------
        z : float or array-like
            Redshift.

        Returns
        -------
        float or ndarray
            ``f = d ln D / d ln a`` delegated to the subclass.
        """
        return self.compute_growth_rate(z)

    def get_linP_Mpc(self, z, k_Mpc, species="bc"):
        """Return three-dimensional linear power on a 1/Mpc grid.

        Parameters
        ----------
        z : float or array-like
            Redshift.
        k_Mpc : float or array-like
            Comoving wavenumber in 1/Mpc.
        species : {'bc', 'bcnu'}, default='bc'
            Transfer-function species.

        Returns
        -------
        float or ndarray
            Linear power in Mpc cubed.
        """

        return self.compute_linP_Mpc(z, k_Mpc, species=species)

    def get_linP_hMpc(self, z, k_hMpc, species="bc"):
        """Return linear matter power at wavenumbers in h/Mpc.

        Parameters
        ----------
        z : float or array_like
            Redshift of the evolved linear spectrum.
        k_hMpc : float or array_like
            Comoving wavenumber in h/Mpc.
        species : {"bc", "bcnu"}, default="bc"
            Baryon-plus-CDM or total-matter transfer-function species.

        Returns
        -------
        float or numpy.ndarray
            Linear power in (Mpc/h)^3, with the shape returned by the
            implementation for ``z`` and ``k_hMpc``.
        """

        h = self.get_h()
        k_Mpc = k_hMpc * h
        pk_Mpc = self.compute_linP_Mpc(z, k_Mpc, species=species)
        pk_hMpc = pk_Mpc * h**3

        return pk_hMpc

    def get_sigma8(self, z, species="bcnu"):
        """Compute the 8 Mpc/h top-hat RMS fluctuation amplitude.

        Parameters
        ----------
        z : float
            Redshift of the linear spectrum.
        species : {'bc', 'bcnu'}, default='bcnu'
            Transfer-function species integrated in the variance.

        Returns
        -------
        float
            ``sigma8`` from a log-wavenumber integral over ``1e-4`` to
            ``1e2`` h/Mpc.
        """
        from scipy.integrate import simpson

        def fft_top_hat(k_hMpc, R_hMpc=8.0):
            """Evaluate Fourier-space spherical top-hat window.

            Parameters
            ----------
            k_hMpc : ndarray
                Wavenumbers in h/Mpc.
            R_hMpc : float, default=8.0
                Top-hat radius in Mpc/h.

            Returns
            -------
            ndarray
                Dimensionless window values, using a small-argument limit.
            """
            x = k_hMpc * R_hMpc
            win = np.zeros_like(k_hMpc)
            # CAMB implementation https://github.com/cmbant/CAMB/blob/master/fortran/results.f90
            _ = np.argwhere(x < 1e-2)[:, 0]
            win[_] = 1 - x[_] ** 2 / 10
            _ = np.argwhere(x >= 1e-2)[:, 0]
            win[_] = 3 / x[_] ** 3 * (np.sin(x[_]) - x[_] * np.cos(x[_]))
            return win

        k_hMpc = np.logspace(-4, 2, 1000)
        linP_hMpc = self.get_linP_hMpc(z, k_hMpc, species=species)
        integrand = (k_hMpc**3 * linP_hMpc / 2 / np.pi**2) * fft_top_hat(k_hMpc) ** 2
        sig8 = np.sqrt(simpson(integrand, x=np.log(k_hMpc)))
        return sig8

    def get_linP_kms(self, z, k_kms, species="bc"):
        """Return velocity-space three-dimensional linear power.

        Parameters
        ----------
        z : float or array-like
            Redshift.
        k_kms : float or array-like
            Velocity wavenumber in s/km.
        species : {'bc', 'bcnu'}, default='bc'
            Transfer-function species.

        Returns
        -------
        float or ndarray
            Three-dimensional power in (km/s)^3.

        Notes
        -----
        ``k_iMpc = M k_ikms`` and ``P3D_kms = M**3 P3D_Mpc``, where
        ``M = H(z)/(1+z)`` in km/s/Mpc. This is a volume-power conversion,
        not the one-factor Jacobian used for P1D.
        """

        k_Mpc = k_kms * self.get_dkms_dMpc(z)
        pk_Mpc = self.get_linP_Mpc(z, k_Mpc, species=species)
        pk_kms = pk_Mpc * self.get_dkms_dMpc(z) ** 3

        return pk_kms

    def get_dkms_diMpc(self, z):
        """Return velocity-to-comoving conversion ``H(z)/(1+z)``.

        Parameters
        ----------
        z : float or array-like
            Redshift.

        Returns
        -------
        float or ndarray
            Conversion in km/s/Mpc, used as ``k_iMpc = M k_ikms``.
        """

        H_z = self.compute_hubble_parameter(z)
        dvdX = H_z / (1 + z)
        return dvdX

    def get_dkms_dMpc(self, z):
        """Return the legacy-named velocity-to-comoving conversion.

        Parameters
        ----------
        z : float or array-like
            Redshift.

        Returns
        -------
        float or ndarray
            ``H(z)/(1+z)`` in km/s/Mpc.
        """
        return self.get_dkms_diMpc(z)

    def get_dkms_dhMpc(self, z):
        """Return velocity-to-comoving conversion in km/s per Mpc/h.

        Parameters
        ----------
        z : float or array-like
            Redshift.

        Returns
        -------
        float or ndarray
            ``H(z)/((1+z) h)`` in km/s/(Mpc/h).
        """

        dvdX_Mpc = self.get_dkms_dMpc(z)
        h = self.get_h()
        dvdX_hMpc = dvdX_Mpc / h
        return dvdX_hMpc

    def get_dAA_dMpc(self, z, lambda_rest_AA=1215.67):
        """Return observed-wavelength to comoving-radial conversion.

        Parameters
        ----------
        z : float or array-like
            Redshift.
        lambda_rest_AA : float, default=1215.67
            Rest wavelength in Angstrom.

        Returns
        -------
        float or ndarray
            Conversion in Angstrom/Mpc.
        """

        import scipy.constants

        # speed of light in km/s
        c_kms = scipy.constants.c / 1e3

        dkms_dMpc = self.get_dkms_dMpc(z)
        dAA_dkms = (1.0 + z) * lambda_rest_AA / c_kms
        return dkms_dMpc * dAA_dkms

    def get_drad_dMpc(self, z):
        """Return angular-to-comoving-transverse conversion in rad/Mpc.

        Parameters
        ----------
        z : float or array-like
            Redshift.

        Returns
        -------
        float or ndarray
            Reciprocal transverse comoving distance in rad/Mpc.
        """

        # this should be defined in proper Mpc (not comoving)
        ang_dist = self.compute_angular_diameter_distance(z)
        D_M = ang_dist * (1 + z)
        return 1.0 / D_M

    def get_ddeg_dMpc(self, z):
        """Return angular-to-comoving-transverse conversion in deg/Mpc.

        Parameters
        ----------
        z : float or array-like
            Redshift.

        Returns
        -------
        float or ndarray
            Reciprocal transverse comoving distance in deg/Mpc.
        """

        drad_dMpc = self.get_drad_dMpc(z)
        return 180.0 / np.pi * drad_dMpc

    def get_darc_dMpc(self, z):
        """Return angular-to-comoving-transverse conversion in arcmin/Mpc.

        Parameters
        ----------
        z : float or array-like
            Redshift.

        Returns
        -------
        float or ndarray
            Reciprocal transverse comoving distance in arcmin/Mpc.
        """

        drad_dMpc = self.get_drad_dMpc(z)
        return 180.0 / np.pi * 60.0 * drad_dMpc

    @staticmethod
    def get_dkms_dMpc_for_cosmologies(cosmologies, zs):
        """Return velocity conversions for one cosmology or a cosmology batch.

        Parameters
        ----------
        cosmologies : BaseCosmology or sequence of BaseCosmology
            One cosmology or an ordered batch.
        zs : float or array-like
            Redshifts.

        Returns
        -------
        float or ndarray
            Conversion with scalar/single-cosmology shape preserved, or leading
            ``(n_cosmologies, n_z)`` axes for a sequence.

        A single ``Cosmology``/``RescaledCosmology`` returns the usual scalar
        or redshift-array result. A sequence returns an array with leading
        ``(n_cosmologies, n_z)`` axes. This is the shape-dispatching public
        interface used by cup1d.
        """

        if isinstance(cosmologies, BaseCosmology):
            return cosmologies.get_dkms_dMpc(zs)
        zs = np.atleast_1d(np.asarray(zs, dtype=float))
        return np.asarray([cosmo.get_dkms_dMpc(zs) for cosmo in cosmologies])

    @staticmethod
    def get_linP_Mpc_params_for_cosmologies(cosmologies, zs, kp_Mpc, species="bc"):
        """Return Mpc-pivot summaries for one cosmology or a batch.

        Parameters
        ----------
        cosmologies : BaseCosmology or sequence of BaseCosmology
            One cosmology or an ordered batch.
        zs : float or array-like
            Redshifts.
        kp_Mpc : float
            Comoving pivot wavenumber in 1/Mpc.
        species : {'bc', 'bcnu'}, default='bc'
            Transfer-function species.

        Returns
        -------
        dict
            ``Delta2_p``, ``n_p``, and ``alpha_p`` summaries with scalar,
            redshift, or leading cosmology-and-redshift axes.

        A single cosmology returns the scalar dictionary for scalar ``zs`` or
        a dictionary of ``(n_z,)`` arrays for redshift arrays. A sequence
        returns dictionary values shaped ``(n_cosmologies, n_z)``.
        """

        if isinstance(cosmologies, BaseCosmology):
            if np.asarray(zs).ndim == 0:
                return cosmologies.get_linP_Mpc_params(zs, kp_Mpc, species=species)
            values = [
                cosmologies.get_linP_Mpc_params(z, kp_Mpc, species=species)
                for z in np.asarray(zs)
            ]
            return {name: np.asarray([row[name] for row in values]) for name in values[0]}
        zs = np.atleast_1d(np.asarray(zs, dtype=float))
        values = [
            [cosmo.get_linP_Mpc_params(z, kp_Mpc, species=species) for z in zs]
            for cosmo in cosmologies
        ]
        return {
            name: np.asarray([[row[name] for row in result] for result in values])
            for name in ("Delta2_p", "n_p", "alpha_p")
        }

    @staticmethod
    def get_linP_kms_params_for_cosmologies(cosmologies, zs, kp_kms, species="bc"):
        """Return velocity-pivot summaries for one cosmology or a batch.

        Parameters
        ----------
        cosmologies : BaseCosmology or sequence of BaseCosmology
            One cosmology or an ordered batch.
        zs : float or array-like
            Redshifts.
        kp_kms : float
            Velocity pivot wavenumber in s/km.
        species : {'bc', 'bcnu'}, default='bc'
            Transfer-function species.

        Returns
        -------
        dict
            ``Delta2_star``, ``n_star``, and ``alpha_star`` summaries with
            scalar, redshift, or leading cosmology-and-redshift axes.

        Return shapes follow :meth:`get_linP_Mpc_params_for_cosmologies`.
        """

        if isinstance(cosmologies, BaseCosmology):
            if np.asarray(zs).ndim == 0:
                return cosmologies.get_linP_kms_params(zs, kp_kms, species=species)
            values = [
                cosmologies.get_linP_kms_params(z, kp_kms, species=species)
                for z in np.asarray(zs)
            ]
            return {name: np.asarray([row[name] for row in values]) for name in values[0]}
        zs = np.atleast_1d(np.asarray(zs, dtype=float))
        values = [
            [cosmo.get_linP_kms_params(z, kp_kms, species=species) for z in zs]
            for cosmo in cosmologies
        ]
        return {
            name: np.asarray([[row[name] for row in result] for result in values])
            for name in ("Delta2_star", "n_star", "alpha_star")
        }

    def get_linP_Mpc_params(self, z, kp_Mpc, species="bc"):
        """Fit finite-window linear-power summaries around an Mpc pivot.

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
            Dimensionless pivot amplitude ``Delta2_p``, logarithmic slope
            ``n_p``, and running ``alpha_p`` fitted over ``0.5--2 kp``.
        """

        # specify wavenumber range to fit
        kmin_over_kp = 0.5
        kmax_over_kp = 2.0
        k_over_kp = np.logspace(np.log10(kmin_over_kp), np.log10(kmax_over_kp), 100)
        k_Mpc = kp_Mpc * k_over_kp

        # get power spectrum in this range
        linP_Mpc = self.get_linP_Mpc(z, k_Mpc, species=species)

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

    def get_linP_kms_params(self, z, kp_kms, species="bc"):
        """Return finite-window linear-power summaries at a velocity pivot.

        Parameters
        ----------
        z : float
            Redshift.
        kp_kms : float
            Pivot wavenumber in s/km.
        species : {'bc', 'bcnu'}, default='bc'
            Transfer-function species.

        Returns
        -------
        dict
            ``Delta2_star``, ``n_star``, and ``alpha_star`` evaluated after
            converting the velocity pivot to 1/Mpc at ``z``.
        """

        # translate the pivot point to Mpc
        kp_Mpc = kp_kms * self.get_dkms_dMpc(z)

        # get the parameters in Mpc
        linP_Mpc_params = self.get_linP_Mpc_params(z, kp_Mpc, species=species)

        # modify the names
        linP_kms_params = {
            "Delta2_star": linP_Mpc_params["Delta2_p"],
            "n_star": linP_Mpc_params["n_p"],
            "alpha_star": linP_Mpc_params["alpha_p"],
        }

        return linP_kms_params
