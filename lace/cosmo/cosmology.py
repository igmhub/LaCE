import numpy as np
import camb

from lace.cosmo import base_cosmology
from lace.cosmo.parameters import normalize_cosmology_params
import lace.cosmo.labeled_cosmologies as lab_cosmo


class Cosmology(base_cosmology.BaseCosmology):
    """
    Class to interact with cosmological parameters and linear power.

    CAMB is an implementation detail of this concrete cosmology class.
    Callers should use the methods inherited from ``BaseCosmology`` rather
    than accessing ``CAMBparams`` or ``CAMBdata`` directly.
    """

    def __init__(
        self,
        cosmo_params_dict=None,
        cosmo_label=None,
        camb_kmax_Mpc=200.0,
        verbose=False,
    ):
        if verbose:
            print("inside Cosmology.__init__")

        if (cosmo_params_dict is not None) and (cosmo_label is not None):
            raise ValueError("You cannot provide both cosmo params and label")

        if cosmo_params_dict is None:
            if cosmo_label is None:
                cosmo_label = "Planck18"
            cosmo_params_dict = lab_cosmo.get_cosmo_params_dict_from_label(cosmo_label)

        self.input_cosmo_label = cosmo_label
        self.input_cosmo_params_dict = normalize_cosmology_params(cosmo_params_dict)
        self.CAMBparams = self._build_camb_params(self.input_cosmo_params_dict)
        self.ks_Mpc = self.CAMBparams.InitPower.pivot_scalar

        self.background_params = self.get_background_params()
        self.CAMBdata = None
        self.linP_Mpc_bc_interp = None
        self.linP_Mpc_bcnu_interp = None
        self.camb_kmax_Mpc = camb_kmax_Mpc

        super().__init__(verbose)

    @classmethod
    def from_dict(cls, params, **kwargs):
        """Construct a cosmology from a parameter dictionary."""

        return cls(cosmo_params_dict=params, **kwargs)

    @staticmethod
    def _build_camb_params(params):
        """Build CAMB parameters from normalized input values."""

        defaults = {
            "H0": 67.66,
            "mnu": 0.0,
            "omch2": 0.11933,
            "ombh2": 0.02242,
            "omk": 0.0,
            "As": np.exp(3.047) * 1e-10,
            "ns": 0.9665,
            "nrun": 0.0,
            "nrunrun": 0.0,
            "pivot_scalar": 0.05,
            "nnu": camb.constants.default_nnu,
            "w": -1.0,
            "wa": 0.0,
            "YHe": None,
            "TCMB": camb.constants.COBE_CMBTemp,
            "standard_neutrino_neff": camb.constants.default_nnu,
            "tau": 0.0,
        }
        values = {**defaults, **params}

        if values["YHe"] is None:
            bbn_predictor = camb.bbn.get_predictor()
            values["YHe"] = bbn_predictor.Y_He(
                values["ombh2"]
                * (camb.constants.COBE_CMBTemp / values["TCMB"]) ** 3,
                values["nnu"] - values["standard_neutrino_neff"],
            ).item()

        camb_params = camb.CAMBparams(
            WantTensors=values.get("r", 0.0) != 0.0
        )
        camb_params.set_cosmology(
            H0=values.get("H0"),
            cosmomc_theta=values.get("cosmomc_theta"),
            ombh2=values["ombh2"],
            omch2=values["omch2"],
            omk=values["omk"],
            mnu=values["mnu"],
            nnu=values["nnu"],
            tau=values["tau"],
            YHe=values["YHe"],
            standard_neutrino_neff=values["standard_neutrino_neff"],
            TCMB=values["TCMB"],
        )

        w = values["w"]
        wa = values["wa"]
        dark_energy_model = (
            "ppf" if ((w + 1) < -1e-6 or (1 + w + wa) < -1e-6) else "fluid"
        )
        camb_params.set_dark_energy(
            w=w, wa=wa, cs2=1.0, dark_energy_model=dark_energy_model
        )
        camb_params.InitPower.set_params(
            As=values["As"],
            ns=values["ns"],
            nrun=values["nrun"],
            nrunrun=values["nrunrun"],
            r=values.get("r", 0.0),
            pivot_scalar=values["pivot_scalar"],
        )
        return camb_params

    # BaseCosmology implementation

    def get_kmax_linP_Mpc(self):
        """Return highest k for which we trust linear power."""

        return self.camb_kmax_Mpc

    def compute_hubble_parameter(self, z):
        """Return H(z) in units of km/s/Mpc."""

        self._ensure_camb_results()
        return self.CAMBdata.hubble_parameter(z)

    def compute_angular_diameter_distance(self, z):
        """Return angular diameter distance in Mpc."""

        self._ensure_camb_results()
        return self.CAMBdata.angular_diameter_distance(z)

    def compute_linP_Mpc(self, z, k_Mpc, species="bc"):
        """Return linear power at ``(z, k_Mpc)``."""

        self._ensure_camb_results(full=True)
        if species == "bc":
            interp = self.linP_Mpc_bc_interp
        elif species == "bcnu":
            interp = self.linP_Mpc_bcnu_interp
        else:
            raise ValueError("species must be 'bc' or 'bcnu'")

        z = np.asarray(z)
        k_Mpc = np.asarray(k_Mpc)
        if z.min() < interp.zmin or z.max() > interp.zmax:
            raise ValueError(
                f"Requested z range [{z.min()}, {z.max()}] is outside "
                f"interpolation range [{interp.zmin}, {interp.zmax}]"
            )
        if k_Mpc.max() > interp.kmax:
            raise ValueError(
                f"Requested k_Mpc={k_Mpc.max()} exceeds "
                f"interpolation range kmax_Mpc={interp.kmax}"
            )

        if z.ndim == 0:
            return interp.P(z.item(), k_Mpc)
        if z.ndim == 1 and k_Mpc.ndim == 1:
            return interp.P(z, k_Mpc, grid=True)
        raise ValueError("z and k_Mpc must be 0D or 1D arrays")

    def compute_growth_rate(self, z):
        """Return the logarithmic growth rate ``f`` at redshift ``z``."""

        self._ensure_camb_results(full=True)
        z_transfer = np.asarray(self.CAMBdata.transfer_redshifts)
        fsig8 = np.asarray(self.CAMBdata.get_fsigma8())
        sig8 = np.asarray(self.CAMBdata.get_sigma8())
        f = fsig8 / sig8
        ind_sort = np.argsort(z_transfer)
        return np.interp(z, z_transfer[ind_sort], f[ind_sort])

    def get_mnu(self):
        """Return the total neutrino mass in eV."""

        return self.CAMBparams.omnuh2 * 93.14

    def get_primordial_params(self):
        """Return public primordial-spectrum parameters."""

        power = self.CAMBparams.InitPower
        return {
            "As": power.As,
            "ns": power.ns,
            "nrun": power.nrun,
            "nrunrun": power.nrunrun,
            "pivot_scalar": power.pivot_scalar,
        }

    def get_background_params(self):
        """Return parameters that change the background expansion."""

        return {
            "H0": self.CAMBparams.H0,
            "ombh2": self.CAMBparams.ombh2,
            "omch2": self.CAMBparams.omch2,
            "omk": self.CAMBparams.omk,
            "omnuh2": self.CAMBparams.omnuh2,
            "mnu": self.get_mnu(),
            "w": self.CAMBparams.DarkEnergy.w,
            "wa": self.CAMBparams.DarkEnergy.wa,
        }

    def same_background(self, cosmo_params):
        """Check whether parameters preserve the background expansion."""

        if cosmo_params is None:
            return True
        back_params = self.get_background_params()
        for name, value in back_params.items():
            if name not in cosmo_params:
                continue
            tolerance = 1e-4 if name == "mnu" else 0.0
            if not np.isclose(value, cosmo_params[name], rtol=0.0, atol=tolerance):
                if self.verbose:
                    print("background parameter differs", name, value, cosmo_params[name])
                return False
        return True

    def print_info(self, simulation=False):
        """Print the relevant cosmological parameters."""

        params = self.CAMBparams
        if simulation:
            omega_bc = (params.omch2 + params.ombh2) / params.h**2
            print(
                "H0 = {:.4E}, Omega_bc = {:.4E}, A_s = {:.4E}, "
                "n_s = {:.4E}, alpha_s = {:.4E}".format(
                    params.H0,
                    omega_bc,
                    params.InitPower.As,
                    params.InitPower.ns,
                    params.InitPower.nrun,
                )
            )
        else:
            print(
                "H0 = {:.4E}, Omega_b h^2 = {:.4E}, Omega_c h^2 = {:.4E}, "
                "Omega_k = {:.4E}, Omega_nu h^2 = {:.4E}, A_s = {:.4E}, "
                "n_s = {:.4E}, alpha_s = {:.4E}".format(
                    params.H0,
                    params.ombh2,
                    params.omch2,
                    params.omk,
                    params.omnuh2,
                    params.InitPower.As,
                    params.InitPower.ns,
                    params.InitPower.nrun,
                )
            )

    def get_CAMBdata(self):
        """Return raw CAMB data for backend-specific diagnostics."""

        self._ensure_camb_results(full=True)
        return self.CAMBdata

    def _ensure_camb_results(self, full=False):
        if full and (
            self.linP_Mpc_bc_interp is None or self.linP_Mpc_bcnu_interp is None
        ):
            self._call_camb_results_full()
        elif self.CAMBdata is None:
            self._call_camb_results_background()

    def _call_camb_results_background(self):
        self.CAMBdata = camb.get_results(self.CAMBparams)

    def _call_camb_results_full(self):
        zs = np.linspace(0, 10, 256)
        self.CAMBparams.set_matter_power(
            redshifts=zs,
            kmax=1.001 * self.camb_kmax_Mpc,
            nonlinear=False,
            silent=True,
        )
        self.CAMBdata = camb.get_results(self.CAMBparams)
        self.linP_Mpc_bc_interp = self.CAMBdata.get_matter_power_interpolator(
            nonlinear=False,
            var1=8,
            var2=8,
            hubble_units=False,
            k_hunit=False,
            log_interp=True,
        )
        self.linP_Mpc_bcnu_interp = self.CAMBdata.get_matter_power_interpolator(
            nonlinear=False,
            var1=7,
            var2=7,
            hubble_units=False,
            k_hunit=False,
            log_interp=True,
        )
