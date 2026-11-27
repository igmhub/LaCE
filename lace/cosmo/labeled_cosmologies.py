"""Named cosmologies: Planck fiducials, reference variations and simulations.

Every label accepted by :class:`lace.cosmo.cosmology.Cosmology` (and by
cup1d's ``set_cosmo``) is defined here:

- the Planck 2018 fiducials (:data:`COSMOLOGY_LABELS`);
- reference variations around them (omch2, H0, mnu, DESI DR2 + ACT);
- the cosmologies of the simulation suites: ``mpg_*`` (MP-Gadget,
  ``data/sim_suites/Australia20/mpg_emu_cosmo.npy``), ``nyx_*`` (Nyx, in the
  configured Nyx directory), ``accel2`` and ``sherwood``.
"""

import warnings

import numpy as np

#: Nyx cosmology catalogue read by default for ``nyx_*`` labels.
DEFAULT_NYX_VERSION = "models_Nyx_Mar2025_with_CGAN_val_3axes"

_PLANCK18_BASE = {
    "H0": 67.66,
    "mnu": 0.0,
    "omch2": 0.119,
    "ombh2": 0.0224,
    "omk": 0.0,
    "As": 2.105e-09,
    "ns": 0.9665,
    "nrun": 0.0,
    "pivot_scalar": 0.05,
    "w": -1,
}
_ERR_OMCH2 = 0.0009  # 1 sigma Planck 2018 uncertainty on omch2
_OMNUH2_MNU03 = 0.00322433285312557  # omnuh2 for mnu = 0.3 eV


def _planck18_variation(**changes):
    """Return the Planck 2018-like base parameters with ``changes`` applied."""
    return {**_PLANCK18_BASE, **changes}


def _desidr2_act():
    """DESI DR2 + P-ACT + DESY5 w0waCDM (ACT 2503.14452, DESI 2504.18464)."""
    # ACT https://arxiv.org/pdf/2503.14452, Table 5 (P-ACT)
    ombh2 = 0.0225
    ns = 0.9709
    As = np.exp(3.056) / 1e10
    # DESI https://arxiv.org/pdf/2504.18464, Table 3 (DESI+P-ACT+DESY5)
    h = 0.6685
    om = 0.3175
    return {
        "H0": h * 100,
        "mnu": 0.0,
        "omch2": om * h**2 - ombh2,
        "ombh2": ombh2,
        "omk": 0.0,
        "As": As,
        "ns": ns,
        "nrun": 0.0,
        "pivot_scalar": 0.05,
        "w": -0.764,
        "wa": -0.77,
    }


def _accel2():
    """ACCEL2 6144^3 160 Mpc (arXiv:2407.04473), Planck15 TT,TE,EE+lowP-like."""
    Omegam = 0.31
    Omegab = 0.0487
    h = 0.675
    return {
        "H0": h * 100,
        "mnu": 0.0,
        "omch2": (Omegam - Omegab) * h**2,
        "ombh2": Omegab * h**2,
        "omk": 0.0,
        "As": np.exp(3.094) / 1e10,
        "ns": 0.96,
        "nrun": 0.0,
        "pivot_scalar": 0.05,
        "w": -1,
    }


def _sherwood():
    """Sherwood 2048^3 40 Mpc/h (MNRAS 464, 897), Planck13+WP+highL+BAO."""
    Omegam = 0.308
    Omegab = 0.0482
    h = 0.678
    return {
        "H0": h * 100,
        "mnu": 0.0,
        "omch2": (Omegam - Omegab) * h**2,
        "ombh2": Omegab * h**2,
        "omk": 0.0,
        "As": np.exp(3.0973) / 1e10,
        "ns": 0.961,
        "nrun": 0.0,
        "pivot_scalar": 0.05,
        "w": -1,
    }


def _planck18_mnu03_varh():
    """mnu = 0.3 eV at fixed Omega_m (cb) of the base, compensated with h."""
    from lace.cosmo.cosmology import Cosmology

    params = _planck18_variation(mnu=0.3)
    cosmo = Cosmology(cosmo_params_dict=params)
    background = cosmo.get_background_params()
    OmegaM = (background["omch2"] + background["ombh2"]) / cosmo.get_h() ** 2
    omh2_nu = background["omch2"] + background["ombh2"] + background["omnuh2"]
    return {**params, "H0": np.sqrt(omh2_nu / OmegaM) * 100}


#: Planck 2018 fiducials; the name states the data and whether neutrinos are
#: massive.
COSMOLOGY_LABELS = ("Planck18_wonu", "Planck18_wnu", "Planck18_noBAO_wnu")
#: Former names, still accepted (with a warning) to read older results.
DEPRECATED_LABELS = {
    "Planck18": "Planck18_wonu",
    "Planck18_mnu": "Planck18_wnu",
    "Planck18_noBAO": "Planck18_noBAO_wnu",
}

# label -> function returning a fresh parameter dictionary
_LABELED_COSMOLOGIES = {
    # Table 2 of https://arxiv.org/abs/1807.06209.pdf (T&E+lensing+BAO) w/ mnu=0
    # Used as baseline cosmology in the DESI Lya P1D analysis
    "Planck18_wonu": lambda: {
        "H0": 67.66,
        "mnu": 0,
        "omch2": 0.11933,
        "ombh2": 0.02242,
        "omk": 0,
        "As": np.exp(3.047) * 1e-10,
        "ns": 0.9665,
        "nrun": 0.0,
        "pivot_scalar": 0.05,
        "w": -1.0,
    },
    # Table 5 of https://arxiv.org/abs/1807.06209.pdf
    "Planck18_wnu": lambda: {
        "H0": 67.10,
        "mnu": 0.06,
        "omch2": 0.1201,
        "ombh2": 0.02236,
        "omk": 0.0,
        "As": np.exp(3.046) * 1e-10,
        "ns": 0.9647,
        "nrun": 0.0,
        "pivot_scalar": 0.05,
        "w": -1.0,
    },
    # Table 2 of https://arxiv.org/abs/1807.06209.pdf (T&E+lensing) w/ mnu=0.06
    # Used as baseline cosmology in the DESI Lya BAO analysis
    "Planck18_noBAO_wnu": lambda: {
        "H0": 67.36,
        "mnu": 0.06,
        "omch2": 0.1200,
        "ombh2": 0.02237,
        "omk": 0,
        "As": np.exp(3.044) * 1e-10,
        "ns": 0.9649,
        "nrun": 0.0,
        "pivot_scalar": 0.05,
        "w": -1.0,
    },
    # reference variations around a Planck 2018-like cosmology
    "Planck18_high_omh2": lambda: _planck18_variation(omch2=0.1309),  # +10%
    "Planck18_low_omh2": lambda: _planck18_variation(omch2=0.1071),  # -10%
    "Planck18_high3s_omh2": lambda: _planck18_variation(omch2=0.119 + 3 * _ERR_OMCH2),
    "Planck18_high1s_omh2": lambda: _planck18_variation(omch2=0.119 + _ERR_OMCH2),
    "Planck18_low3s_omh2": lambda: _planck18_variation(omch2=0.119 - 3 * _ERR_OMCH2),
    "Planck18_low1s_omh2": lambda: _planck18_variation(omch2=0.119 - _ERR_OMCH2),
    "Planck18_h74": lambda: _planck18_variation(H0=74.00),
    # mnu = 0.3 eV at fixed omh2 (omch2 compensates)
    "Planck18_mnu03": lambda: _planck18_variation(mnu=0.3, omch2=0.119 - _OMNUH2_MNU03),
    "Planck18_mnu03_varh": _planck18_mnu03_varh,
    "DESIDR2_ACT": _desidr2_act,
    # Planck-like cosmologies with the primordial parameters of each suite
    "Planck18_nyx": lambda: _planck18_variation(As=2.24e-09, ns=0.937),
    "Planck18_mpg": lambda: _planck18_variation(H0=67.0, ombh2=0.022, As=2.26e-09, ns=0.982),
    # other simulations
    "accel2": _accel2,
    "ACCEL2_6144_160": _accel2,
    "sherwood": _sherwood,
    "Sherwood_2048_40": _sherwood,
}


def get_sim_suite_cosmologies(suite, nyx_version=DEFAULT_NYX_VERSION):
    """Return the stored cosmology catalogue of a simulation suite.

    Parameters
    ----------
    suite : {"mpg", "nyx"}
        MP-Gadget (``Australia20/mpg_emu_cosmo.npy`` under the configured LaCE
        data path) or Nyx (``nyx_emu_cosmo_<nyx_version>.npy`` in the
        configured Nyx directory).
    nyx_version : str, default=DEFAULT_NYX_VERSION
        Nyx catalogue version.

    Returns
    -------
    dict
        Simulation label mapped to its stored entry (with ``cosmo_params``).

    Raises
    ------
    ValueError
        If the suite is unknown or its catalogue cannot be read.
    """
    from lace.configuration import get_data_path, get_nyx_path

    if suite == "mpg":
        fname = get_data_path() / "sim_suites" / "Australia20" / "mpg_emu_cosmo.npy"
    elif suite == "nyx":
        fname = get_nyx_path() / f"nyx_emu_cosmo_{nyx_version}.npy"
    else:
        raise ValueError(f"Unknown simulation suite {suite!r}: use 'mpg' or 'nyx'")
    try:
        return np.load(fname, allow_pickle=True).item()
    except OSError as error:
        raise ValueError(f"{fname} not found") from error


def _sim_suite_cosmo_params(label, nyx_version):
    """Parameters of simulation ``label`` (``mpg_*`` or ``nyx_*``)."""
    suite = label[:3]
    catalogue = get_sim_suite_cosmologies(suite, nyx_version=nyx_version)
    if label not in catalogue:
        raise ValueError(f"Unknown cosmology label = {label} (not in the {suite} catalogue)")
    params = catalogue[label]["cosmo_params"]
    if suite == "mpg":
        return dict(params)
    # Nyx catalogues store H_0, omega_m (physical), A_s and n_s
    return {
        "H0": params["H_0"],
        "ombh2": 0.02233,
        "omch2": params["omega_m"] - 0.02233,
        "mnu": 0.0,
        "As": params["A_s"],
        "ns": params["n_s"],
        "nrun": params.get("nrun", 0.0),
    }


def get_cosmo_params_dict_from_label(label, nyx_version=DEFAULT_NYX_VERSION):
    """Return a mutable cosmology parameter dictionary for a named cosmology.

    Parameters
    ----------
    label : str
        ``"Planck18_wonu"`` (TT,TE,EE+lowE+lensing+BAO parameters, massless
        neutrinos; the DESI P1D baseline), ``"Planck18_wnu"`` (Table 5,
        ``mnu = 0.06`` eV) or ``"Planck18_noBAO_wnu"`` (TT,TE,EE+lowE+lensing,
        ``mnu = 0.06`` eV; the DESI Lya BAO baseline); the former names
        ``"Planck18"``, ``"Planck18_mnu"`` and ``"Planck18_noBAO"`` are
        accepted with a warning. Also the reference variations
        (``"Planck18_high_omh2"``, ``"Planck18_mnu03"``, ``"DESIDR2_ACT"``,
        ...) and the simulations: ``"mpg_<sim>"``, ``"nyx_<sim>"``,
        ``"accel2"`` and ``"sherwood"``.
    nyx_version : str, default=DEFAULT_NYX_VERSION
        Nyx catalogue read for ``nyx_*`` labels.

    Returns
    -------
    dict
        CAMB/LaCE parameter values in physical units.

    Raises
    ------
    ValueError
        If ``label`` is unknown, or its simulation catalogue cannot be read.
    """
    if label in DEPRECATED_LABELS:
        warnings.warn(
            f"cosmology label {label!r} is deprecated: use {DEPRECATED_LABELS[label]!r}",
            FutureWarning, stacklevel=2,
        )
        label = DEPRECATED_LABELS[label]
    if label[:3] in ("mpg", "nyx"):
        return _sim_suite_cosmo_params(label, nyx_version)
    if label in _LABELED_COSMOLOGIES:
        return _LABELED_COSMOLOGIES[label]()
    raise ValueError(f"Unknown cosmology label = {label}")
