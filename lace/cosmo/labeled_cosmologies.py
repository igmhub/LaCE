import numpy as np

#: Cosmology labels defined in LaCE. Other labels are looked up in cup1d.
COSMOLOGY_LABELS = ("Planck18", "Planck18_mnu", "Planck18_noBAO")


def get_cosmo_params_dict_from_label(label):
    """Return a mutable cosmology parameter dictionary for a named fiducial.

    Parameters
    ----------
    label : str
        ``"Planck18"``, ``"Planck18_mnu"`` or ``"Planck18_noBAO"``. Any other
        label is looked up in the cosmologies defined by cup1d (simulation
        suites such as ``"accel2"`` or ``"mpg_central"``, and reference
        variations), when cup1d is installed; see
        :func:`get_cosmo_params_dict_from_cup1d`.

    Returns
    -------
    dict
        CAMB/LaCE parameter values in physical units.

    Raises
    ------
    ValueError
        If ``label`` is defined neither in LaCE nor in cup1d.

    Notes
    -----
    LaCE labels take precedence over cup1d labels with the same name; e.g.
    ``"Planck18"`` always returns the LaCE definition.
    """
    if label == "Planck18":
        # Table 2 of https://arxiv.org/abs/1807.06209.pdf (T&E+lensing+BAO) w/ mnu=0
        # Used as baseline cosmology in the DESI Lya P1D analysis
        cosmo_params = {
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
        }
    elif label == "Planck18_mnu":
        # Table 2 of https://arxiv.org/abs/1807.06209.pdf (T&E+lensing+BAO) w/ mnu=0.06
        # Used in on the of the variations of the DESI Lya P1D analysis
        cosmo_params = {
            "H0": 67.66,
            "mnu": 0.06,
            "omch2": 0.11933,
            "ombh2": 0.02242,
            "omk": 0,
            "As": np.exp(3.047) * 1e-10,
            "ns": 0.9665,
            "nrun": 0.0,
            "pivot_scalar": 0.05,
            "w": -1.0,
        }
    elif label == "Planck18_noBAO":
        # Table 2 of https://arxiv.org/abs/1807.06209.pdf (T&E+lensing) w/ mnu=0.06
        # Used as baseline cosmology in the DESI Lya BAO analysis
        cosmo_params = {
            "H0": 67.36,
            "mnu": 0.06,
            "omch2": 0.1200,
            "ombh2": 0.02237,
            "omk": 0,
            "As": np.exp(3.044) * 1e-10,
            "ns": 0.9646,
            "nrun": 0.0,
            "pivot_scalar": 0.05,
            "w": -1.0,
        }
    else:
        cosmo_params = get_cosmo_params_dict_from_cup1d(label)

    return cosmo_params


def get_cosmo_params_dict_from_cup1d(label):
    """Return the parameters of a cosmology defined in cup1d.

    cup1d is an optional, downstream package: it is imported only here, and
    LaCE does not require it.

    Parameters
    ----------
    label : str
        Cosmology label accepted by ``cup1d.theory.cosmology.set_cosmo``.

    Returns
    -------
    dict
        CAMB/LaCE parameter values in physical units.

    Raises
    ------
    ValueError
        If cup1d is not installed or does not define ``label``.
    """
    try:
        from cup1d.theory.cosmology import set_cosmo
    except ImportError as err:
        raise ValueError(
            f"Unknown cosmology label = {label} (not defined in LaCE, and "
            "cup1d, which defines additional labels, is not installed)"
        ) from err

    try:
        # constructing a Cosmology does not run CAMB
        cosmo = set_cosmo(cosmo_label=label)
    except ValueError as err:
        raise ValueError(
            f"Unknown cosmology label = {label} (defined neither in LaCE nor "
            f"in cup1d: {err})"
        ) from err

    return dict(cosmo.input_cosmo_params_dict)
