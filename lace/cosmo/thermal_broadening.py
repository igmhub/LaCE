import numpy as np


def thermal_broadening_kms(T_0):
    """Convert gas temperature to thermal broadening RMS.

    Parameters
    ----------
    T_0 : float or array-like
        Gas temperature in K.

    Returns
    -------
    float or ndarray
        Thermal RMS width in km/s.
    """

    sigma_T_kms = 9.1 * np.sqrt(T_0 / 1.0e4)
    return sigma_T_kms


def T0_from_broadening_kms(sigma_T_kms):
    """Convert a thermal broadening RMS to gas temperature.

    Parameters
    ----------
    sigma_T_kms : float or array-like
        Thermal RMS width in km/s.

    Returns
    -------
    float or ndarray
        Gas temperature in K.
    """

    T_0 = 1.0e4 * (sigma_T_kms / 9.1) ** 2
    return T_0
