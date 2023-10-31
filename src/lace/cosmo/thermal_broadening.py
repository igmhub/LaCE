import numpy as np

def thermal_broadening_kms(T_0):
    """Thermal broadening RMS in velocity units, given T_0"""

    sigma_T_kms=9.1 * np.sqrt(T_0/1.e4)
    return sigma_T_kms

def T0_from_broadening_kms(sigma_T_kms):
    """T0 given thermal broadening RMS in velocity units"""

    T_0 = 1.e4 * (sigma_T_kms/9.1)**2
    return T_0

