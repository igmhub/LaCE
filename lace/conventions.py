"""Scientific naming and array conventions shared by LaCE interfaces."""

from __future__ import annotations

from typing import Any

import numpy as np

UNIT_SUFFIXES = {
    "iMpc": "Mpc^-1",
    "Mpc": "Mpc (or the corresponding positive power for a spectrum)",
    "ikms": "(km/s)^-1 = s/km",
    "kms": "km/s (or the corresponding positive power)",
}
LEGACY_UNIT_KEYS = {
    "k_Mpc": "k_iMpc",
    "k_kms": "k_ikms",
    "p1d_Mpc": "P1D_Mpc",
    "p3d_Mpc": "P3D_Mpc",
    "Pk_kms": "P1D_kms",
    "dkms_dMpc": "dkms_diMpc",
}


def canonicalize_unit_keys(values: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy with legacy scientific keys made canonical.

    Parameters
    ----------
    values : dict
        Mapping that may contain legacy serialized science keys.

    Returns
    -------
    dict
        Shallow copy retaining all original keys and adding canonical aliases
        only where the canonical key was absent.
    """
    result = dict(values)
    for old, new in LEGACY_UNIT_KEYS.items():
        if new not in result and old in result:
            result[new] = result[old]
    return result


def validate_wavenumber(values: Any, *, name: str) -> np.ndarray:
    """Return a finite, positive, one-dimensional wavenumber array.

    Parameters
    ----------
    values : array-like
        Wavenumbers in the units implied by ``name``.
    name : str
        Field name included in validation errors.

    Returns
    -------
    ndarray
        One-dimensional floating-point array.

    Raises
    ------
    ValueError
        If the input is not one-dimensional, finite, and strictly positive.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise ValueError(f"{name} must be a finite, positive 1D array")
    return array
