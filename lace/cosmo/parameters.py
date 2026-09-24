"""Utilities for normalizing cosmology parameter dictionaries."""

from collections.abc import Mapping

import numpy as np


def normalize_cosmology_params(params):
    """Return a canonical copy of a cosmology parameter dictionary.

    The accepted aliases match the historical CAMB input interface, while the
    returned dictionary uses the names consumed by :class:`Cosmology`.
    """

    if not isinstance(params, Mapping):
        raise TypeError("Cosmology parameters must be provided as a mapping")

    normalized = dict(params)

    aliases = {
        "omegabh2": "ombh2",
        "omegach2": "omch2",
        "omegak": "omk",
        "theta_MC_100": "theta_MC_100",
    }
    for alias, canonical in aliases.items():
        if alias in normalized and canonical not in normalized:
            normalized[canonical] = normalized[alias]
        if alias != canonical:
            normalized.pop(alias, None)

    if "logA" in normalized and "As" not in normalized:
        normalized["As"] = np.exp(normalized["logA"]) / 1e10
    normalized.pop("logA", None)

    theta_names = ("theta", "cosmomc_theta", "theta_MC_100")
    provided_theta_names = [name for name in theta_names if name in normalized]
    if len(provided_theta_names) > 1:
        raise ValueError(
            "Provide only one of theta, cosmomc_theta, or theta_MC_100"
        )
    if provided_theta_names and "H0" in normalized:
        raise ValueError("H0 cannot be provided together with a theta parameter")

    if "theta" in normalized:
        normalized["cosmomc_theta"] = normalized.pop("theta") / 100.0
    elif "theta_MC_100" in normalized:
        normalized["cosmomc_theta"] = normalized.pop("theta_MC_100") / 100.0

    return normalized
