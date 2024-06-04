"""Read GenIC configuration file. File addapted from code by Simeon Bird."""

import numpy as np
import argparse
import configobj
import validate

# define variables and default values in GenIC
GenICconfigspec = """
Omega0 = float(0,1)
OmegaLambda = float(0,1)
OmegaBaryon = float(0,1,default=0.0486)
HubbleParam = float(0,2)
BoxSize = float(0,10000000)
Sigma8 = float(default=-1)
InputPowerRedshift = float(default=-1)
DifferentTransferFunctions = integer(0,1, default=1)
Omega_fld = float(0,1,default=0)
w0_fld = float(default=-1)
wa_fld = float(default=0)
MNue = float(min=0, default=0)
MNum = float(min=0, default=0)
MNut = float(min=0, default=0)
MWDM_Therm = float(min=0, default=0)
PrimordialIndex = float(default=0.971)
PrimordialAmp = float(default=2.215e-9)
PrimordialRunning = float(default=0.0)""".split(
    "\n"
)


def L_Mpc_from_paramfile(paramfile, verbose=False):
    config = read_genic_paramfile(paramfile, verbose)
    L_hkpc = config["BoxSize"]
    h = config["HubbleParam"]
    L_Mpc = L_hkpc / 1000 / h
    if verbose:
        print("in L_Mpc,", L_hkpc, h, L_Mpc)
    return L_Mpc


def read_genic_paramfile(paramfile, verbose=False):
    """Parse a GenIC parameter file and returns a dictionary"""

    config = configobj.ConfigObj(
        infile=paramfile, configspec=GenICconfigspec, file_error=True
    )
    # check file is healthy
    _check_genic_config(config)
    if verbose:
        print("successfully read healthy configuration file")
    return config


def _check_genic_config(config):
    """Check that the MP-GenIC config file is sensible."""
    vtor = validate.Validator()
    config.validate(vtor)
    # Check unsupported configurations
    if config["DifferentTransferFunctions"] == 0.0:
        raise ValueError("Can only work with different transfer functions.")
    if config["InputPowerRedshift"] > 0:
        raise ValueError("Can not specify input redshift")
    if config["Sigma8"] > 0.0:
        raise ValueError("Can not specify Sigma8.")
    if not (config["wa_fld"] == 0.0):
        raise ValueError("wa_fld != 0 not supported.")
    if config["MWDM_Therm"] > 0:
        raise ValueError("Warm dark matter cutoff not yet supported.")


def _build_cosmology_params_class(config):
    """Build a correctly-named-for-class set of cosmology parameters."""
    # Class takes omega_m h^2 as parameters
    h0 = config["HubbleParam"]
    # Compute sum of neutrino masses
    omeganu = (
        (config["MNue"] + config["MNum"] + config["MNut"]) / 93.14 / h0**2
    )
    omega0 = config["Omega0"]
    omegaL = config["OmegaLambda"]
    omegab = config["OmegaBaryon"]
    omegacdm = omega0 - omegab - omeganu
    omegak = 1 - omegaL - omega0
    params = {
        "h": h0,
        "Omega_cdm": omegacdm,
        "Omega_b": omegab,
        "Omega_k": omegak,
    }
    params["A_s"] = config["PrimordialAmp"]
    params["n_s"] = config["PrimordialIndex"]
    params["alpha_s"] = config["PrimordialRunning"]
    # Set up massive neutrinos
    if omeganu > 0:
        params["m_ncdm"] = "%.8f,%.8f,%.8f" % (
            config["MNue"],
            config["MNum"],
            config["MNut"],
        )
        params["N_ncdm"] = 3
        params["N_ur"] = 0.00641
    else:
        params["N_ur"] = 3.046

    return params


def _build_cosmology_params_camb(config):
    """Build a correctly-named-for-camb set of cosmology parameters."""
    # Class takes omega_m h^2 as parameters
    h0 = config["HubbleParam"]
    # Compute sum of neutrino masses
    mnu = config["MNue"] + config["MNum"] + config["MNut"]
    omeganu = mnu / 93.14 / h0**2
    omega0 = config["Omega0"]
    omegaL = config["OmegaLambda"]
    omegab = config["OmegaBaryon"]
    omegacdm = omega0 - omegab - omeganu
    omegak = 1 - omegaL - omega0

    params = {"H0": 100.0 * h0}
    params["omch2"] = omegacdm * h0**2
    params["ombh2"] = omegab * h0**2
    params["mnu"] = mnu
    params["omk"] = omegak
    params["As"] = config["PrimordialAmp"]
    params["ns"] = config["PrimordialIndex"]
    params["nrun"] = config["PrimordialRunning"]
    params["w"] = config["w0_fld"]

    return params


def class_from_genic(paramfile, verbose=False):
    """Parse a GenIC parameter file and returns a dictionary to setup CLASS"""

    # read GenIC configuration file, and store information
    config = read_genic_paramfile(paramfile, verbose)

    # rename parameters to be used in CLASS
    params = _build_cosmology_params_class(config)
    if verbose:
        print("translated parameters to CLASS format")

    if verbose:
        print("params", params)

    return params


def camb_from_genic(paramfile, verbose=False):
    """Parse a GenIC parameter file and returns a dictionary to setup CAMB"""

    # read GenIC configuration file, and store information
    config = read_genic_paramfile(paramfile, verbose)

    # rename parameters to be used in CAMB
    params = _build_cosmology_params_camb(config)
    if verbose:
        print("translated parameters to CAMB format")

    if verbose:
        print("params", params)

    return params
