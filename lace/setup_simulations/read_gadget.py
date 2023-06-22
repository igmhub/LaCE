"""Read MP-Gadget configuration file. File addapted from code by Simeon Bird."""

import numpy as np
import argparse
import configobj
import validate

# define variables and default values in Gadget
Gadget_configspec = """
OutputList = string(default='')
Nmesh = integer(default=1)
    Omega0 = float(0,1)
    OmegaLambda = float(0,1)
    OmegaBaryon = float(0,1,default=0.0486)
    HubbleParam = float(0,2)
    MNue = float(min=0, default=0)
    MNum = float(min=0, default=0)
    MNut = float(min=0, default=0)""".split(
    "\n"
)


def _check_gadget_config(config):
    """Check that the MP-Gadget config file is sensible."""
    vtor = validate.Validator()
    config.validate(vtor)
    # Check unsupported configurations
    # if config['MNue']+config['MNum']+config['MNut'] > 0.:
    # print("Sim has neutrinos")


def read_gadget_paramfile(paramfile, verbose=False):
    """Parse a MP-Gadget parameter file and returns a dictionary"""

    config = configobj.ConfigObj(
        infile=paramfile, configspec=Gadget_configspec, file_error=True
    )
    # check file is healthy
    _check_gadget_config(config)
    if verbose:
        print("successfully read healthy configuration file")
    return config


def snapshot_redshifts(config):
    scale_factors = [float(astr) for astr in config["OutputList"].split(",")]
    # add last snapshot, when simulation ends
    scale_factors.append(float(config["TimeMax"]))
    zs = 1.0 / np.array(scale_factors) - 1.0
    return zs


def redshifts_from_paramfile(paramfile, verbose=False):
    config = read_gadget_paramfile(paramfile, verbose)
    return snapshot_redshifts(config)


def _build_cosmology_params_class(config):
    """Build a correctly-named-for-class set of cosmology parameters."""
    # Class takes omega_m h^2 as parameters
    h0 = config["HubbleParam"]
    # Compute sum of neutrino masses
    omeganu = (config["MNue"] + config["MNum"] + config["MNut"]) / 93.14 / h0**2
    omega0 = config["Omega0"]
    omegaL = config["OmegaLambda"]
    omegab = config["OmegaBaryon"]
    omegacdm = omega0 - omegab - omeganu
    omegak = 1 - omegaL - omega0
    params = {"h": h0, "Omega_cdm": omegacdm, "Omega_b": omegab, "Omega_k": omegak}
    # these are not present in Gadget file
    # params['A_s'] = config["PrimordialAmp"]
    # params['n_s'] = config['PrimordialIndex']
    # params['alpha_s'] = config['PrimordialRunning']
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
    if "w0_fld" in config:
        params["w"] = config["w0_fld"]
    else:
        print("assume w=-1")
        params["w"] = -1
    # these are not in Gadget file
    # params['TCMB'] = config["CMBTemperature"]
    # params['As'] = config["PrimordialAmp"]
    # params['ns'] = config['PrimordialIndex']
    # params['nrun'] = config['PrimordialRunning']

    return params


def class_from_gadget(paramfile, verbose=False):
    """Parse a Gadget parameter file and returns a dictionary to setup CLASS"""

    # read Gadget configuration file, and store information
    config = configobj.ConfigObj(
        infile=paramfile, configspec=Gadget_configspec, file_error=True
    )
    # check file is healthy
    _check_gadget_config(config)
    if verbose:
        print("successfully read healthy configuration file")

    # rename parameters to be used in CLASS
    params = _build_cosmology_params_class(config)
    if verbose:
        print("translated parameters to CLASS format")

    if verbose:
        print("params", params)

    return params


def camb_from_gadget(paramfile, verbose=False):
    """Parse a Gadget parameter file and returns a dictionary to setup CAMB"""

    # read Gadget configuration file, and store information
    config = configobj.ConfigObj(
        infile=paramfile, configspec=Gadget_configspec, file_error=True
    )
    # check file is healthy
    _check_gadget_config(config)
    if verbose:
        print("successfully read healthy configuration file")

    # rename parameters to be used in CAMB
    params = _build_cosmology_params_camb(config)
    if verbose:
        print("translated parameters to CAMB format")

    if verbose:
        print("params", params)

    return params
