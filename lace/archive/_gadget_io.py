"""Read MP-Gadget and MP-GenIC parameter files of the Gadget simulations.

Used by :class:`lace.archive.gadget_archive.GadgetArchive` to recompute the
linear-power parameters of a simulation. Adapted from code by Simeon Bird.
"""

import configobj
import numpy as np
import validate

# variables and default values in MP-Gadget files
GADGET_CONFIGSPEC = """
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

# variables and default values in MP-GenIC files
GENIC_CONFIGSPEC = """
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


def read_gadget_paramfile(paramfile, verbose=False):
    """Parse and validate an MP-Gadget parameter file.

    Parameters
    ----------
    paramfile : path-like
        MP-Gadget configuration file.
    verbose : bool, default=False
        Print successful validation diagnostics.

    Returns
    -------
    configobj.ConfigObj
        Parsed configuration with configspec defaults applied.
    """
    config = configobj.ConfigObj(
        infile=paramfile, configspec=GADGET_CONFIGSPEC, file_error=True
    )
    config.validate(validate.Validator())
    if verbose:
        print("successfully read healthy configuration file")
    return config


def snapshot_redshifts(config):
    """Return scheduled and final MP-Gadget output redshifts.

    Parameters
    ----------
    config : configobj.ConfigObj
        Validated configuration containing scale-factor output times.

    Returns
    -------
    ndarray
        Redshifts converted from ``OutputList`` plus ``TimeMax``.
    """
    scale_factors = [float(astr) for astr in config["OutputList"].split(",")]
    # add last snapshot, when simulation ends
    scale_factors.append(float(config["TimeMax"]))
    return 1.0 / np.array(scale_factors) - 1.0


def read_genic_paramfile(paramfile, verbose=False):
    """Parse and validate an MP-GenIC initial-condition parameter file.

    Parameters
    ----------
    paramfile : path-like
        GenIC configuration file.
    verbose : bool, default=False
        Print successful validation diagnostics.

    Returns
    -------
    configobj.ConfigObj
        Parsed configuration with configspec defaults applied.

    Raises
    ------
    ValueError
        If unsupported common transfer functions, input-redshift normalization,
        sigma-eight normalization, evolving dark energy, or warm dark matter
        is requested.
    """
    config = configobj.ConfigObj(
        infile=paramfile, configspec=GENIC_CONFIGSPEC, file_error=True
    )
    config.validate(validate.Validator())
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
    if verbose:
        print("successfully read healthy configuration file")
    return config


def camb_from_genic(paramfile, verbose=False):
    """Parse an MP-GenIC file and return a CAMB-ready parameter mapping.

    Parameters
    ----------
    paramfile : path-like
        GenIC configuration file.
    verbose : bool, default=False
        Print parsing diagnostics.

    Returns
    -------
    dict
        ``H0`` in km/s/Mpc, physical densities, total neutrino mass in eV,
        curvature, primordial parameters and dark-energy equation of state.
    """
    config = read_genic_paramfile(paramfile, verbose)

    h0 = config["HubbleParam"]
    mnu = config["MNue"] + config["MNum"] + config["MNut"]
    omeganu = mnu / 93.14 / h0**2
    omega0 = config["Omega0"]
    omegab = config["OmegaBaryon"]
    omegacdm = omega0 - omegab - omeganu
    omegak = 1 - config["OmegaLambda"] - omega0

    params = {
        "H0": 100.0 * h0,
        "omch2": omegacdm * h0**2,
        "ombh2": omegab * h0**2,
        "mnu": mnu,
        "omk": omegak,
        "As": config["PrimordialAmp"],
        "ns": config["PrimordialIndex"],
        "nrun": config["PrimordialRunning"],
        "w": config["w0_fld"],
    }
    if verbose:
        print("params", params)
    return params
