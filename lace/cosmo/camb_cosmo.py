"""Setup CAMB cosmology objects, and compute linear power and similar."""

import os
import numpy as np
import camb

# specify global settings to CAMB calls
camb_kmin_Mpc=1.e-4
camb_npoints=1000
camb_fluid=8
# no need to go beyond this k_Mpc when fitting linear power only
camb_fit_kmax_Mpc=1.5
# set kmax in transfer function beyond what you need (avoid warnings)
camb_extra_kmax=1.001

def get_cosmology(H0=67.0, mnu=0.0, omch2=0.12, ombh2=0.022, omk=0.0,
            As=2.1e-09, ns=0.965, nrun=0.0,pivot_scalar=0.05,w=-1):
    """Given set of cosmological parameters, return CAMB cosmology object."""

    pars = camb.CAMBparams()
    # set background cosmology
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, omk=omk, mnu=mnu)
    # set DE
    pars.set_dark_energy(w=w)
    # set primordial power
    pars.InitPower.set_params(As=As, ns=ns, nrun=nrun,pivot_scalar=pivot_scalar)

    return pars


def get_cosmology_from_dictionary(params,cosmo_fid=None):
    """Given a dictionary with parameters, return CAMB cosmology object.
        - cosmo_fid will be used for parameters not provided."""

    pars = camb.CAMBparams()
    # use default values for those not provided
    if cosmo_fid is None:
        cosmo_fid=get_cosmology()
    # collect background parameters
    if 'theta' in params: ## If theta is provided, override H0
        cosmomc_theta=params['theta']/100
        H0=None
    # collect background parameters
    elif 'cosmomc_theta' in params: ## If theta is provided, override H0
        cosmomc_theta=params['cosmomc_theta']
        H0=None
    elif 'H0' in params:
        H0=params['H0']
        cosmomc_theta=None
    else:
        H0=cosmo_fid.H0
        cosmomc_theta=None
    if 'ombh2' in params: ombh2=params['ombh2']
    elif "omegabh2" in params: ombh2=params['omegabh2']
    else: ombh2=cosmo_fid.ombh2
    if 'omch2' in params: omch2=params['omch2']
    elif "omegach2" in params: omch2=params['omegach2']
    else: omch2=cosmo_fid.omch2
    if 'omk' in params: omk=params['omk']
    else: omk=cosmo_fid.omk
    if 'mnu' in params: mnu=params['mnu']
    else: mnu=get_mnu(cosmo_fid)
    if 'tau' in params: tau=params["tau"]
    else: tau=cosmo_fid.Reion.optical_depth
    # update cosmology object
    pars.set_cosmology(H0=H0, cosmomc_theta=cosmomc_theta, ombh2=ombh2,
            omch2=omch2, omk=omk, mnu=mnu, tau=tau)
    
    # set DE
    if 'w' in params: w=params['w']
    else: w=cosmo_fid.DarkEnergy.w
    pars.set_dark_energy(w=w)

    # collect primorial power parameters
    if 'As' in params:
        As=params['As']
    elif "logA" in params:
        As=np.exp(params["logA"])/(10**10)
    else: As=cosmo_fid.InitPower.As
    if 'ns' in params: ns=params['ns']
    else: ns=cosmo_fid.InitPower.ns
    if 'nrun' in params: nrun=params['nrun']
    else: nrun=cosmo_fid.InitPower.nrun
    if 'pivot_scalar' in params: pivot_scalar=params['pivot_scalar']
    else: pivot_scalar=cosmo_fid.InitPower.pivot_scalar
    # update primordial power
    pars.InitPower.set_params(As=As, ns=ns, nrun=nrun,
                pivot_scalar=pivot_scalar)

    return pars


def get_mnu(pars):
    """Extract neutrino masses from CAMB object"""

    # eq 12 in https://archive.org/pdf/astro-ph/0603494.pdf
    return pars.omnuh2*93.14


def print_info(pars,simulation=False):
    """Given CAMB cosmology object, print relevant parameters"""

    if simulation:
        Omh2=(pars.omch2+pars.ombh2)
        Om=Omh2/(pars.H0/100.0)**2
        print('H0 = {:.4E}, Omega_bc = {:.4E}, A_s = {:.4E}, n_s = {:.4E}, alpha_s = {:.4E}'.format(pars.H0,Om,pars.InitPower.As,pars.InitPower.ns,pars.InitPower.nrun))
    else:
        print('H0 = {:.4E}, Omega_b h^2 = {:.4E}, Omega_c h^2 = {:.4E}, Omega_k = {:.4E}, Omega_nu h^2 = {:.4E}, A_s = {:.4E}, n_s = {:.4E}, alpha_s = {:.4E}'.format(pars.H0,pars.ombh2,pars.omch2,pars.omk,pars.omnuh2,pars.InitPower.As,pars.InitPower.ns,pars.InitPower.nrun))
    return


def set_fast_camb_options(pars):
    """Tune options in CAMB to speed-up evaluations"""

    pars.Want_CMB = False
    pars.WantDerivedParameters = False
    pars.WantCls = False
    pars.Want_CMB_lensing = False
    pars.DoLensing = False
    pars.Want_cl_2D_array = False
    pars.max_l_tensor = 0
    pars.max_l = 2
    pars.max_eta_k_tensor = 0
    pars.Reion.Reionization = False
    pars.Reion.ReionizationModel = False
    pars.Reion.include_helium_fullreion = False
    pars.Accuracy.AccuratePolarization = False
    pars.Accuracy.AccurateReionization = False
    pars.SourceTerms.counts_density = False
    pars.SourceTerms.limber_windows = False
    pars.SourceTerms.counts_timedelay = False
    pars.SourceTerms.counts_potential = False
    pars.SourceTerms.counts_ISW = False
    pars.SourceTerms.line_basic = False
    pars.SourceTerms.line_distortions = False
    pars.SourceTerms.use_21cm_mK = False


def get_camb_results(pars,zs=None,fast_camb=False,camb_kmax_Mpc=30):
    """Call camb.get_results, the slowest function in CAMB calls.
        - pars: CAMBparams object describing the cosmology
        - zs (optional): redshifts where we want to evaluate the linear power
        - fast_camb (optional): tune settings for fast computations
        - camb_kmax_Mpc (optional): maximum k to compute power spectrum"""

    if zs is not None:
        # check if we want to use fast version
        if fast_camb:
            set_fast_camb_options(pars)
            kmax_Mpc=camb_fit_kmax_Mpc
        else:
            kmax_Mpc=camb_kmax_Mpc

        # camb_extra_kmax will allow to evaluate the power later on at kmax
        pars.set_matter_power(redshifts=zs,kmax=camb_extra_kmax*kmax_Mpc,
                nonlinear=False,silent=True)

    return camb.get_results(pars)


def get_f_of_z(pars,camb_results,z):
    """Given pre-computed CAMB results, compute log growth rate at z"""

    # surprisingly, there is no option to specify the fluid (CDM+baryons?)
    transfer=camb_results.get_matter_transfer_data()
    # these seem to be in reverse order (see below)
    s8=transfer.sigma_8
    # note there is a bug in the CAMB documentation
    f_s8sq=transfer.sigma2_vdelta_8
    # compute logarithmic growth rate
    fz=f_s8sq/s8**2

    # return f(z) for input redshift
    zs=list(camb_results.transfer_redshifts)
    if z not in zs:
        raise ValueError('redshift {} not pre-computed'.format(z))
    iz=zs.index(z)
    return fz[iz]


def get_linP_hMpc(pars,zs,camb_results=None,fluid=camb_fluid):
    """Given a CAMB cosmology, and a set of redshifts, compute the linear
        power spectrum in units of h/Mpc. Other inputs:
        - camb_results: if provided, use that to speed things up.
        - fluid: specify transfer function to use (8=CDM+baryons). """

    if camb_results is None:
        # if you want to use fast_camb, you should pass camb_results
        camb_results = get_camb_results(pars,zs=zs,fast_camb=False)

    # make sure that all models are evaluated at the same points in 1/Mpc
    h=pars.H0/100.0
    kmin_hMpc=camb_kmin_Mpc/h
    # set_matter_power chose already kmax
    # (we use camb_extra_kmax=1.001 here to avoid warnings)
    kmax_hMpc=pars.Transfer.kmax/h/camb_extra_kmax

    # maxkh and npoints where we want to compute the power, in h/Mpc
    kh, zs_out, Ph = camb_results.get_matter_power_spectrum(var1=fluid,
            var2=fluid,npoints=camb_npoints,
            minkh=kmin_hMpc,maxkh=kmax_hMpc)

    ## So we need to make sure that every element in zs is contained in
    ## zs out
    check_zs =  all(item in zs_out for item in zs)
    assert check_zs==True, "You have asked for redshifts outside of the provided camb_results"

    ## Assuming this is ok, now we construct a list of the linear power at the
    ## zs we have been asked for, regardless of what comes out of
    ## camb_results
    P_out=np.empty((len(zs),len(kh)))
    for aa,zz in enumerate(zs):
        P_out[aa,:]=Ph[zs_out.index(zz)]
    #return kh, zs_out, Ph
    return kh, zs, P_out
    


def get_linP_Mpc(pars,zs,camb_results=None,fluid=camb_fluid):
    """Given a CAMB cosmology, and a set of redshifts, compute the linear
        power spectrum for CDM+baryons, in units of 1/Mpc. Other inputs:
        - camb_results: if provided, use that to speed things up.
        - fluid: specify transfer function to use (8=CDM+baryons). """

    # get linear power in units of Mpc/h
    k_hMpc, zs_out, P_hMpc = get_linP_hMpc(pars,zs=zs,
            camb_results=camb_results,fluid=fluid)
    # translate to Mpc
    h = pars.H0 / 100.0
    k_Mpc = k_hMpc * h
    P_Mpc = P_hMpc / h**3
    return k_Mpc, zs_out, P_Mpc


def get_linP_kms(pars,zs=[3],camb_results=None,fluid=camb_fluid):
    """Given a CAMB cosmology, and a set of redshifts, compute the linear
        power spectrum for CDM+baryons, in units of s/km.
        - camb_results: if provided, use that to speed things up.
        - fluid: specify transfer function to use (8=CDM+baryons). """

    # avoid calling twice to get_results
    if camb_results is None:
        # if you want to use fast_camb, you should pass camb_results
        camb_results = get_camb_results(pars,zs,fast_camb=False)

    # get linear power in units of Mpc/h
    k_hMpc, zs_out, P_hMpc = get_linP_hMpc(pars,zs,camb_results=camb_results,
            fluid=fluid)

    # each redshift will now have a different set of wavenumbers
    Nz=len(zs)
    Nk=len(k_hMpc)
    k_kms=np.empty([Nz,Nk])
    P_kms=np.empty([Nz,Nk])
    for iz in range(Nz):
        z = zs[iz]
        dvdX = dkms_dhMpc(pars,z,camb_results)
        k_kms[iz] = k_hMpc/dvdX
        P_kms[iz] = P_hMpc[iz]*dvdX**3

    return k_kms, zs_out, P_kms


def dkms_dMpc(cosmo,z,camb_results=None):
    """Compute factor to translate velocity separations (in km/s) to comoving
        separations (in Mpc). At z=3 it should return roughly 70.
    Inputs:
        - cosmo: CAMBparams object.
        - z: redshift
        - camb_results (optional): CAMBdata object, avoid calling get_results
    """

    h=cosmo.H0/100.0
    return h*dkms_dhMpc(cosmo,z,camb_results=camb_results)


def dkms_dhMpc(cosmo,z,camb_results=None):
    """Compute factor to translate velocity separations (in km/s) to comoving
        separations (in Mpc/h). At z=3 it should return roughly 100.
    Inputs:
        - cosmo: CAMBparams object.
        - z: redshift
        - camb_results (optional): CAMBdata object, avoid calling get_results
    """

    if camb_results is None:
        camb_results = camb.get_results(cosmo)

    H_z=camb_results.hubble_parameter(z)
    dvdX=H_z/(1+z)/(cosmo.H0/100.0)
    return dvdX


def shift_primordial_pivot(cosmo_dict,pivot_scalar):
    """ Shift the value of A_s calculated at a new pivot point.
        It used to be used in test_simulation if using a different pivot
        than stored in JSON, but it is now deprecated.
        Currently assumes zero running """

    # some old sims did not store pivot scalar (was 0.05 by default)
    if "pivot_scalar" in cosmo_dict.keys():
        pivot_old=cosmo_dict["pivot_scalar"]
    else:
        pivot_old=0.05

    if pivot_scalar==pivot_old:
        return cosmo_dict
    else:
        print('re-scale primordial power; old / new = {} / {} 1/Mpc'.format(
                    pivot_old,pivot_scalar))

    # make sure we do not have running in dictionary!
    assert "nrun" not in cosmo_dict,"update shift_primordial_pivot"

    # compute new amplitude of primordial power
    new_As=cosmo_dict["As"]*(pivot_scalar/pivot_old)**(cosmo_dict["ns"]-1)

    ## Update dictionary with new As and pivot
    cosmo_dict["As"]=new_As
    cosmo_dict["pivot_scalar"]=pivot_scalar

    return cosmo_dict


def get_Nyx_cosmology(params):
    """Given a dictionary with Nyx sim params, return CAMB cosmology."""

    # baryon density (from Walther, private communication)
    ombh2 = 0.02233
    H0=params['H_0']
    ommh2=params['omega_m']
    omch2=ommh2-ombh2

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=0.0)

    # collect primorial power parameters
    As=params['A_s']
    ns=params['n_s']
    # update primordial power
    pars.InitPower.set_params(As=As, ns=ns)

    return pars
