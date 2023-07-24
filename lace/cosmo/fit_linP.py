import numpy as np
import os
import camb
from lace.cosmo import camb_cosmo
from lace.utils.poly_p1d import fit_polynomial


def get_linP_Mpc_zs(cosmo, zs, kp_Mpc):
    """For each redshift, obtain and fit linear power parameters around kp_Mpc"""

    # run slowest part of CAMB computation, to avoid repetition
    camb_results = camb_cosmo.get_camb_results(cosmo, zs, fast_camb=True)

    # compute linear power at all zs
    k_Mpc, zs_out, P_Mpc = camb_cosmo.get_linP_Mpc(cosmo, zs, camb_results)

    fp = np.empty(len(zs))
    for iz,z in enumerate(zs):
        # check that CAMB did not reverse the order of redshifts
        assert z==zs_out[iz],'redshifts not sorted out correctly'
        # compute logarithmic growth rate at each z (f = f sigma_8 / sigma_8)
        fp[iz]=camb_cosmo.get_f_of_z(cosmo,camb_results,z)

    return fit_linP_Mpc_zs(k_Mpc, P_Mpc, fp, kp_Mpc, zs)

def fit_linP_Mpc_zs(k_Mpc, P_Mpc, fp, kp_Mpc, zs):
    """For each redshift, only fit linear power parameters around kp_Mpc"""

    assert kp_Mpc < max(k_Mpc), "Pivot higher than k_max"

    # specify wavenumber range to fit
    kmin_Mpc = 0.5 * kp_Mpc
    kmax_Mpc = 2.0 * kp_Mpc

    # loop over all redshifts, and collect linP parameters
    linP_zs = []
    for iz,z in enumerate(zs):
        # fit polynomial of log power over wavenumber range
        linP_Mpc = fit_polynomial(
            kmin_Mpc / kp_Mpc, kmax_Mpc / kp_Mpc, k_Mpc / kp_Mpc, P_Mpc[iz]
        )
        # translate the polynomial to our parameters
        ln_A_p = linP_Mpc[0]
        Delta2_p = np.exp(ln_A_p) * kp_Mpc**3 / (2 * np.pi**2)
        n_p = linP_Mpc[1]
        # note that the curvature is alpha/2
        alpha_p = 2.0 * linP_Mpc[2]

        linP_z = {
            "Delta2_p": Delta2_p,
            "n_p": n_p,
            "alpha_p": alpha_p,
            "f_p": fp[iz],
        }

        linP_zs.append(linP_z)

    return linP_zs


def compute_gz(cosmo, z_star, camb_results=None):
    """Compute logarithmic derivative of Hubble expansion, normalized to EdS:
    g(z) = dln H(z) / dln(1+z)^3/2 = 2/3 (1+z)/H(z) dH/dz"""

    if camb_results is None:
        camb_results = camb_cosmo.get_camb_results(cosmo)

    # compute derivative of Hubble
    dz = z_star / 100.0
    z_minus = z_star - dz
    z_plus = z_star + dz
    H_minus = camb_results.hubble_parameter(z=z_minus)
    H_plus = camb_results.hubble_parameter(z=z_plus)
    dHdz = (H_plus - H_minus) / (z_plus - z_minus)
    # compute hubble at z, and return g(z)
    Hz = camb_results.hubble_parameter(z=z_star)
    gz = dHdz / Hz * (1 + z_star) * 2 / 3
    return gz


def compute_fz(cosmo, z, kp_Mpc):
    """Given cosmology, compute logarithmic growth rate (f) at z, around
    pivot point k_p (in 1/Mpc):
    f(z) = d lnD / d lna = - 1/2 * (1+z)/P(z) dP/dz"""

    # will compute derivative of linear power at z
    dz = z / 100.0
    zs = [z + dz, z, z - dz]

    # slowest part to run
    camb_results = camb_cosmo.get_camb_results(cosmo, zs=zs, fast_camb=True)
    k_Mpc, zs_out, P_Mpc = camb_cosmo.get_linP_Mpc(cosmo, zs, camb_results)

    # compute derivative of linear power
    z_minus = zs_out[0]
    z = zs_out[1]
    z_plus = zs_out[2]
    P_minus = P_Mpc[0]
    Pz = P_Mpc[1]
    P_plus = P_Mpc[2]
    dPdz = (P_plus - P_minus) / (z_plus - z_minus)

    # compute logarithmic growth rate
    fz_k = -0.5 * dPdz / Pz * (1 + z)
    # compute mean around k_p
    mask = (k_Mpc > 0.5 * kp_Mpc) & (k_Mpc < 2.0 * kp_Mpc)
    fz = np.mean(fz_k[mask])

    return fz


def fit_linP_kms(
    cosmo, z_star, kp_kms, deg=2, camb_results=None, fit_min=0.5, fit_max=2.0
):
    """Given input cosmology, compute linear power at z_star
    (in km/s) and fit polynomial around kp_kms.
    - camb_results optional to avoid calling get_results."""

    k_kms = np.logspace(np.log10(0.5 * kp_kms), np.log10(2.0 * kp_kms), 100)

    if camb_results == None:
        zs = [z_star]
        camb_results = camb_cosmo.get_camb_results(cosmo, zs=zs, fast_camb=True)

    assert z_star in list(camb_results.transfer_redshifts), (
        "Transfer function " "not calculated for z_star in camb_results"
    )

    ## Get conversion factor from velocity units to comoving
    H_z = camb_results.hubble_parameter(z_star)
    dvdX = H_z / (1 + z_star) / (cosmo.H0 / 100.0)
    k_hMpc = k_kms * dvdX

    camb_interp = camb_results.get_matter_power_interpolator(
        var1=8, var2=8, nonlinear=False
    )

    k_hMpc = dvdX * k_kms
    P_hMpc = camb_interp.P(z_star, k_hMpc)
    P_kms = P_hMpc * (dvdX**3)

    # specify wavenumber range to fit
    kmin_kms = fit_min * kp_kms
    kmax_kms = fit_max * kp_kms
    # compute ratio
    P_fit = fit_polynomial(
        kmin_kms / kp_kms, kmax_kms / kp_kms, k_kms / kp_kms, P_kms, deg=deg
    )

    return P_fit


def parameterize_cosmology_kms(
    cosmo, camb_results, z_star, kp_kms, fit_min=0.5, fit_max=2.0
):
    """Given input cosmology, compute set of parameters that describe
    the linear power around z_star and wavenumbers kp_kms"""

    # call get_results first, to avoid calling it twice
    if camb_results == None:
        zs = [z_star]
        camb_results = camb_cosmo.get_camb_results(cosmo, zs=zs, fast_camb=True)

    # compute linear power, in km/s, at z_star
    # and fit a second order polynomial to the log power, around kp_kms
    linP_kms = fit_linP_kms(
        cosmo,
        z_star,
        kp_kms,
        deg=2,
        camb_results=camb_results,
        fit_min=fit_min,
        fit_max=fit_max,
    )

    # translate the polynomial to our parameters
    ln_A_star = linP_kms[0]
    Delta2_star = np.exp(ln_A_star) * kp_kms**3 / (2 * np.pi**2)
    n_star = linP_kms[1]
    # note that the curvature is alpha/2
    alpha_star = 2.0 * linP_kms[2]

    # get logarithmic growth rate at z_star (from f sigma_8 / sigma_8 at z)
    f_star = camb_cosmo.get_f_of_z(cosmo, camb_results, z_star)

    # compute deviation from EdS expansion
    g_star = compute_gz(cosmo, z_star, camb_results=camb_results)

    results = {
        "f_star": f_star,
        "g_star": g_star,
        "linP_kms": linP_kms,
        "Delta2_star": Delta2_star,
        "n_star": n_star,
        "alpha_star": alpha_star,
    }

    return results
