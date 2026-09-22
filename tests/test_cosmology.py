"""Regression tests for LaCE's public cosmology interfaces."""

import numpy as np

from lace.cosmo import cosmology, rescale_cosmology


REDSHIFT = 2.33


def test_default_cosmology_growth_and_linear_power():
    """The default cosmology reproduces the Tutorial_cosmo reference values."""

    cosmo = cosmology.Cosmology()

    np.testing.assert_allclose(
        cosmo.get_sigma8(REDSHIFT), 0.31132737031617397, rtol=1.0e-6
    )
    np.testing.assert_allclose(
        cosmo.get_growth_rate(REDSHIFT), 0.9679356260858405, rtol=1.0e-6
    )

    k_Mpc = np.array([0.1, 0.7, 1.0])
    expected_linear_power = np.array([1529.1168512, 29.59981873, 12.83345793])
    np.testing.assert_allclose(
        cosmo.get_linP_Mpc(z=REDSHIFT, k_Mpc=k_Mpc),
        expected_linear_power,
        rtol=1.0e-6,
    )


def test_rescaled_cosmology_applies_primordial_tilt():
    """A fixed-background ``ns`` change rescales linear power as expected."""

    fiducial_cosmology = cosmology.Cosmology()
    rescaled_cosmology = rescale_cosmology.RescaledCosmology(
        fid_cosmo=fiducial_cosmology, new_params_dict={"ns": 0.96}
    )
    k_Mpc = np.array([0.1, 0.7, 1.0])

    fiducial_power = fiducial_cosmology.get_linP_Mpc(z=REDSHIFT, k_Mpc=k_Mpc)
    rescaled_power = rescaled_cosmology.get_linP_Mpc(z=REDSHIFT, k_Mpc=k_Mpc)
    pivot_Mpc = fiducial_cosmology.CAMBparams.InitPower.pivot_scalar
    fiducial_ns = fiducial_cosmology.CAMBparams.InitPower.ns
    expected_scaling = (k_Mpc / pivot_Mpc) ** (0.96 - fiducial_ns)

    np.testing.assert_allclose(
        rescaled_power, fiducial_power * expected_scaling, rtol=1.0e-6
    )
    np.testing.assert_allclose(
        rescaled_cosmology.get_growth_rate(REDSHIFT),
        fiducial_cosmology.get_growth_rate(REDSHIFT),
        rtol=1.0e-6,
    )
