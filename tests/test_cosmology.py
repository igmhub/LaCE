"""Regression tests for LaCE's public cosmology interfaces."""

import numpy as np

from lace.cosmo import cosmology


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
