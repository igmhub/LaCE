"""Compare rescaled and fresh-CAMB star parameters.

The star parameters are the dimensionless compressed linear-power amplitude,
slope, and curvature used by cup1d at z_star=3 and kp_kms=0.009 s/km.
"""

import numpy as np
import pytest

from lace.cosmo.cosmology import Cosmology
from lace.cosmo.rescale_cosmology import (
    IncompatibleBackgroundError,
    RescaledCosmology,
)


Z_STAR = 3.0
KP_KMS = 0.009


@pytest.fixture(scope="module")
def fiducial_cosmology():
    return Cosmology()


@pytest.mark.parametrize(
    "changed_params",
    [
        {"As": 2.2e-9},
        {"ns": 0.96},
        {"nrun": -0.01},
    ],
    ids=["As", "ns", "nrun"],
)
def test_rescaled_star_parameters_match_fresh_camb(
    fiducial_cosmology, changed_params
):
    """Fixed-background rescaling matches fresh CAMB star parameters."""

    rescaled = RescaledCosmology(fiducial_cosmology, changed_params)
    fresh = Cosmology(cosmo_params_dict=changed_params)

    # RescaledCosmology and fresh CAMB predict the same cup1d inputs.
    for key in ("Delta2_star", "n_star", "alpha_star"):
        np.testing.assert_allclose(
            rescaled.get_linP_kms_params(Z_STAR, KP_KMS)[key],
            fresh.get_linP_kms_params(Z_STAR, KP_KMS)[key],
            rtol=1e-4,
            atol=1e-6,
        )


def test_changed_background_reports_parameter_values(fiducial_cosmology):
    """An incompatible background identifies every changed value."""

    changes = {"H0": 74.0, "omch2": 0.13}
    with pytest.raises(IncompatibleBackgroundError) as caught:
        RescaledCosmology(fiducial_cosmology, changes)

    error = caught.value
    assert set(error.changes) == set(changes)
    for name, requested_value in changes.items():
        fiducial_value, stored_requested_value = error.changes[name]
        assert fiducial_value == fiducial_cosmology.get_background_params()[name]
        assert stored_requested_value == requested_value
        assert name in str(error)
        assert repr(fiducial_value) in str(error)
        assert repr(requested_value) in str(error)


def test_nonstandard_pivot_is_rejected():
    """The current rescaling convention explicitly requires k_s=0.05/Mpc."""

    fiducial = Cosmology()
    fiducial.CAMBparams.InitPower.pivot_scalar = 0.04
    with pytest.raises(ValueError, match="pivot_scalar=0.05"):
        RescaledCosmology(fiducial, {"ns": 0.96})
