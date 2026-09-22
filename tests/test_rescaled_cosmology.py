"""Compare rescaled and fresh-CAMB star parameters.

The star parameters are the dimensionless compressed linear-power amplitude,
slope, and curvature used by cup1d at z_star=3 and kp_kms=0.009 s/km.
"""

import numpy as np
import pytest

from lace.cosmo.cosmology import Cosmology
from lace.cosmo.rescale_cosmology import RescaledCosmology


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
