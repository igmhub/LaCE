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
        {"nrunrun": 0.002},
    ],
    ids=["As", "ns", "nrun", "nrunrun"],
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


@pytest.mark.parametrize(
    "name, value",
    [
        ("nnu", 4.0),
        ("YHe", 0.24),
        ("TCMB", 2.73),
        ("standard_neutrino_neff", 4.0),
        ("tau", 0.06),
        ("pivot_scalar", 0.04),
    ],
)
def test_transfer_or_background_changes_cannot_use_primordial_rescaling(
    fiducial_cosmology, name, value
):
    """CAMB transfer inputs must not silently take the rescaling route."""

    assert name in fiducial_cosmology.get_background_params() or name == "pivot_scalar"
    assert not fiducial_cosmology.same_background({name: value})
    with pytest.raises(IncompatibleBackgroundError) as caught:
        RescaledCosmology(fiducial_cosmology, {name: value})
    assert set(caught.value.changes) == {name}


def test_nonstandard_pivot_is_rejected():
    """The current rescaling convention explicitly requires k_s=0.05/Mpc."""

    fiducial = Cosmology()
    fiducial.CAMBparams.InitPower.pivot_scalar = 0.04
    with pytest.raises(ValueError, match="pivot_scalar=0.05"):
        RescaledCosmology(fiducial, {"ns": 0.96})


def test_batch_cosmology_accessors_match_scalar(fiducial_cosmology):
    """Batch summaries preserve the scalar Cosmology/RescaledCosmology API."""

    cosmologies = [
        fiducial_cosmology,
        RescaledCosmology(fiducial_cosmology, {"ns": 0.96}),
    ]
    zs = np.array([2.5, 3.0])
    # The same public accessors dispatch to scalar-shaped output for one
    # cosmology and leading-batch output for a cosmology sequence.
    np.testing.assert_allclose(
        fiducial_cosmology.get_dkms_dMpc_for_cosmologies(fiducial_cosmology, zs),
        fiducial_cosmology.get_dkms_dMpc(zs),
    )
    scalar_summary = fiducial_cosmology.get_linP_Mpc_params_for_cosmologies(
        fiducial_cosmology, Z_STAR, 0.7
    )
    assert scalar_summary == fiducial_cosmology.get_linP_Mpc_params(Z_STAR, 0.7)

    dkms = fiducial_cosmology.get_dkms_dMpc_for_cosmologies(cosmologies, zs)
    mpc = fiducial_cosmology.get_linP_Mpc_params_for_cosmologies(
        cosmologies, zs, 0.7
    )
    kms = fiducial_cosmology.get_linP_kms_params_for_cosmologies(
        cosmologies, zs, KP_KMS
    )
    for index, cosmo in enumerate(cosmologies):
        np.testing.assert_allclose(dkms[index], cosmo.get_dkms_dMpc(zs))
        for redshift_index, redshift in enumerate(zs):
            scalar_mpc = cosmo.get_linP_Mpc_params(redshift, 0.7)
            scalar_kms = cosmo.get_linP_kms_params(redshift, KP_KMS)
            for name, value in scalar_mpc.items():
                np.testing.assert_allclose(mpc[name][index, redshift_index], value)
            for name, value in scalar_kms.items():
                np.testing.assert_allclose(kms[name][index, redshift_index], value)
