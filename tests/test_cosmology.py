"""Regression tests for LaCE's public cosmology interfaces."""

import numpy as np
import pytest

from lace.cosmo import cosmology


REDSHIFT = 2.33


def test_default_cosmology_growth_and_linear_power():
    """The default cosmology reproduces its reference values.

    References updated when CAMB started running without the CMB (as Cobaya's
    CAMB in lya_interface): its k sampling moves P(k) by up to ~1e-5 here.
    """

    cosmo = cosmology.Cosmology()

    np.testing.assert_allclose(
        cosmo.get_sigma8(REDSHIFT), 0.31133249465449764, rtol=1.0e-6
    )
    np.testing.assert_allclose(
        cosmo.get_growth_rate(REDSHIFT), 0.9679356280449923, rtol=1.0e-6
    )

    k_Mpc = np.array([0.1, 0.7, 1.0])
    expected_linear_power = np.array([1529.1302892587607, 29.59988149397743, 12.833450310960972])
    np.testing.assert_allclose(
        cosmo.get_linP_Mpc(z=REDSHIFT, k_Mpc=k_Mpc),
        expected_linear_power,
        rtol=1.0e-6,
    )


def test_planck18_labels():
    from lace.cosmo.labeled_cosmologies import get_cosmo_params_dict_from_label

    assert get_cosmo_params_dict_from_label("Planck18_wonu")["omch2"] == 0.11933
    with pytest.warns(FutureWarning):
        assert get_cosmo_params_dict_from_label("Planck18") == get_cosmo_params_dict_from_label("Planck18_wonu")


def test_unknown_cosmology_label_raises():
    import pytest

    with pytest.raises(ValueError, match="Unknown cosmology label"):
        cosmology.Cosmology(cosmo_label="not_a_cosmology")


def test_simulation_cosmology_labels_are_defined_in_lace(monkeypatch):
    import sys

    import pytest

    # a None entry makes any import of cup1d fail: LaCE must not need it
    monkeypatch.setitem(sys.modules, "cup1d", None)
    cosmo = cosmology.Cosmology(cosmo_label="accel2")
    params = cosmo.input_cosmo_params_dict
    # ACCEL2 (Planck15-like) cosmology
    assert params["H0"] == pytest.approx(67.5)
    assert params["As"] == pytest.approx(np.exp(3.094) / 1e10)
    assert params["ns"] == pytest.approx(0.96)
    assert cosmo.input_cosmo_label == "accel2"
