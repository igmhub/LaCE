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


def test_lace_labels_take_precedence_over_cup1d():
    from lace.cosmo.labeled_cosmologies import get_cosmo_params_dict_from_label

    # LaCE labels are resolved in LaCE, never looked up in cup1d
    assert get_cosmo_params_dict_from_label("Planck18")["omch2"] == 0.11933


def test_unknown_cosmology_label_raises():
    import pytest

    with pytest.raises(ValueError, match="Unknown cosmology label"):
        cosmology.Cosmology(cosmo_label="not_a_cosmology")


def test_cosmology_label_without_cup1d_raises(monkeypatch):
    import sys

    import pytest

    # a None entry makes the import fail as if cup1d were not installed
    monkeypatch.setitem(sys.modules, "cup1d.theory.cosmology", None)
    with pytest.raises(ValueError, match="cup1d"):
        cosmology.Cosmology(cosmo_label="accel2")


def test_cosmology_label_falls_back_to_cup1d():
    import pytest

    pytest.importorskip("cup1d.theory.cosmology")
    cosmo = cosmology.Cosmology(cosmo_label="accel2")
    params = cosmo.input_cosmo_params_dict
    # ACCEL2 (Planck15-like) cosmology as defined in cup1d
    assert params["H0"] == pytest.approx(67.5)
    assert params["As"] == pytest.approx(np.exp(3.094) / 1e10)
    assert params["ns"] == pytest.approx(0.96)
    assert cosmo.input_cosmo_label == "accel2"
