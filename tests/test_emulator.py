"""Regression tests for the packaged P1D emulator."""

import numpy as np
import pytest

from lace.emulator import set_emulator


def test_unsupported_emulator_label_is_rejected():
    with pytest.raises(ValueError, match="Supported emulators"):
        set_emulator("CH24_mpg_gpr")


def test_ch24_mpgcen_gpr_emulation():
    """The Tutorial_emulator central-parameter example remains reproducible."""

    emulator = set_emulator("CH24_mpgcen_gpr")
    k_Mpc = np.geomspace(0.1, 4.0, 100)
    input_params = {
        "Delta2_p": [0.30, 0.35, 0.40],
        "n_p": [-2.3, -2.3, -2.3],
        "alpha_p": [-0.215, -0.215, -0.215],
        "mF": [0.66, 0.66, 0.66],
        "gamma": [1.5, 1.5, 1.5],
        "sigT_Mpc": [0.128, 0.128, 0.128],
        "kF_Mpc": [10.5, 10.5, 10.5],
    }

    p1d_Mpc = emulator.emulate_p1d_Mpc(input_params, k_Mpc)

    assert p1d_Mpc.shape == (3, len(k_Mpc))
    assert np.all(np.isfinite(p1d_Mpc))
    assert np.all(p1d_Mpc > 0)
    np.testing.assert_allclose(
        p1d_Mpc[:, [0, 49, 99]],
        [
            [0.54043298, 0.31625583, 0.04142718],
            [0.56596408, 0.32841439, 0.04143160],
            [0.59275015, 0.34078060, 0.04083727],
        ],
        rtol=1.0e-6,
    )
    assert np.all(np.diff(p1d_Mpc[:, 0]) > 0)


def test_ch24_nyxcen_gpr_prediction():
    emulator = set_emulator("CH24_nyxcen_gpr")
    model = {
        "Delta2_p": [0.35], "n_p": [-2.3], "alpha_p": [-0.215],
        "mF": [0.66], "gamma": [1.5], "sigT_Mpc": [0.128], "kF_Mpc": [10.5],
    }
    result = emulator.emulate_p1d_Mpc(model, np.array([0.1, 1.0, 4.0]))
    np.testing.assert_allclose(result, [[0.54896665, 0.22852632, 0.03803230]], rtol=1e-6)


def test_malformed_parameter_shapes_are_rejected():
    emulator = set_emulator("CH24_mpgcen_gpr")
    model = {"Delta2_p": [0.3, 0.4], "n_p": [-2.3], "mF": [0.66, 0.66],
             "gamma": [1.5, 1.5], "sigT_Mpc": [0.128, 0.128], "kF_Mpc": [10.5, 10.5]}
    with pytest.raises(ValueError, match="same length"):
        emulator.predict(model)
