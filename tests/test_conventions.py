import numpy as np
import pytest

from lace.conventions import canonicalize_unit_keys, validate_wavenumber
from lace.emulator.base_emulator import BaseEmulator


class _Emulator(BaseEmulator):
    def emulate_p1d_Mpc(self, model, k_Mpc, return_covar=False, z=None):
        return np.asarray(k_Mpc) * model["amplitude"]


def test_legacy_archive_keys_are_canonicalized():
    old = {"k_Mpc": np.array([0.1]), "p1d_Mpc": np.array([2.0])}
    new = canonicalize_unit_keys(old)
    assert new["k_iMpc"] is old["k_Mpc"]
    assert new["P1D_Mpc"] is old["p1d_Mpc"]


def test_canonical_key_wins():
    assert canonicalize_unit_keys({"k_Mpc": 1, "k_iMpc": 2})["k_iMpc"] == 2


def test_wavenumber_contract():
    np.testing.assert_equal(validate_wavenumber([0.1, 1], name="k_iMpc"), [0.1, 1])
    with pytest.raises(ValueError, match="k_iMpc"):
        validate_wavenumber([0.0, 1], name="k_iMpc")


def test_canonical_emulator_method_delegates_to_legacy_implementation():
    emulator = _Emulator()
    np.testing.assert_equal(
        emulator.emulate_P1D_Mpc({"amplitude": 2}, [1, 2]),
        [2, 4],
    )
