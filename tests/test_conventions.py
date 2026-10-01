import numpy as np
import pytest

from lace.conventions import canonicalize_unit_keys, validate_wavenumber
from lace.emulator.gp_emulator_multi import GPEmulator


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


def test_canonical_gp_method_uses_the_active_gp_contract(monkeypatch):
    emulator = object.__new__(GPEmulator)
    calls = {}

    def emulate_p1d(model, k_iMpc, verbose=False, return_coeff=False):
        calls.update(
            model=model,
            k_iMpc=k_iMpc,
            verbose=verbose,
            return_coeff=return_coeff,
        )
        return np.asarray(k_iMpc) * model["amplitude"]

    monkeypatch.setattr(emulator, "emulate_p1d_Mpc", emulate_p1d)
    np.testing.assert_equal(
        emulator.emulate_P1D_Mpc(
            {"amplitude": 2}, [1, 2], verbose=True, return_coeff=True
        ),
        [2, 4],
    )
    assert calls["verbose"] is True
    assert calls["return_coeff"] is True


@pytest.mark.parametrize("kwargs", [{"return_covar": True}, {"z": 3.0}])
def test_canonical_gp_method_rejects_retired_options(kwargs):
    emulator = object.__new__(GPEmulator)
    with pytest.raises(NotImplementedError):
        emulator.emulate_P1D_Mpc({}, [1.0], **kwargs)
