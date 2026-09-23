"""Reference emulator parameters for central simulations.

The values below are the emulator inputs of the CH24 central simulations at
redshift ``z=3``. They are kept in code so lightweight notebooks can reproduce a
central prediction without opening a simulation archive.
"""

from __future__ import annotations


CENTRAL_PARAMETERS_Z3 = {
    "CH24_mpgcen_gpr": {
        "Delta2_p": 0.3501252719027313,
        "n_p": -2.300047197725595,
        "mF": 0.6604100706377194,
        "sigT_Mpc": 0.12817463664956008,
        "gamma": 1.512170923999183,
        "kF_Mpc": 10.6348381789184,
    },
    "CH24_nyxcen_gpr": {
        "Delta2_p": 0.3633269286538266,
        "n_p": -2.3013313851865567,
        "alpha_p": -0.21539434795118495,
        "mF": 0.6476532233333333,
        "sigT_Mpc": 0.13102461634633503,
        "gamma": 1.5200821,
        "kF_Mpc": 12.936003969403709,
    },
}


def get_central_parameters_z3(emulator_label: str) -> dict[str, float]:
    """Return a copy of the central-simulation emulator inputs at ``z=3``."""
    try:
        return CENTRAL_PARAMETERS_Z3[emulator_label].copy()
    except KeyError as error:
        supported_labels = ", ".join(CENTRAL_PARAMETERS_Z3)
        raise ValueError(
            f"No z=3 central parameters are stored for {emulator_label!r}. "
            f"Supported labels: {supported_labels}."
        ) from error
