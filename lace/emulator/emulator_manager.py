"""Factory for the supported production P1D emulators."""

from lace.emulator.gp_emulator_multi import GPEmulator


SUPPORTED_EMULATORS = ("CH24_mpgcen_gpr", "CH24_nyxcen_gpr")


def set_emulator(emulator_label):
    """Load a supported production emulator by label."""
    if emulator_label not in SUPPORTED_EMULATORS:
        supported = ", ".join(SUPPORTED_EMULATORS)
        raise ValueError(
            f"Emulator {emulator_label!r} is not supported. "
            f"Supported emulators are: {supported}."
        )
    return GPEmulator(emulator_label=emulator_label)
