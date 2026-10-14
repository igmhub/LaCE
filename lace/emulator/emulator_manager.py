"""Factory for the supported production P1D emulators."""

from lace.emulator.gp_emulator_multi import GPEmulator


SUPPORTED_EMULATORS = ("CH24_mpgcen_gpr", "CH24_nyxcen_gpr")


def set_emulator(emulator_label, *, model_path=None, data_path=None, normalization_path=None):
    """Load a supported production GP P1D emulator.

    Parameters
    ----------
    emulator_label : {"CH24_mpgcen_gpr", "CH24_nyxcen_gpr"}
        Production emulator label understood internally by LaCE.
    model_path, data_path, normalization_path : path-like, optional
        Explicit bundle, data, and normalization locations. Supplied paths
        override persistent configuration and checkout defaults.

    Returns
    -------
    GPEmulator
        Initialized production P1D emulator.

    Raises
    ------
    ValueError
        If ``emulator_label`` is not a supported internal LaCE label.
    """
    if emulator_label not in SUPPORTED_EMULATORS:
        supported = ", ".join(SUPPORTED_EMULATORS)
        raise ValueError(
            f"Emulator {emulator_label!r} is not supported. "
            f"Supported emulators are: {supported}."
        )
    return GPEmulator(
        emulator_label=emulator_label,
        model_path=model_path,
        data_path=data_path,
        normalization_path=normalization_path,
    )
