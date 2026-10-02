from lace.emulator.constants import TrainingSet
from lace.archive import gadget_archive, nyx_archive
from scipy.interpolate import interp1d


def interp_k_Mpc_central(sim: list[dict], k_Mpc: list[float]) -> None:
    """Interpolate central simulation P1D arrays onto a common k grid.

    Parameters
    ----------
    sim : list of dict
        Simulation dictionaries modified in place.
    k_Mpc : array_like
        Target comoving wavenumber grid in 1/Mpc.
    """
    for sim_dict in sim:
        # Create interpolation function for this simulation's P1D
        p1d_interp = interp1d(
            sim_dict["k_Mpc"], sim_dict["p1d_Mpc"], fill_value="extrapolate"
        )
        # Interpolate P1D onto new k grid
        sim_dict["p1d_Mpc"] = p1d_interp(k_Mpc)
        # Update k grid
        sim_dict["k_Mpc"] = k_Mpc


def select_training(
    archive: gadget_archive.GadgetArchive | nyx_archive.NyxArchive | None,
    training_set: str | TrainingSet | None,
    emu_params: list[str],
    drop_sim: list[str] | None,
    drop_z: list[float] | None,
    include_central: bool,
    z_max: float,
    nyx_file: str | None = None,
    train: bool = True,
    print_func: callable = print,
    average: str = "both",
    kp_Mpc: float = 0.7,
    z_star: float = 3,
    kp_kms: float = 0.009,
) -> tuple[gadget_archive.GadgetArchive | nyx_archive.NyxArchive, list[dict]]:
    """Select and preprocess simulations for emulator training.

    Parameters
    ----------
    archive : GadgetArchive or NyxArchive, optional
        Existing simulation archive.
    training_set : str or TrainingSet, optional
        Named training collection when ``archive`` is not supplied.
    emu_params : sequence of str
        Required emulator-input fields retained from archive entries.
    drop_sim, drop_z : sequence, optional
        Simulation or redshift selections excluded from training.
    include_central : bool
        Include the Nyx central simulation in the training cube.
    z_max : float
        Largest retained redshift.
    nyx_file : path-like, optional
        Explicit Nyx HDF5 file for named Nyx training sets.
    train : bool, default=True
        Reject simultaneous archive and training-set inputs during training.
    print_func : callable, default=print
        Progress-reporting callable.
    average : str, default="both"
        Archive averaging convention.
    kp_Mpc, z_star, kp_kms : float, default=0.7, 3, 0.009
        Linear-power pivots in 1/Mpc and s/km.

    Returns
    -------
    archive : GadgetArchive or NyxArchive
        Selected archive.
    training_data : list of dict
        Emulator-ready simulation measurements.

    Raises
    ------
    ValueError
        If archive/training-set selection is ambiguous or invalid.
    """
    if (archive is None) and (training_set is None):
        raise ValueError("Archive or training_set must be provided")

    if (training_set is not None) and (archive is None):
        if isinstance(training_set, str):
            try:
                training_set = TrainingSet(training_set)
            except ValueError:
                raise ValueError(
                    f"Invalid training_set value '{training_set}'. Available options: {', '.join(t.value for t in TrainingSet)}"
                )
        elif not isinstance(training_set, TrainingSet):
            raise ValueError(
                f"Invalid training_set type. Expected str or TrainingSet, got {type(training_set)}"
            )

        print_func(f"Selected training set {training_set}")

        if training_set in [TrainingSet.PEDERSEN21, TrainingSet.CABAYOL23]:
            archive = gadget_archive.GadgetArchive(
                postproc=training_set,
                kp_Mpc=kp_Mpc,
                z_star=z_star,
                kp_kms=kp_kms,
            )
        elif training_set.startswith("Nyx23"):
            archive = nyx_archive.NyxArchive(
                nyx_version=training_set[6:],
                nyx_file=nyx_file,
                include_central=include_central,
                kp_Mpc=kp_Mpc,
                z_star=z_star,
                kp_kms=kp_kms,
            )
            central_idx = [
                i
                for i, sim in enumerate(archive.data)
                if sim["sim_label"] == "nyx_central"
            ]
            if central_idx:
                interp_k_Mpc_central(
                    [archive.data[i] for i in central_idx],
                    archive.data[10000]["k_Mpc"],
                )

        training_data = archive.get_training_data(
            emu_params=emu_params,
            drop_sim=drop_sim,
            drop_z=drop_z,
            z_max=z_max,
            average=average,
        )
    elif (training_set is None) and (archive is not None):
        print_func("Use custom archive provided by the user to train emulator")
        training_data = archive.get_training_data(
            emu_params=emu_params,
            drop_sim=drop_sim,
            drop_z=drop_z,
            z_max=z_max,
            average=average,
        )

    elif (training_set is not None) and (archive is not None):
        if train:
            raise ValueError(
                "Provide either archive or training set for training"
            )
        else:
            print_func(
                "Using custom archive provided by the user to load emulator"
            )
            training_data = archive.get_training_data(
                emu_params=emu_params,
                drop_sim=drop_sim,
                drop_z=drop_z,
                z_max=z_max,
                average=average,
            )

    return archive, training_data
