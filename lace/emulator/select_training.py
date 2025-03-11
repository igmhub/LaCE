from lace.emulator.constants import TrainingSet
from lace.archive import gadget_archive, nyx_archive
from scipy.interpolate import interp1d


def interp_k_Mpc_central(sim: list[dict], k_Mpc: list[float]) -> None:
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
    kp_Mpc: float = 0.7,
    z_star: float = 3,
    kp_kms: float = 0.009,
) -> tuple[gadget_archive.GadgetArchive | nyx_archive.NyxArchive, list[dict]]:
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
            z_max=z_max,
        )
    elif (training_set is None) and (archive is not None):
        print_func("Use custom archive provided by the user to train emulator")
        training_data = archive.get_training_data(
            emu_params=emu_params,
            drop_sim=drop_sim,
            drop_z=drop_z,
            z_max=z_max,
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
            )

    return archive, training_data
