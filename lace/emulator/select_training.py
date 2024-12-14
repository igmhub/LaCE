from lace.emulator.constants import TrainingSet
from lace.archive import gadget_archive, nyx_archive
from lace.archive.constants import (
    LIST_SIM_MPG_CUBE_COMBINED,
    LIST_ALL_SIMS_COMBINED
)
from scipy.interpolate import interp1d


def interp_k_Mpc_central(sim: list[dict], 
                         k_Mpc: list[float]) -> None:
    """Interpolate P1D onto new k grid for central simulations."""
    for sim_dict in sim:
        # Create interpolation function for this simulation's P1D
        p1d_interp = interp1d(
            sim_dict["k_Mpc"],
            sim_dict["p1d_Mpc"],
            fill_value="extrapolate"
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
) -> tuple[gadget_archive.GadgetArchive | nyx_archive.NyxArchive, list[dict]]:
    """Select training data from archive or training set."""
    if (archive is None) and (training_set is None):
        raise ValueError("Archive or training_set must be provided")

    if (training_set is not None) and (archive is None):
        try:
            training_set = TrainingSet(training_set)
        except ValueError:
            raise ValueError(
                f"Invalid training_set value '{training_set}'. "
                f"Available options: {', '.join(t.value for t in TrainingSet)}"
            )

        print_func(f"Selected training set {training_set}")

        if training_set in [TrainingSet.PEDERSEN21, TrainingSet.CABAYOL23]:
            archive = gadget_archive.GadgetArchive(postproc=training_set)
        elif training_set.startswith("Nyx23"):
            archive = nyx_archive.NyxArchive(
                nyx_version=training_set[6:],
                nyx_file=nyx_file,
                include_central=include_central,
                kp_Mpc=0.7
            )
            central_idx = [i for i, sim in enumerate(archive.data) if sim["sim_label"] == "nyx_central"]
            if central_idx:
                interp_k_Mpc_central(
                    [archive.data[i] for i in central_idx],
                    archive.data[10000]["k_Mpc"]
                )
        elif training_set == TrainingSet.COMBINED:
            gadget_archive_obj = gadget_archive.GadgetArchive(postproc="Cabayol23")
            nyx_archive_obj = nyx_archive.NyxArchive(nyx_version="Jul2024")

            gadget_training_data = gadget_archive_obj.get_training_data(emu_params=emu_params)
            nyx_training_data = nyx_archive_obj.get_training_data(emu_params=emu_params)

            # Adjust interpolating p1d of nyx to k_Mpc of gadget
            k_Mpc_gadget = gadget_training_data[0]["k_Mpc"]
            k_Mpc_gadget = k_Mpc_gadget[(k_Mpc_gadget>0)&(k_Mpc_gadget<4)]
            
            for sim in nyx_training_data:
                sim["p1d_Mpc"] = interp1d(
                    sim["k_Mpc"],
                    sim["p1d_Mpc"],
                    fill_value="interpolate"
                )(k_Mpc_gadget)
                sim["k_Mpc"] = k_Mpc_gadget

            training_data = gadget_training_data + nyx_training_data

            gadget_archive_obj.data.extend(nyx_training_data)
            gadget_archive_obj.list_sim_cube = LIST_SIM_MPG_CUBE_COMBINED
            gadget_archive_obj.list_sim = LIST_ALL_SIMS_COMBINED

            return gadget_archive_obj, training_data


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
            raise ValueError("Provide either archive or training set for training")
        else:
            print_func("Using custom archive provided by the user to load emulator")
            training_data = archive.get_training_data(
                emu_params=emu_params,
                drop_sim=drop_sim,
                drop_z=drop_z,
                z_max=z_max,
            )

    return archive, training_data
