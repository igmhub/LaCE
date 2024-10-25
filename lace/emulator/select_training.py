from lace.emulator.constants import TrainingSet
from lace.archive import gadget_archive, nyx_archive

def select_training(archive, training_set, emu_params, drop_sim, drop_z, z_max, nyx_file=None, train=True, print_func=print):
    if (archive is None) and (training_set is None):
        raise ValueError("Archive or training_set must be provided")

    if (training_set is not None) and (archive is None):
        if training_set not in TrainingSet:
            raise ValueError(f"Invalid training_set value {training_set}. Available options: {', '.join(TrainingSet)}")

        print_func(f"Selected training set {training_set}")

        if training_set in [TrainingSet.PEDERSEN21, TrainingSet.CABAYOL23]:
            archive = gadget_archive.GadgetArchive(postproc=training_set)
        elif training_set.startswith("Nyx23"):
            archive = nyx_archive.NyxArchive(nyx_version=training_set[6:], nyx_file=nyx_file)

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
