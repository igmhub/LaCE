import argparse
import os, sys
import torch
import numpy as np
from lace.emulator.nn_emulator import NNEmulator
from lace.archive import nyx_archive, gadget_archive


def str_to_bool(s):
    if s == "True":
        return True
    elif s == "False":
        return False


def save_emu(model, label_training_set, emulator_label, drop_sim, drop_z):
    # set folder name
    folder = (
        os.environ["LACE_REPO"]
        + "/src/lace/data/NNmodels/"
        + label_training_set
        + "/"
    )

    # set file name
    fname = emulator_label

    if drop_sim != False:
        fname += "_drop_sim_" + drop_sim
        _drop_sim = drop_sim
    else:
        _drop_sim = None

    if drop_z != False:
        fname += "_drop_z_" + str(np.round(drop_z, 2))
        _drop_z = drop_z
    else:
        _drop_z = None

    fname += ".pt"

    # set metadata
    metadata = {
        "training_set": label_training_set,
        "emulator_label": emulator_label,
        "drop_sim": _drop_sim,
        "drop_z": _drop_z,
    }

    model.metadata = metadata

    model_data = {"metadata": metadata, "emulator": model}

    torch.save(model_data, folder + fname)


def main():
    # Create an ArgumentParser instance
    parser = argparse.ArgumentParser(
        description="Passing the emulator_label option"
    )

    parser.add_argument(
        "--training_set",
        default=None,
        choices=["Cabayol23", "Nyx23_Oct2023"],
        required=True,
    )
    parser.add_argument(
        "--emulator_label",
        default=None,
        choices=["Cabayol23", "Cabayol23_extended", "Nyx_v0", "Nyx_v1"],
        required=True,
    )
    parser.add_argument(
        "--drop_sim",
        default=None,
        choices=["True", "False"],
        help="Drop one simulation from the training set at a time",
        required=True,
    )
    parser.add_argument(
        "--drop_z",
        default=None,
        choices=["True", "False"],
        help="Drop one redshift from the training set at a time",
        required=True,
    )

    # Parse the command-line arguments
    args = parser.parse_args()
    args.drop_sim = str_to_bool(args.drop_sim)
    args.drop_z = str_to_bool(args.drop_z)

    if args.training_set == "Cabayol23":
        archive = gadget_archive.GadgetArchive(postproc="Cabayol23")
    elif args.training_set[:5] == "Nyx23":
        archive = nyx_archive.NyxArchive(nyx_version=args.training_set[6:])
    else:
        print("Training_set not implemented")
        sys.exit()

    # l1O sims
    if args.drop_sim == True:
        for isim, sim_label in enumerate(archive.list_sim_cube):
            print("Simulation l1O " + str(isim))
            nn_emu = NNEmulator(
                archive=archive,
                emulator_label=args.emulator_label,
                drop_sim=sim_label,
                drop_z=None,
                initial_weights=False,
            )
            save_emu(
                nn_emu.nn.state_dict(),
                args.training_set,
                args.emulator_label,
                sim_label,
                False,
            )

    # l1O redshifts
    if args.drop_z == True:
        for iz, redshift in enumerate(archive.list_sim_redshifts):
            print("Redshift l1O " + str(iz))
            nn_emu = NNEmulator(
                archive=archive,
                emulator_label=args.emulator_label,
                drop_sim=None,
                drop_z=redshift,
                initial_weights=False,
            )
            save_emu(
                nn_emu.nn.state_dict(),
                args.training_set,
                args.emulator_label,
                False,
                redshift,
            )

    # no l1O
    if (args.drop_sim == False) & (args.drop_z == False):
        print("No l1O")
        nn_emu = NNEmulator(
            archive=archive,
            emulator_label=args.emulator_label,
            drop_sim=None,
            drop_z=None,
            initial_weights=False,
        )
        save_emu(
            nn_emu.nn.state_dict(),
            args.training_set,
            args.emulator_label,
            args.drop_sim,
            args.drop_z,
        )


if __name__ == "__main__":
    main()
