import numpy as np
from lace.emulator.emulator_manager import set_emulator
from lace.archive import gadget_archive, nyx_archive
from lace.emulator.gp_emulator_new import GPEmulator


def main():
    """Train and store latest GP models. One need to be done once"""

    # Train CH24_mpg_gp emulator
    # archive = gadget_archive.GadgetArchive(postproc="Cabayol23")
    # print("Training CH24_mpg_gp emulator")
    # emulator = GPEmulator(
    #     archive=archive, emulator_label="CH24_mpg_gp", train=True
    # )

    # Train CH24_nyx_gp emulator
    archive = nyx_archive.NyxArchive(nyx_version="Jul2024")
    print("Training CH24_nyx_gp emulator")
    emulator = GPEmulator(
        archive=archive, emulator_label="CH24_nyx_gp", train=True
    )


if __name__ == "__main__":
    main()
