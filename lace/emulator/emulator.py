import sys
from lace.emulator import gp_emulator
from lace.emulator import nn_emulator

# from lace.emulator import pnd_archive
from lace.archive import pnd_archive
from lace.archive import interface_archive

from lace.emulator.nn_architecture import MDNemulator_polyfit
from lace.emulator.nn_emulator import NNEmulator
from lace.emulator.gp_emulator import GPEmulator


def P1D_emulator(
    archive=None,
    emu_algorithm=None,
    archive_label="Gadget",
    emulator_label=None,
    zmax=4.5,
    kmax_Mpc=4,
    ndeg=5,
    train=True,
    emu_type="polyfit",
    model_path=None,
    save_path=None,
    nepochs_nn=1,
):
    """archive: Data archive used for training the emulator. It is an optional argument that must be provided when using a custom emulator.
    emu_algorithm: This argument specifies the type of emulator algorithm to be used. It is an optional argument that must be provided when using a custom emulator. Options are 'GP' for Gaussian Process and 'NN' for Neural Network.
    archive_label: Type of archive being used. The default value is set to 'Gadget'. Adding Nyx is work in progress
    emulator_label: Specific emulator label calling the emulators in Pdersetn et al. 2021: 'Pedersen21', and Cabayol-Garcia et al. 2023:'Cabayol23'.
    zmax: This argument specifies the maximum redshift value. It is used when training a custom emulator. The default value is 4.5.
    kmax_Mpc: This argument represents the maximum wavenumber in units of Mpc^(-1). It is used when training a custom emulator. The default value is 4.
    ndeg: Degree of the polynomial fit. It is used when training a custom emulator. The default value is 5.
    train: Boolean flag that determines whether to train the emulator or load a pre-trained one. When set to True, the emulator is trained. When set to False, a pre-trained emulator is loaded. The default value is True. Only valid for the NN emulator.
    emu_type: Type of emulator to be used. It is used when training a custom emulator. The default value is 'polyfit'.
    model_path: String that specifies the path to a pre-trained emulator model.
    save_path: String that indicates the path to save the emulator model once trained. Only valid for the NN emulator.
    nepochs_nn: Integer fixing the number of epochs the NN is trained."""

    if emulator_label == "Pedersen21":
        print("Select emulator used in Pedersen et al. 2021")
        emuparams = ["Delta2_p", "n_p", "mF", "sigT_Mpc", "gamma", "kF_Mpc"]
        zmax, kmax_Mpc, emu_type = 4.5, 3, "k_bin"

        print(
            r"Gaussian Process emulator predicting the P1D at each k-bin. It goes to scales of 3Mpc^{-1} and z<4.5. The parameters passed to the emulator will be overwritten to match these ones."
        )

        archive = pnd_archive.archivePND(sim_suite="Pedersen21")
        archive.get_training_data()

        emulator = GPEmulator(archive=archive, emu_type=emu_type)
        return emulator

    if emulator_label == "Pedersen23":
        print("Select emulator used in Pedersen et al. 2023")
        emuparams = ["Delta2_p", "n_p", "mF", "sigT_Mpc", "gamma", "kF_Mpc"]
        zmax, kmax_Mpc, ndeg, emu_type = 4.5, 3, 4, "polyfit"

        print(
            r"Gaussian Process emulator predicting the optimal P1D fitting coefficients to a 5th degree polynomial. It goes to scales of 4Mpc^{-1} and z<4.5. The parameters passed to the emulator will be overwritten to match these ones"
        )

        archive = pnd_archive.archivePND(sim_suite="Pedersen21")
        archive.get_training_data()

        emulator = GPEmulator(archive=archive, emu_type=emu_type)
        return emulator

    elif emulator_label == "Cabayol23":
        print("Select emulator used in Cabayol-Garcia et al. 2023")
        emuparams = ["Delta2_p", "n_p", "mF", "sigT_Mpc", "gamma", "kF_Mpc"]
        zmax, kmax_Mpc, ndeg = 4.5, 4, 5

        print(
            r"Neural network emulator predicting the optimal P1D fitting coefficients to a 5th degree polynomial. It goes to scales of 4Mpc^{-1} and z<4.5. The parameters passed to the emulator will be overwritten to match these ones"
        )

        if train == True:
            print("Train emulator")
            archive = pnd_archive.archivePND(sim_suite="Cabayol23")
            archive.get_training_data()

            emulator = NNEmulator(
                archive,
                emuparams,
                nepochs=nepochs_nn,
                step_size=75,
                kmax_Mpc=4,
                ndeg=5,
                Nsim=30,
                train=True,
            )
        if train == False:
            if model_path == None:
                raise ValueError("if train==False, a model path must be provided")
            else:
                print("Load pre-trained emulator")
                archive = pnd_archive.archivePND(sim_suite="Cabayol23")
                archive.get_training_data()
                emulator = NNEmulator(
                    archive, emuparams, train=False, model_path=model_path
                )

    if emulator_label == None:
        print("Select custom emulator")
        if (archive == None) | (emu_algorithm == None):
            raise ValueError(
                "If an emulator label is not provided, the archive and emulator type are required"
            )

        if (emu_algorithm == "NN") & (archive_label == "Gadget"):
            emuparams = ["Delta2_p", "n_p", "mF", "sigT_Mpc", "gamma", "kF_Mpc"]
            emulator = NNEmulator(
                archive,
                emuparams,
                zmax=zmax,
                kmax_Mpc=kmax_Mpc,
                ndeg=ndeg,
                save_path=save_path,
            )

        if (emu_algorithm == "NN") & (archive_label == "Nyx"):
            emuparams = ["Delta2_p", "n_p", "mF", "sigT_Mpc", "gamma", "lambda_P"]
            raise Exception("Work in progress")

        if (emu_algorithm == "GP") & (archive_label == "Gadget"):
            emuparams = ["Delta2_p", "n_p", "mF", "sigT_Mpc", "gamma", "kF_Mpc"]
            emulator = GPEmulator(archive=archive)

    return emulator
