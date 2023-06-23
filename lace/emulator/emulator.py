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
    """Emulates the P1D power spectrum.

    Args:
        archive (class): Data archive used for training the emulator.
            Required when using a custom emulator.
        emu_algorithm (str): Type of emulator algorithm to be used.
            Required when using a custom emulator.
            Options are 'GP' for Gaussian Process and 'NN' for Neural Network.
        archive_label (str): Type of archive being used. Default is 'Gadget'.
            Adding Nyx is work in progress.
        emulator_label (str): Specific emulator label. Options are
            'Pedersen21', 'Pedersen23', and 'Cabayol23'.
        zmax (float): Maximum redshift value. Used when training a custom
            emulator. Default is 4.5.
        kmax_Mpc (int): Maximum wavenumber in units of Mpc^(-1). Used when
            training a custom emulator. Default is 4.
        ndeg (int): Degree of the polynomial fit. Used when training a custom
            emulator. Default is 5.
        train (bool): Flag that determines whether to train the emulator or
            load a pre-trained one. Default is True. Only valid for the NN emulator.
        emu_type (str): Type of emulator to be used. Used when training a
            custom emulator. Default is 'polyfit'.
        model_path (str): Path to a pre-trained emulator model.
        save_path (str): Path to save the emulator model once trained.
            Only valid for the NN emulator.
        nepochs_nn (int): Number of epochs the NN is trained. Default is 1.

    Returns:
        Emulator: The trained or loaded emulator object.

    Raises:
        ValueError: If train is False but no model_path is provided.
        ValueError: If emulator_label is None and archive or emu_algorithm is not provided.
        Exception: If emu_algorithm is 'NN' and archive_label is 'Nyx'. Work in progress.

    Notes:
        - When emulator_label is 'Pedersen21' or 'Pedersen23', the function prints
          additional information about the selected emulator.
        - When emulator_label is 'Cabayol23' and train is True, the function also prints
          information about training the emulator.
    """

    emulator_label_all = ["Pedersen21", "Pedersen23", "Cabayol23"]
    if emulator_label is not None:
        print("Selected pre-tuned emulator")
        try:
            if emulator_label in emulator_label_all:
                pass
            else:
                print(
                    "Invalid emulator_label value. Available options: ",
                    emulator_label_all,
                )
                raise
        except:
            print("An error occurred while checking the emulator_label value.")
            raise
    else:
        print("Selected custome emulator")
        if (archive == None) | (emu_algorithm == None) | (archive_label == None):
            raise ValueError(
                "archive, emu_algorithm, archive_label must be provided"
                + "if an emulator label is not"
            )

    if emulator_label == "Pedersen21":
        print("Select emulator used in Pedersen et al. 2021")
        emuparams = ["Delta2_p", "n_p", "mF", "sigT_Mpc", "gamma", "kF_Mpc"]
        zmax, kmax_Mpc, emu_type = 4.5, 3, "k_bin"

        print(
            r"Gaussian Process emulator predicting the P1D at each k-bin."
            + " It goes to scales of 3Mpc^{-1} and z<4.5. The parameters "
            + "passed to the emulator will be overwritten to match these ones."
        )

        archive = pnd_archive.archivePND(sim_suite="Pedersen21")
        archive.get_training_data()

        emulator = GPEmulator(archive=archive, emu_type=emu_type)
        return emulator
    elif emulator_label == "Pedersen23":
        print("Select emulator used in Pedersen et al. 2023")
        emuparams = ["Delta2_p", "n_p", "mF", "sigT_Mpc", "gamma", "kF_Mpc"]
        zmax, kmax_Mpc, ndeg, emu_type = 4.5, 3, 4, "polyfit"

        print(
            r"Gaussian Process emulator predicting the optimal P1D"
            + "fitting coefficients to a 5th degree polynomial. It "
            + "goes to scales of 4Mpc^{-1} and z<4.5. The parameters"
            + " passed to the emulator will be overwritten to match "
            + "these ones"
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
            r"Neural network emulator predicting the optimal P1D "
            + "fitting coefficients to a 5th degree polynomial. It "
            + "goes to scales of 4Mpc^{-1} and z<4.5. The parameters "
            + "passed to the emulator will be overwritten to match "
            + "these ones"
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
    elif emulator_label is None:
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
        elif (emu_algorithm == "NN") & (archive_label == "Nyx"):
            emuparams = ["Delta2_p", "n_p", "mF", "sigT_Mpc", "gamma", "lambda_P"]
            raise Exception("Work in progress")

        elif (emu_algorithm == "GP") & (archive_label == "Gadget"):
            emuparams = ["Delta2_p", "n_p", "mF", "sigT_Mpc", "gamma", "kF_Mpc"]
            emulator = GPEmulator(archive=archive)
        else:
            raise ValueError(
                "Combination of emu_algorithm and archive_label not supported"
            )

    return emulator
