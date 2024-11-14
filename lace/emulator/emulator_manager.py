"""Emulator manager"""

from lace.emulator.nn_emulator import NNEmulator
from lace.emulator.gp_emulator import GPEmulator
from lace.emulator.constants import (
    EmulatorLabel,
    TrainingSet,
    GADGET_LABELS,
    NYX_LABELS,
)


def emulators_supported():
    """List of emulators supported.
    LaCE emulators:
        - **Pedersen21**: GPEmulator used in Pedersen21, k-bin emulator.
          Superseded by Pedersen21_ext.
        - **Pedersen21_ext**: GPEmulator like Pedersen21 but using Cabayol23 postprocessing.
        - **Pedersen21_ext8**: GPEmulator like Pedersen21_ext but accessing smaller scales.
        - **Pedersen23**: GPEmulator used in Pedersen23, polynomial emulator.
          Superseded by Pedersen23_ext.
        - **Pedersen23_ext**: GPEmulator like Pedersen23 but using Cabayol23 postprocessing.
          Recommended GP emulator.
        - **Pedersen23_ext8**: GPEmulator like Pedersen23_ext but accessing smaller scales.
          Recommended GP emulator for accessing small scales.
        - **CH24**: GPEmulator based on non-linear smoothing.
        - **Cabayol23**: NNEmulator used in Cabayol23, polynomial emulator.
          Superseded by Cabayol23+.
        - **Cabayol23+**: NNEmulator like Cabayol23 but using better architecture.
          Recommended NN emulator.
        - **Cabayol23_extended**: NNEmulator used in Cabayol23 accessing smaller scales than Cabayol23.
          Superseded by Cabayol23_extended+.
        - **Cabayol23+_extended**: NNEmulator like Cabayol23_extended but using better architecture.
          Recommended NN emulator for accessing small scales.

    Nyx emulators:
        - **Nyx_v0**: NNEmulator using amplitude and slope, polynomial emulator.
          Superseded by Nyx_alphap.
        - **Nyx_alphap**: NNEmulator using amplitude, slope, and running, polynomial emulator.
          Recommended emulator.
    """
    return [label.value for label in EmulatorLabel]


def set_emulator(emulator_label, archive=None, drop_sim=None):
    """Loads emulator.

    Parameters
    ----------
    emulator_label : str
        Name of the emulator.
    archive : object, optional
        Archive with the emulator training data. If None, an archive
        corresponding to the emulator_label will be loaded.
    drop_sim : str, optional
        Drop a specific simulation from the training set.

    Returns
    -------
    object
        Loaded emulator.
    """

    if emulator_label not in emulators_supported():
        msg = f"Emulator {emulator_label} not supported. Supported emulators are {emulators_supported()}"
        raise ValueError(msg)

    emulator_label = EmulatorLabel(emulator_label)

    if emulator_label in {EmulatorLabel.PEDERSEN21, EmulatorLabel.PEDERSEN23}:
        training_set = TrainingSet.PEDERSEN21
    elif emulator_label in {
        EmulatorLabel.PEDERSEN21_EXT,
        EmulatorLabel.PEDERSEN23_EXT,
        EmulatorLabel.CH24,
        EmulatorLabel.PEDERSEN23_EXT8,
    }:
        training_set = TrainingSet.CABAYOL23
    elif emulator_label in {EmulatorLabel.NYX_ALPHAP_COV}:
        training_set = TrainingSet.NYX23_JUL2024
    else:
        training_set = (
            TrainingSet.CABAYOL23
            if emulator_label in GADGET_LABELS
            else TrainingSet.NYX23_OCT2023
        )

    if archive is not None:
        expected_prefix = "mpg" if emulator_label in GADGET_LABELS else "nyx"
        if archive.data[0]["sim_label"][:3] != expected_prefix:
            raise ValueError(
                f"WARNING: training data in archive are not {expected_prefix} sims"
            )

    if emulator_label in {
        EmulatorLabel.PEDERSEN21,
        EmulatorLabel.PEDERSEN23,
        EmulatorLabel.PEDERSEN21_EXT,
        EmulatorLabel.PEDERSEN23_EXT,
        EmulatorLabel.CH24,
        EmulatorLabel.PEDERSEN23_EXT8,
    }:
        emulator = GPEmulator(
            archive=archive,
            training_set=training_set,
            emulator_label=emulator_label,
            drop_sim=drop_sim,
        )
    else:
        folder = {
            EmulatorLabel.CABAYOL23: "NNmodels/Cabayol23_Feb2024/",
            EmulatorLabel.CABAYOL23_EXTENDED: "NNmodels/Cabayol23_Feb2024/",
            EmulatorLabel.CABAYOL23_PLUS: "NNmodels/Cabayol23+/",
            EmulatorLabel.CABAYOL23_PLUS_EXTENDED: "NNmodels/Cabayol23+_extended/",
            EmulatorLabel.NYX_V0: "NNmodels/Nyxv0_Oct2023/",
            EmulatorLabel.NYX_ALPHAP: "NNmodels/Nyxap_Oct2023/",
            EmulatorLabel.NYX_ALPHAP_COV: "NNmodels/testing_models/",
        }.get(emulator_label, "")

        model_path = f"{folder}{emulator_label.value}{'_drop_sim' + drop_sim if drop_sim else ''}.pt"

        emulator = NNEmulator(
            archive=archive,
            training_set=training_set,
            emulator_label=emulator_label,
            model_path=model_path,
            drop_sim=drop_sim,
            train=False,
        )

    return emulator
