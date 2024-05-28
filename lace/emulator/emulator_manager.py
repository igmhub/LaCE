"""Emulator manager"""

from lace.emulator.nn_emulator import NNEmulator
from lace.emulator.gp_emulator import GPEmulator


def emulators_supported():
    """List of emulators supported
    LaCE emulators:
        Pedersen21: GPEmulator used in Pedersen21, k-bin emulator.
            Superseed by Pedersen21_ext
        Pedersen21_ext: GPEmulator like Pedersen21 but using Cabayol23 postprocessing.
        Pedersen21_ext8: GPEmulator like Pedersen21_ext but accessing smaller scales.
        Pedersen23: GPEmulator used in Pedersen23, polynomial emulator.
            Superseed by Pedersen23_ext
        Pedersen23_ext: GPEmulator like Pedersen23 but using Cabayol23 postprocessing.
            Recommended GP emulator
        Pedersen23_ext8: GPEmulator like Pedersen23_ext but accessing smaller scales
            Recommended GP emulator for accessing small scales
        CH24: GPEmulator based on non-linear smoothing
        Cabayol23: NNEmulator used in Cabayol23, polynomial emulator.
            Superseed by Cabayol23+
        Cabayol23+: NNEmulator like Cabayol23 but using better architecture
            Recommended NN emulator
        Cabayol23_extended: NNEmulator used in Cabayol23 accessing smaller scales than Cabayol23
            Superseed by Cabayol23_extended+
        Cabayol23+_extended: NNEmulator like Cabayol23_extended but using better architecture
            Recommended NN emulator for accessing small scales

    Nyx emulators:
        Nyx_v0: NNEmulator using amplitude and slope, polynomial emulator.
            Superseed by Nyx_alphap
        Nyx_alphap: NNEmulator using amplitude, slope, and running, polynomial emulator.
            Recommended emulator

    """

    emulators_supported = [
        "Pedersen21",
        "Pedersen21_ext", 
        "Pedersen23",
        "Pedersen23_ext",
        "CH24",
        "Cabayol23",
        "Cabayol23_extended",
        "Cabayol23+",
        "Cabayol23+_extended",
        "Pedersen21_ext8",
        "Pedersen23_ext8",
        "Nyx_v0",
        "Nyx_alphap",
    ]
    return emulators_supported


def set_emulator(emulator_label, archive=None, drop_sim=None):
    """Loads emulator

    Parameters:
    ----------
    emulator_label: str
        Name of the emulator
    archive: object, optional
        Archive with the emulator training data. If None, an archive
        corresponding to the emulator_label will be loaded
    drop_sim: str, optional
        Drop a specific simulation from the training set

    Returns:
    -------
    emulator: object
    Loaded emulator
    """

    if emulator_label not in emulators_supported():
        msg = (
            "Emulator "
            + emulator_label
            + " not supported. Supported emulators are "
            + emulators_supported
        )
        raise ValueError(msg)

    if (emulator_label == "Pedersen21") | (emulator_label == "Pedersen23"):
        if archive is None:
            emulator = GPEmulator(
                training_set="Pedersen21",
                emulator_label=emulator_label,
                drop_sim=drop_sim,
            )
        else:
            if archive.data[0]["sim_label"][:3] != "mpg":
                raise ValueError(
                    "WARNING: training data in archive are not mpg sims"
                )

            emulator = GPEmulator(
                archive=archive,
                emulator_label=emulator_label,
                drop_sim=drop_sim,
            )
    elif (
        (emulator_label == "Pedersen21_ext")
        | (emulator_label == "Pedersen23_ext")
        | (emulator_label == "CH24")
        | (emulator_label == "Pedersen23_ext8")
        | (emulator_label == "Pedersen23_ext8")
    ):
        if archive is None:
            emulator = GPEmulator(
                training_set="Cabayol23",
                emulator_label=emulator_label,
                drop_sim=drop_sim,
            )
        else:
            if archive.data[0]["sim_label"][:3] != "mpg":
                raise ValueError(
                    "WARNING: training data in archive are not mpg sims"
                )

            emulator = GPEmulator(
                archive=archive,
                emulator_label=emulator_label,
                drop_sim=drop_sim,
            )
    elif (
        (emulator_label == "Cabayol23")
        | (emulator_label == "Cabayol23+")
        | (emulator_label == "Cabayol23_extended")
        | (emulator_label == "Cabayol23+_extended")
    ):
        if (emulator_label == "Cabayol23") | (
            emulator_label == "Cabayol23_extended"
        ):
            folder = "NNmodels/Cabayol23_Feb2024/"
        elif emulator_label == "Cabayol23+":
            folder = "NNmodels/Cabayol23+/"
        elif emulator_label == "Cabayol23+_extended":
            folder = "NNmodels/Cabayol23+_extended/"

        if drop_sim is None:
            model_path = folder + emulator_label + ".pt"
        else:
            model_path = (
                folder + emulator_label + "_drop_sim_" + drop_sim + ".pt"
            )
        if archive is None:
            emulator = NNEmulator(
                training_set="Cabayol23",
                emulator_label=emulator_label,
                model_path=model_path,
                drop_sim=drop_sim,
                train=False,
            )
        else:
            if archive.data[0]["sim_label"][:3] != "mpg":
                raise ValueError(
                    "WARNING: training data in archive are not mpg sims"
                )

            emulator = NNEmulator(
                archive=archive,
                training_set="Cabayol23",
                emulator_label=emulator_label,
                model_path=model_path,
                drop_sim=drop_sim,
                train=False,
            )
    elif ((emulator_label == "Nyx_v0")|
        | (emulator_label == "Nyx_alphap")
    ):
        if emulator_label == "Nyx_v0":
            folder = "NNmodels/Nyxv0_Oct2023/"
        elif emulator_label == "Nyx_alphap":
            folder = "NNmodels/Nyxap_Oct2023/"
            
        if drop_sim is None:
            model_path = folder + emulator_label + ".pt"
        else:
            model_path = (
                folder + emulator_label + "_drop_sim_" + drop_sim + ".pt"
            )
        if archive is None:
            emulator = NNEmulator(
                training_set="Nyx23_Oct2023",
                emulator_label=emulator_label,
                model_path=model_path,
                drop_sim=drop_sim,
                train=False,
            )
        else:
            if archive.data[0]["sim_label"][:3] != "nyx":
                raise ValueError(
                    "WARNING: training data in archive are not nyx sims"
                )

            emulator = NNEmulator(
                archive=archive,
                training_set="Nyx23_Oct2023",
                emulator_label=emulator_label,
                model_path=model_path,
                drop_sim=drop_sim,
                train=False,
            )
    else:
        raise ValueError(emulator_label + " not supported")

    return emulator
