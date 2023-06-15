import numpy as np
import copy
import sys
import os

from lace.emulator.archives.pnd_archive import archivePND
from lace.emulator.archives.p1d_archive_Nyx import archiveP1D_Nyx


def get_sim_option_list(sim_suite):
    """
    Get the simulation option list based on the specified simulation suite.

    Args:
        sim_suite (str): Name of the simulation suite.

    Returns:
        tuple: A tuple containing the following elements:
            - sim_option_list (list): List of simulation options available for the specified simulation suite.
            - sim_especial_list (list): List of special simulation options available for the specified simulation suite.
            - sim_option_dict (dict): Dictionary mapping simulation options to their corresponding values.
    Note:
        To be modified for new suite.

    """
    if (
        (sim_suite == "Pedersen21")
        | (sim_suite == "Cabayol23")
        | (sim_suite == "768_768")
    ):
        sim_option_list = [
            "growth",
            "neutrinos",
            "central",
            "seed",
            "curved",
            "reionization",
            "running",
        ]
        sim_especial_list = sim_option_list.copy()

        sim_option_dict = {
            "growth": 100,
            "neutrinos": 101,
            "central": 30,
            "seed": 102,
            "curved": 103,
            "reionization": 104,
            "running": 105,
        }

        for ii in range(31):
            sim_option_list.append(ii)
            sim_option_dict[ii] = ii

    return sim_option_list, sim_especial_list, sim_option_dict


class Archive(archivePND, archiveP1D_Nyx):
    def __init__(
        self,
        sim_suite="Cabayol23",
    ):
        sim_suite_all = ["Pedersen21", "Cabayol23", "768_768", "Nyx"]
        try:
            if sim_suite in sim_suite_all:
                pass
            else:
                print(
                    "Invalid sim_suite value. Available options: ",
                    sim_suite_all,
                )
                raise
        except:
            print("An error occurred while checking the sim_suite value.")
            raise

        if (
            (sim_suite == "Pedersen21")
            | (sim_suite == "Cabayol23")
            | (sim_suite == "768_768")
        ):
            archivePND.__init__(self, sim_suite=sim_suite)
        elif sim_suite == "Nyx":
            # to be updated
            archiveP1D_Nyx.__init__(self)
