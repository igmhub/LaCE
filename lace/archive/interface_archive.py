import numpy as np
import copy
import sys
import os

from lace.archive.pnd_archive import archivePND
from lace.archive.pnd_archive import get_sim_option_list as lace_get_sim_option_list
from lace.archive.p1d_archive_Nyx import archiveP1D_Nyx

# from lace.archive.p1d_archive_Nyx import get_sim_option_list as nyx_get_sim_option_list


class Archive(archivePND, archiveP1D_Nyx):
    """
    Archive class for storing simulation data.

    The Archive class inherits from archivePND and archiveP1D_Nyx classes.

    Parameters:
        sim_suite (str, optional): The simulation suite.
            Available options are "Pedersen21", "Cabayol23", "768_768", and "Nyx".
            Defaults to "Cabayol23".

    Raises:
        ValueError: If the provided sim_suite value is not valid.

    """

    def __init__(self, verbose=False):
        self.sim_suite_all = ["Pedersen21", "Cabayol23", "768_768"]
        if verbose:
            print(
                "The list of simulation suites available is:",
                self.sim_suite_all,
            )

    def get_sims_available(self, sim_suite="Cabayol23"):
        if (
            (sim_suite == "Pedersen21")
            | (sim_suite == "Cabayol23")
            | (sim_suite == "768_768")
        ):
            sim_option_list, _, _ = lace_get_sim_option_list(sim_suite)
            print(sim_option_list)

        elif sim_suite == "Nyx":
            print("Not implemented yet")
            # nyx_get_sim_option_list()

    def get_training_data(self, training_set="Cabayol23"):
        """
        Get training data for emulator

        Parameters:
            training_set (str, optional): The simulation suite. Defaults to "Cabayol23".

        Raises:
            ValueError: If the provided training_set value is not valid.

        """

        try:
            if training_set in self.sim_suite_all:
                pass
            else:
                print(
                    "Invalid training_set value. Available options: ",
                    self.sim_suite_all,
                )
                raise
        except:
            print("An error occurred while checking the training_set value.")
            raise

        if (
            (training_set == "Pedersen21")
            | (training_set == "Cabayol23")
            | (training_set == "768_768")
        ):
            archivePND.__init__(self, sim_suite=training_set)
            archivePND.get_training_data(self)

        elif training_set == "Nyx":
            print("Not implemented yet")
            # # to be updated
            # archiveP1D_Nyx.__init__(self)
            # archiveP1D_Nyx.get_training_data(self)

    def get_testing_data(
        self,
        testing_set="Cabayol23",
        pick_sim="central",
        tau_scaling=1,
    ):
        """
        Get testing data for emulator

        Parameters:
            testing_set (str, optional): The simulation suite. Defaults to "Cabayol23".

        Raises:
            ValueError: If the provided testing_set value is not valid.

        """

        try:
            if testing_set in self.sim_suite_all:
                pass
            else:
                print(
                    "Invalid testing_set value. Available options: ",
                    self.sim_suite_all,
                )
                raise
        except:
            print("An error occurred while checking the testing_set value.")
            raise

        if (
            (testing_set == "Pedersen21")
            | (testing_set == "Cabayol23")
            | (testing_set == "768_768")
        ):
            archivePND.__init__(self, sim_suite=testing_set, pick_sim=pick_sim)
            archivePND.get_testing_data(self, tau_scaling=tau_scaling)

        elif testing_set == "Nyx":
            print("Not implemented yet")
            # # to be updated
            # archiveP1D_Nyx.__init__(self)
            # archiveP1D_Nyx.get_testing_data(self)

    def print_entry(self, entry, fiducial_keys=True):
        """
        Print basic information about a particular entry in the archive.

        Parameters:
            self (object): The object instance.
            entry (int): The index of the entry to print.
            fiducial_keys (bool or list, optional): If True, the default fiducial keys will be used.
                If a list is provided, it will be used as the keys. Defaults to True.

        Returns:
            None

        Raises:
            ValueError: If the provided entry index is out of range.

        """

        if fiducial_keys is True:
            keys = [
                "z",
                "Delta2_p",
                "n_p",
                "alpha_p",
                "f_p",
                "mF",
                "sigT_Mpc",
                "gamma",
                "kF_Mpc",
            ]
        else:
            keys = fiducial_keys

        if entry >= len(self.data):
            raise ValueError("{} entry does not exist in archive".format(entry))

        data = self.data[entry]
        info = "entry = {}".format(entry)
        for key in keys:
            info += ", {} = {:.4f}".format(key, data[key])
        print(info)

    def plot_samples(self, param_1, param_2, tau_scalings=True, temp_scalings=True):
        """
        Plot each point in the archive for the specified parameter pair (param_1, param_2).

        Parameters:
            self (object): The object instance.
            param_1 (str): The name of the first parameter.
            param_2 (str): The name of the second parameter.
            tau_scalings (bool, optional): Flag to mask post-process scalings related to tau. Defaults to True.
            temp_scalings (bool, optional): Flag to mask post-process scalings related to temperature. Defaults to True.

        Returns:
            None

        """

        import matplotlib.pyplot as plt

        # mask post-process scalings (optional)
        emu_data = self.data
        Nemu = len(emu_data)
        if not tau_scalings:
            mask_tau = [x["scale_tau"] == 1.0 for x in emu_data]
        else:
            mask_tau = [True] * Nemu
        if not temp_scalings:
            mask_temp = [
                (x["scale_T0"] == 1.0) & (x["scale_gamma"] == 1.0) for x in emu_data
            ]
        else:
            mask_temp = [True] * Nemu

        # figure out values of param_1,param_2 in archive
        emu_1 = np.array(
            [emu_data[i][param_1] for i in range(Nemu) if (mask_tau[i] & mask_temp[i])]
        )
        emu_2 = np.array(
            [emu_data[i][param_2] for i in range(Nemu) if (mask_tau[i] & mask_temp[i])]
        )
        emu_z = np.array(
            [emu_data[i]["z"] for i in range(Nemu) if (mask_tau[i] & mask_temp[i])]
        )
        zmin = min(emu_z)
        zmax = max(emu_z)
        plt.scatter(emu_1, emu_2, c=emu_z, s=1, vmin=zmin, vmax=zmax)
        cbar = plt.colorbar()
        cbar.set_label("Redshift", labelpad=+1)
        plt.xlabel(param_1)
        plt.ylabel(param_2)
        plt.show()

    def plot_3D_samples(
        self, param_1, param_2, param_3, tau_scalings=True, temp_scalings=True
    ):
        """
        Plot each point in the archive for the specified parameter trio (param_1, param_2, param_3) in 3D.

        Parameters:
            self (object): The object instance.
            param_1 (str): The name of the first parameter.
            param_2 (str): The name of the second parameter.
            param_3 (str): The name of the third parameter.
            tau_scalings (bool, optional): Flag to mask post-process scalings related to tau. Defaults to True.
            temp_scalings (bool, optional): Flag to mask post-process scalings related to temperature. Defaults to True.

        Returns:
            None

        """

        from mpl_toolkits import mplot3d
        import matplotlib.pyplot as plt

        # mask post-process scalings (optional)
        emu_data = self.data
        Nemu = len(emu_data)
        if not tau_scalings:
            mask_tau = [x["scale_tau"] == 1.0 for x in emu_data]
        else:
            mask_tau = [True] * Nemu
        if not temp_scalings:
            mask_temp = [
                (x["scale_T0"] == 1.0) & (x["scale_gamma"] == 1.0) for x in emu_data
            ]
        else:
            mask_temp = [True] * Nemu

        # figure out values of param_1,param_2 in archive
        emu_1 = np.array(
            [emu_data[i][param_1] for i in range(Nemu) if (mask_tau[i] & mask_temp[i])]
        )
        emu_2 = np.array(
            [emu_data[i][param_2] for i in range(Nemu) if (mask_tau[i] & mask_temp[i])]
        )
        emu_3 = np.array(
            [emu_data[i][param_3] for i in range(Nemu) if (mask_tau[i] & mask_temp[i])]
        )

        emu_z = np.array(
            [emu_data[i]["z"] for i in range(Nemu) if (mask_tau[i] & mask_temp[i])]
        )
        zmin = min(emu_z)
        zmax = max(emu_z)
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.scatter3D(emu_1, emu_2, emu_3, c=emu_z, cmap="brg", s=8)
        ax.set_xlabel(param_1)
        ax.set_ylabel(param_2)
        ax.set_zlabel(param_3)
        plt.show()
