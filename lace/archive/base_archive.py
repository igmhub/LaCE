import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from lace.utils.exceptions import ExceptionList
from lace.utils.misc import split_string


class BaseArchive(object):
    """
    A base class for archiving and processing data.

    Methods:
        _set_labels(): Extract labels from self.data and set them as attributes.
        _average_over_samples(average="both"): Compute averages over phases, axes, or both.
        get_training_data(average=None, val_scaling=None, drop_sim=None, z_max=None):
            Retrieves training data based on provided flags.
        get_testing_data(sim_label, val_scaling=None, average=None, z_max=None):
            Retrieves testing data based on provided flags.

    """

    def _set_labels(self):
        """
        Extract labels from self.data and set them as attributes.

        Returns:
            None

        """

        list_labels = [
            "sim_label",
            "ind_snap",
            "ind_phase",
            "ind_axis",
            "ind_rescaling",
        ]

        # put measurements in arrays
        N = len(self.data)
        for label in list_labels:
            prop = []
            for ii in range(N):
                prop.append(self.data[ii][label])

            setattr(self, label, np.array(prop))

    def _average_over_samples(self, average="both"):
        """
        Compute averages over either phases, axes, or both.

        Args:
            average (str): Flag indicating the type of averaging. Valid options are:
                - "both": Compute averages over phases and axes.
                - "phases": Compute averages over phases while holding axes fixed.
                - "axes": Compute averages over axes while holding phases fixed.
                (default: "both")

        Returns:
            Averages

        """

        average_avail = ["both", "axes", "phases"]
        if average not in average_avail:
            msg = "Invalid average value. Available options:"
            raise ExceptionList(msg, average_avail)

        keys_avoid = [
            "sim_label",
            "ind_snap",
            "ind_phase",
            "ind_axis",
            "ind_rescaling",
            "cosmo_params",
        ]

        keys_spe = ["p1d_Mpc"]
        if self.also_P3D:
            keys_spe.append("k3d_Mpc")
            keys_spe.append("mu3d")
            keys_spe.append("p3d_Mpc")

        # get number of phases and axes
        n_phases = np.unique(self.ind_phase).shape[0]
        n_axes = np.unique(self.ind_axis).shape[0]

        # set loop over simulations
        if average == "both":
            tot_nsam = n_phases * n_axes
            loop = list(
                product(
                    np.unique(self.sim_label),
                    np.unique(self.ind_rescaling),
                    np.unique(self.ind_snap),
                )
            )
        elif average == "phases":
            tot_nsam = n_phases
            loop = list(
                product(
                    np.unique(self.sim_label),
                    np.unique(self.ind_rescaling),
                    np.unique(self.ind_snap),
                    np.arange(n_axes),
                )
            )
        elif average == "axes":
            tot_nsam = n_axes
            loop = list(
                product(
                    np.unique(self.sim_label),
                    np.unique(self.ind_rescaling),
                    np.unique(self.ind_snap),
                    np.arange(n_phases),
                )
            )
        nloop = len(loop)

        arch_av = []
        # iterate over loop
        for ind_loop in range(nloop):
            dict_av = {}
            if average == "both":
                sim_label, ind_rescaling, ind_snap = loop[ind_loop]
                ind_merge = np.argwhere(
                    (self.sim_label == sim_label)
                    & (self.ind_rescaling == ind_rescaling)
                    & (self.ind_snap == ind_snap)
                )[:, 0]

                dict_av["ind_axis"] = "average"
                dict_av["ind_phase"] = "average"
            elif average == "phases":
                sim_label, ind_rescaling, ind_snap, ind_axis = loop[ind_loop]
                ind_merge = np.argwhere(
                    (self.sim_label == sim_label)
                    & (self.ind_rescaling == ind_rescaling)
                    & (self.ind_snap == ind_snap)
                    & (self.ind_axis == ind_axis)
                )[:, 0]

                dict_av["ind_axis"] = ind_axis
                dict_av["ind_phase"] = "average"
            elif average == "axes":
                sim_label, ind_rescaling, ind_snap, ind_phase = loop[ind_loop]
                ind_merge = np.argwhere(
                    (self.sim_label == sim_label)
                    & (self.ind_rescaling == ind_rescaling)
                    & (self.ind_snap == ind_snap)
                    & (self.ind_phase == ind_phase)
                )[:, 0]

                dict_av["ind_axis"] = "average"
                dict_av["ind_phase"] = ind_phase

            # if no simulations fulfilling this criteria
            if ind_merge.shape[0] == 0:
                continue

            dict_av["sim_label"] = sim_label
            dict_av["ind_rescaling"] = ind_rescaling
            dict_av["ind_snap"] = ind_snap
            dict_av["cosmo_params"] = self.data[ind_merge[0]]["cosmo_params"]

            # list of available keys to merger
            key_list = self.data[ind_merge[0]].keys()
            # iterate over keys
            for key in key_list:
                if key in keys_avoid:
                    continue
                if key in keys_spe:
                    mean = np.zeros_like(self.data[ind_merge[0]][key])
                else:
                    mean = 0

                for imerge in range(tot_nsam):
                    if (key == "p1d_Mpc") | (key == "p3d_Mpc"):
                        mean += (
                            self.data[ind_merge[imerge]][key]
                            * self.data[ind_merge[imerge]]["mF"] ** 2
                        )
                    else:
                        mean += self.data[ind_merge[imerge]][key]

                if (key == "p1d_Mpc") | (key == "p3d_Mpc"):
                    dict_av[key] = mean / dict_av["mF"] ** 2 / tot_nsam
                else:
                    dict_av[key] = mean / tot_nsam

            arch_av.append(dict_av)

        return arch_av

    def get_training_data(
        self,
        emu_params,
        average=None,
        val_scaling=None,
        drop_sim=None,
        drop_z=None,
        z_max=None,
        verbose=False,
    ):
        """
        Retrieves the training data based on the provided flags.

        Parameters:
            emu_params (list): The parameters that must be defined for each
                element of the training data. There are intended to be emulator parameters.
            average (str, optional): The flag indicating the type of average computed.
            val_scaling (int or None, optional): The scaling value. Defaults to None.
            drop_sim (str or None, optional): The simulation to drop. Defaults to None.
            drop_z (str or None, optional): The red to drop. Defaults to None.
            z_max (int, float or None, optional): The maximum redshift. Defaults to None.

        Returns:
            List: The retrieved training data.

        Raises:
            TypeError: If the input arguments have invalid types.
            ExceptionList: If the input arguments contain invalid values.

        Notes:
            The retrieved training data is stored in the 'training_data' attribute of the parent class.

        """

        ## check input
        if isinstance(emu_params, list) == False:
            raise TypeError("emu_params must be a list")

        if isinstance(average, (str, type(None))) == False:
            raise TypeError("average must be a string or None")
        if average is None:
            average = self.training_average

        operations = split_string(average)
        operations_avail = ["axes", "phases", "both", "individual"]
        for operation in operations:
            if operation not in operations_avail:
                msg = "Invalid average value. Available options:"
                raise ExceptionList(msg, operations_avail)

        if isinstance(val_scaling, (int, float, type(None))) == False:
            raise TypeError("val_scaling must be an int or None")
        if val_scaling is None:
            val_scaling = self.training_val_scaling
        else:
            if val_scaling not in self.scalings_avail:
                msg = "Invalid val_scaling value. Available options:"
                raise ExceptionList(msg, self.scalings_avail)

        if isinstance(drop_sim, (str, type(None))) == False:
            raise TypeError("drop_sim must be a string or None")
        if drop_sim is not None:
            if drop_sim not in self.list_sim_cube:
                msg = "Invalid drop_sim value. Available options:"
                raise ExceptionList(msg, self.list_sim_cube)

        if drop_z is not None:
            if drop_z not in self.list_sim_redshifts:
                msg = "Invalid drop_z value. Available options:"
                raise ExceptionList(
                    msg, np.array(self.list_sim_redshifts).astype("str")
                )

        if isinstance(z_max, (int, float, type(None))) == False:
            raise TypeError("z_max must be a number or None")
        if z_max is None:
            z_max = self.training_z_max
        ## done

        ## put training points here
        training_data = []

        key_power = ["k_Mpc", "p1d_Mpc"]
        for operation in operations:
            if operation == "individual":
                arch_av = self.data
            else:
                arch_av = self._average_over_samples(average=operation)

            for ii in range(len(arch_av)):
                list_keys = list(arch_av[ii].keys())

                if drop_sim is None or drop_z is None:
                    mask = (
                        (arch_av[ii]["sim_label"] in self.list_sim_cube)
                        & (arch_av[ii]["sim_label"] != drop_sim)
                        & (arch_av[ii]["z"] != drop_z)
                        & (
                            (val_scaling == "all")
                            | (arch_av[ii]["val_scaling"] == val_scaling)
                        )
                        & (arch_av[ii]["z"] <= z_max)
                    )
                elif drop_sim is not None and drop_z is not None:
                    mask = (
                        (arch_av[ii]["sim_label"] in self.list_sim_cube)
                        & (
                            (arch_av[ii]["sim_label"] != drop_sim)
                            | (arch_av[ii]["z"] != drop_z)
                        )
                        & (
                            (val_scaling == "all")
                            | (arch_av[ii]["val_scaling"] == val_scaling)
                        )
                        & (arch_av[ii]["z"] <= z_max)
                    )

                if mask:
                    if all(x in list_keys or (x == 'A_UVB' and 'cosmo_params' in list_keys) for x in emu_params):
                        if any(
                            np.isnan(arch_av[ii][x]) for x in emu_params if x is not 'A_UVB'
                        ) | any(
                            np.any(np.isnan(arch_av[ii][x])) for x in key_power
                        ):                     
                            if verbose:
                                print(
                                    "Archive element "
                                    + str(ii)
                                    + " contains nans"
                                )
                        else:
                            training_data.append(arch_av[ii])
                    else:
                        if verbose:
                            print(
                                "Archive element "
                                + str(ii)
                                + " does not contain all emulator parameters"
                            )

        return training_data

    def get_testing_data(
        self,
        sim_label,
        ind_rescaling=None,
        z_min=None,
        z_max=None,
        emu_params=None,
        verbose=False,
    ):
        """
        Retrieves the testing data based on the provided flags.

        Parameters:
            sim_label (str): The simulation label.
            val_scaling (int or None, optional): The scaling value. Defaults to None.
            z_min (int, float or None, optional): The minimum redshift (included). Defaults to None.
            z_max (int, float or None, optional): The maximum redshift (included). Defaults to None.
            emu_params (list, optional): The parameters that must be defined for each
                element of the training data. There are intended to be emulator parameters.
                Only relevant if evaluating the emulator for the testing data.

        Returns:
            List: The retrieved testing data.

        Raises:
            TypeError: If the input arguments have invalid types.
            ExceptionList: If the input arguments contain invalid values.

        """

        ## check input

        if isinstance(sim_label, str) == False:
            raise TypeError("sim_label must be a string")
        if sim_label not in self.list_sim:
            msg = "Invalid sim_label value. Available options:"
            raise ExceptionList(msg, self.list_sim)

        if isinstance(ind_rescaling, (int, type(None))) == False:
            raise TypeError("val_scaling must be an int or None")
        if ind_rescaling is None:
            if sim_label == "nyx_central":
                ind_rescaling = 1
            else:
                ind_rescaling = self.testing_ind_rescaling
        else:
            if ind_rescaling not in self.scalings_avail:
                msg = "Invalid ind_rescaling value. Available options:"
                raise ExceptionList(msg, self.scalings_avail)

        if isinstance(z_max, (int, float, type(None))) == False:
            raise TypeError("z_max must be a number or None")
        if z_max is None:
            z_max = self.testing_z_max

        if isinstance(z_min, (int, float, type(None))) == False:
            raise TypeError("z_min must be a number or None")
        if z_min is None:
            z_min = self.testing_z_min

        if isinstance(emu_params, (list, type(None))) == False:
            raise TypeError("emu_params must be a list or None")
        ## done

        ## put testing points here
        testing_data = []
        # we use the average of axes (and phases is available), but
        # could implement other options in the future
        arch_av = self._average_over_samples(average="both")

        key_power = ["k_Mpc", "p1d_Mpc"]
        for ii in range(len(arch_av)):
            list_keys = list(arch_av[ii].keys())
            mask = (
                (arch_av[ii]["sim_label"] == sim_label)
                & (arch_av[ii]["ind_rescaling"] == ind_rescaling)
                & (arch_av[ii]["z"] <= z_max)
                & (arch_av[ii]["z"] >= z_min)
            )
            if mask:
                if emu_params is None:
                    testing_data.append(arch_av[ii])
                elif all(x in list_keys for x in emu_params):
                    if any(np.isnan(arch_av[ii][x]) for x in emu_params) | any(
                        np.any(np.isnan(arch_av[ii][x])) for x in key_power
                    ):
                        if verbose:
                            print(
                                "Archive element " + str(ii) + " contains nans"
                            )
                    else:
                        testing_data.append(arch_av[ii])
                else:
                    if verbose:
                        print(
                            "Archive element "
                            + str(ii)
                            + " does not contain all emulator parameters"
                        )

        return testing_data

    def plot_samples(self, param_1, param_2):
        """For parameter pair (param1,param2), plot each point in the archive"""

        emu_data = self.get_training_data(emu_params=[param_1, param_2])
        Nemu = len(emu_data)

        # figure out values of param_1,param_2 in archive
        emu_1 = np.array([emu_data[i][param_1] for i in range(Nemu)])
        emu_2 = np.array([emu_data[i][param_2] for i in range(Nemu)])

        emu_z = np.array([emu_data[i]["z"] for i in range(Nemu)])
        zmin = min(emu_z)
        zmax = max(emu_z)
        plt.scatter(emu_1, emu_2, c=emu_z, s=1, vmin=zmin, vmax=zmax)
        cbar = plt.colorbar()
        cbar.set_label("Redshift", labelpad=+1)
        plt.xlabel(param_1)
        plt.ylabel(param_2)
        plt.show()

        return

    def plot_3D_samples(self, param_1, param_2, param_3):
        """For parameter trio (param1,param2,param3), plot each point in the archive"""
        from mpl_toolkits import mplot3d

        emu_data = self.get_training_data(
            emu_params=[param_1, param_2, param_3]
        )
        Nemu = len(emu_data)

        # figure out values of param_1,param_2 in archive
        emu_1 = np.array([emu_data[i][param_1] for i in range(Nemu)])
        emu_2 = np.array([emu_data[i][param_2] for i in range(Nemu)])
        emu_3 = np.array([emu_data[i][param_3] for i in range(Nemu)])

        emu_z = np.array([emu_data[i]["z"] for i in range(Nemu)])
        zmin = min(emu_z)
        zmax = max(emu_z)

        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.scatter3D(emu_1, emu_2, emu_3, c=emu_z, cmap="brg", s=8)
        ax.set_xlabel(param_1)
        ax.set_ylabel(param_2)
        ax.set_zlabel(param_3)
        plt.show()

        return

    def print_entry(self, entry):
        """
        Print basic information about a particular entry in the archive.

        Parameters:
            self (object): The object instance.
            entry (int): The index of the entry to print.

        Returns:
            None

        Raises:
            ValueError: If the provided entry index is out of range.

        """

        if entry >= len(self.data):
            raise ValueError("{} entry does not exist in archive".format(entry))

        list_print = ["z"] + self.emu_params

        info = "entry = {}".format(entry)
        for key in list_print:
            info += ", {} = {:.4f}".format(key, self.data[entry][key])

        print(info)
