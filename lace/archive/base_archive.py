import numpy as np
from itertools import product
from lace.archive.exceptions import ExceptionList
from lace.emulator.utils import split_string


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
            "cosmo_pars",
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
        self, average=None, val_scaling=None, drop_sim=None, z_max=None
    ):
        """
        Retrieves the training data based on the provided flags.

        Parameters:
            average (str, optional): The flag indicating the type of average computed.
            val_scaling (int or None, optional): The scaling value. Defaults to None.
            drop_sim (str or None, optional): The simulation to drop. Defaults to None.
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

        if isinstance(z_max, (int, float, type(None))) == False:
            raise TypeError("z_max must be a number or None")
        if z_max is None:
            z_max = self.training_z_max
        ## done

        ## put training points here
        training_data = []

        for operation in operations:
            if operation == "individual":
                arch_av = self.data
            else:
                arch_av = self._average_over_samples(average=operation)

            for ii in range(len(arch_av)):
                list_keys = list(arch_av[ii].keys())
                mask = (
                    (arch_av[ii]["sim_label"] in self.list_sim_cube)
                    & (arch_av[ii]["sim_label"] != drop_sim)
                    & (
                        (val_scaling == "all")
                        | (arch_av[ii]["val_scaling"] == val_scaling)
                    )
                    & (arch_av[ii]["z"] <= z_max)
                    & (x in list_keys for x in self.emu_params)
                )
                if mask:
                    training_data.append(arch_av[ii])

        return training_data

    def get_testing_data(self, sim_label, val_scaling=None, z_max=None):
        """
        Retrieves the testing data based on the provided flags.

        Parameters:
            sim_label (str): The simulation label.
            val_scaling (int or None, optional): The scaling value. Defaults to None.
            z_max (int, float or None, optional): The maximum redshift. Defaults to None.

        Returns:
            List: The retrieved testing data.

        Raises:
            TypeError: If the input arguments have invalid types.
            ExceptionList: If the input arguments contain invalid values.

        """

        ## check input
        if isinstance(val_scaling, (int, float, type(None))) == False:
            raise TypeError("val_scaling must be an int or None")
        if val_scaling is None:
            val_scaling = self.testing_val_scaling
        else:
            if val_scaling not in self.scalings_avail:
                msg = "Invalid val_scaling value. Available options:"
                raise ExceptionList(msg, self.scalings_avail)

        if isinstance(z_max, (int, float, type(None))) == False:
            raise TypeError("z_max must be a number or None")
        if z_max is None:
            z_max = self.testing_z_max
        ## done

        ## put testing points here
        testing_data = []
        # we use the average of axes (and phases is available), but
        # could implement other options in the future
        arch_av = self._average_over_samples(average="both")

        for ii in range(len(arch_av)):
            list_keys = list(arch_av[ii].keys())
            mask = (
                (arch_av[ii]["sim_label"] == sim_label)
                & (arch_av[ii]["val_scaling"] == val_scaling)
                & (arch_av[ii]["z"] <= z_max)
                & (x in list_keys for x in self.emu_params)
            )
            if mask:
                testing_data.append(arch_av[ii])

        return testing_data
