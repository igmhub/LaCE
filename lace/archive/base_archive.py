import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from typing import List, Dict, Union, Optional, Any, Tuple
from numpy.typing import NDArray

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

    def _set_labels(self) -> None:
        """Extract labels from self.data and set them as attributes."""
        list_labels = [
            "sim_label",
            "ind_snap",
            "ind_phase",
            "ind_axis",
            "ind_rescaling",
        ]

        # Put measurements in arrays
        N = len(self.data)
        for label in list_labels:
            prop = [self.data[ii][label] for ii in range(N)]
            setattr(self, label, np.array(prop))

    def _average_over_samples(self, 
                              average: str = "both", 
                              drop_axis: Optional[List[int]] = None) -> List[Dict]:
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
            raise ExceptionList("Invalid average value. Available options:", average_avail)

        keys_avoid = [
            "sim_label", "ind_snap", "ind_phase", "ind_axis",
            "ind_rescaling", "cosmo_params", "star_params"
        ]

        keys_spe = ["p1d_Mpc"]
        if self.also_P3D:
            keys_spe.extend(["k3d_Mpc", "mu3d", "p3d_Mpc"])

        # Get number of phases and axes
        n_phases = len(np.unique(self.ind_phase))
        n_axes = len(np.unique(self.ind_axis))

        # Set loop over simulations
        base_loop = [
            np.unique(self.sim_label),
            np.unique(self.ind_rescaling),
            np.unique(self.ind_snap)
        ]
        
        if average == "both":
            loop = list(product(*base_loop))
        elif average == "phases":
            loop = list(product(*base_loop, range(n_axes)))
        else:  # average == "axes"
            loop = list(product(*base_loop, range(n_phases)))

        arch_av = []
        # Iterate over loop
        for loop_vals in loop:
            dict_av: Dict[str, Any] = {}
            
            if average == "both":
                sim_label, ind_rescaling, ind_snap = loop_vals
                mask = (
                    (self.sim_label == sim_label) &
                    (self.ind_rescaling == ind_rescaling) &
                    (self.ind_snap == ind_snap)
                )
                if drop_axis is not None:
                    mask &= ~np.isin(self.ind_axis, drop_axis)
                ind_merge = np.where(mask)[0]

                dict_av.update({
                    "ind_axis": "average",
                    "ind_phase": "average"
                })
            elif average == "phases":
                sim_label, ind_rescaling, ind_snap, ind_axis = loop_vals
                mask = (
                    (self.sim_label == sim_label) &
                    (self.ind_rescaling == ind_rescaling) &
                    (self.ind_snap == ind_snap) &
                    (self.ind_axis == ind_axis)
                )
                if drop_axis is not None:
                    mask &= ~np.isin(self.ind_axis, drop_axis)
                ind_merge = np.where(mask)[0]

                dict_av.update({
                    "ind_axis": ind_axis,
                    "ind_phase": "average"
                })
            else:  # average == "axes"
                sim_label, ind_rescaling, ind_snap, ind_phase = loop_vals
                mask = (
                    (self.sim_label == sim_label) &
                    (self.ind_rescaling == ind_rescaling) &
                    (self.ind_snap == ind_snap) &
                    (self.ind_phase == ind_phase)
                )
                if drop_axis is not None:
                    mask &= ~np.isin(self.ind_axis, drop_axis)
                ind_merge = np.where(mask)[0]

                dict_av.update({
                    "ind_axis": "average",
                    "ind_phase": ind_phase
                })

            # Skip if no matching simulations
            if len(ind_merge) == 0:
                continue

            dict_av.update({
                "sim_label": sim_label,
                "ind_rescaling": ind_rescaling,
                "ind_snap": ind_snap,
                "cosmo_params": self.data[ind_merge[0]]["cosmo_params"],
                "star_params": self.data[ind_merge[0]]["star_params"]
            })

            # Average over available keys
            for key in self.data[ind_merge[0]].keys():
                if key in keys_avoid:
                    continue

                if key in keys_spe:
                    mean = np.zeros_like(self.data[ind_merge[0]][key])
                else:
                    mean = 0

                for idx in ind_merge:
                    if key in ("p1d_Mpc", "p3d_Mpc"):
                        mean += self.data[idx][key] * self.data[idx]["mF"] ** 2
                    else:
                        mean += self.data[idx][key]

                if key in ("p1d_Mpc", "p3d_Mpc"):
                    dict_av[key] = mean / dict_av["mF"] ** 2 / len(ind_merge)
                else:
                    dict_av[key] = mean / len(ind_merge)

            arch_av.append(dict_av)

        return arch_av

    def get_training_data(
        self,
        emu_params: List[str],
        average: Optional[str] = None,
        val_scaling: Optional[Union[int, float]] = None,
        drop_sim: Optional[Union[str, List[str]]] = None,
        drop_z: Optional[Union[float, List[float]]] = None,
        drop_snap: Optional[Union[str, List[str]]] = None,
        drop_axis: Optional[Union[int, List[int]]] = None,
        z_min: Optional[float] = None,
        z_max: Optional[float] = None,
        verbose: bool = False
    ) -> List[Dict]:
        """
        Retrieves the training data based on the provided flags.

        Parameters:
            emu_params (list): The parameters that must be defined for each
                element of the training data. There are intended to be emulator parameters.
            average (str, optional): The flag indicating the type of average computed.
            val_scaling (int or None, optional): The scaling value. Defaults to None.
            drop_sim (str, list, or None, optional): The simulation to drop. Defaults to None.
            drop_z (str, list, or None, optional): The red to drop. Defaults to None.
            drop_snap (str, list, or None, optional): The snapshot to drop. Defaults to None.
            z_min (int, float or None, optional): The minimum redshift. Defaults to None.
            z_max (int, float or None, optional): The maximum redshift. Defaults to None.

        Returns:
            List: The retrieved training data.

        Raises:
            TypeError: If the input arguments have invalid types.
            ExceptionList: If the input arguments contain invalid values.

        Notes:
            The retrieved training data is stored in the 'training_data' attribute of the parent class.

        """
        average = average or self.training_average

        operations = split_string(average)
        operations_avail = ["axes", "phases", "both", "individual"]
        invalid_ops = set(operations) - set(operations_avail)
        if invalid_ops:
            raise ExceptionList("Invalid average value. Available options:", operations_avail)

        if val_scaling is None:
            val_scaling = self.training_val_scaling
        else:
            if val_scaling not in self.scalings_avail:
                msg = "Invalid val_scaling value. Available options:"
                raise ExceptionList(msg, self.scalings_avail)

        # Convert single values to lists for consistent handling
        drop_sim = [drop_sim] if not isinstance(drop_sim, list) else drop_sim
        drop_snap = [drop_snap] if not isinstance(drop_snap, list) else drop_snap
        drop_z = [drop_z] if not isinstance(drop_z, list) else drop_z
        drop_axis = [drop_axis] if not isinstance(drop_axis, list) else drop_axis

        # Validate inputs
        if drop_sim[0] is not None:
            invalid_sims = set(drop_sim) - set(self.list_sim_cube)
            if invalid_sims:
                raise ExceptionList("Invalid drop_sim value(s). Available options:", self.list_sim_cube)

        z_max = z_max if z_max is not None else self.training_z_max
        z_min = z_min if z_min is not None else self.training_z_min

        training_data = []
        key_power = ["k_Mpc", "p1d_Mpc"]

        for operation in operations:
            arch_av = self.data if operation == "individual" else self._average_over_samples(
                average=operation, 
                drop_axis=drop_axis
            )

            for ii in range(len(arch_av)):
                list_keys = list(arch_av[ii].keys())

                mask = (
                    (arch_av[ii]["sim_label"] in self.list_sim_cube)
                    & (
                        (arch_av[ii]["sim_label"] not in drop_sim)
                        & (arch_av[ii]["z"] not in drop_z)
                        & (arch_av[ii]["ind_axis"] not in drop_axis)
                    )
                    & (
                        (
                            arch_av[ii]["sim_label"]
                            + "_"
                            + arch_av[ii]["z"].astype("str")
                            not in drop_snap
                        )
                    )
                    & (
                        (val_scaling == "all")
                        | (arch_av[ii]["val_scaling"] == val_scaling)
                    )
                    & (arch_av[ii]["z"] <= z_max)
                    & (arch_av[ii]["z"] >= z_min)
                )

                if mask:
                    if all(
                        x in list_keys
                        or (
                            x in ["A_lya", "n_lya", "omega_m", "H_0", "A_UVB"]
                            and "cosmo_params" in list_keys
                        )
                        for x in emu_params
                    ):
                        if any(
                            np.isnan(arch_av[ii][x])
                            for x in emu_params
                            if x
                            not in ["A_lya", "n_lya", "omega_m", "H_0", "A_UVB"]
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
        sim_label: str,
        ind_rescaling: Optional[int] = None,
        z_min: Optional[float] = None,
        z_max: Optional[float] = None,
        drop_axis: Optional[Union[int, List[int]]] = None,
        emu_params: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ) -> List[Dict[str, Any]]:
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
        if sim_label not in self.list_sim:
            raise ExceptionList("Invalid sim_label value. Available options:", self.list_sim)

        if ind_rescaling is None:
            ind_rescaling = 1 if sim_label == "nyx_central" else self.testing_ind_rescaling
        elif ind_rescaling not in self.scalings_avail:
            raise ExceptionList("Invalid ind_rescaling value. Available options:", self.scalings_avail)

        # Validate drop_axis
        drop_axis_list = [drop_axis] if not isinstance(drop_axis, list) else drop_axis
        if drop_axis_list[0] is not None:
            invalid_axes = [axis for axis in drop_axis_list if axis not in self.list_sim_axes]
            if invalid_axes:
                raise ExceptionList("Invalid drop_axis value(s). Available options:", 
                                  self.list_sim_axes.astype("str"))

        # Set z bounds
        z_max = z_max if z_max is not None else self.testing_z_max
        z_min = z_min if z_min is not None else self.testing_z_min

        ## done

        ## put testing points here
        testing_data = []
        # we use the average of axes (and phases is available), but
        # could implement other options in the future
        arch_av = self._average_over_samples(
            average="both", drop_axis=drop_axis
        )

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
                # elif all(x in list_keys for x in emu_params):
                elif all(
                    x in list_keys
                    or (
                        isinstance(list_keys, dict)
                        and x in list_keys.get("cosmo", [])
                    )
                    for x in emu_params.keys()
                ):
                    print("activated")
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

    def plot_samples(self, 
                     param_1: str, 
                     param_2: str, 
                     param_3: Optional[str] = None) -> None:
        """
        For parameter pair (param1,param2), plot each point in the archive colored by redshift.
        """

        emu_data = self.get_training_data(emu_params=[param_1, param_2])
        
        # Extract arrays efficiently using list comprehension
        emu_vals = np.array([[d[param_1], d[param_2], d["z"]] for d in emu_data])
        emu_1, emu_2, emu_z = emu_vals.T

        # Plot scatter with redshift coloring
        plt.scatter(emu_1, emu_2, c=emu_z, s=1, vmin=emu_z.min(), vmax=emu_z.max())
        cbar = plt.colorbar()
        cbar.set_label("Redshift", labelpad=+1)
        plt.xlabel(param_1)
        plt.ylabel(param_2)
        plt.show()

    def plot_3D_samples(self, 
                        param_1: str, 
                        param_2: str, 
                        param_3: str) -> None:
        """
        For parameter trio (param1,param2,param3), plot each point in the archive colored by redshift.

        Args:
            param_1: First parameter name to plot
            param_2: Second parameter name to plot 
            param_3: Third parameter name to plot

        Returns:
            None
        """
        from mpl_toolkits import mplot3d

        emu_data = self.get_training_data(emu_params=[param_1, param_2, param_3])

        emu_vals = np.array([[d[param_1], d[param_2], d[param_3], d["z"]] for d in emu_data])
        emu_1, emu_2, emu_3, emu_z = emu_vals.T

        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.scatter3D(emu_1, emu_2, emu_3, c=emu_z, cmap="brg", s=8)
        ax.set_xlabel(param_1)
        ax.set_ylabel(param_2)
        ax.set_zlabel(param_3)
        plt.show()

    def print_entry(self, 
                    entry: int) -> None:
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
            raise ValueError(f"{entry} entry does not exist in archive")

        list_print = ["z"] + self.emu_params
        info = [f"entry = {entry}"]
        info.extend(f"{key} = {self.data[entry][key]:.4f}" for key in list_print)
        
        print(", ".join(info))
