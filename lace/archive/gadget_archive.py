import numpy as np
import copy
import sys
import os
import json
from lace.setup_simulations import read_genic, read_gadget
from lace.cosmo import camb_cosmo
from lace.cosmo import fit_linP

from lace.archive.base_archive import BaseArchive
from lace.archive.exceptions import ExceptionList


class GadgetArchive(BaseArchive):
    """
    Bookkeeping of Lya flux P1D & P3D measurements from a suite of Gadget simulations.

    Methods:
        __init__(self, postproc, linP_dir)
        _set_info_postproc(self, postproc)
        _sim2file_name(self, sim_label)
        _get_nz_linP(self, sim_label)
        _get_file_names(self, sim_label, ind_phase, ind_z, ind_axis)
        _get_sim(self, sim_label, ind_z, ind_axis)
        _load_data(self, pick_sim, drop_sim, z_max, nsamples=None)
    """

    def __init__(self, postproc="Cabayol23", kp_Mpc=None):
        """
        Initialize the archivePND object.

        Args:
            postproc (str): Name of the simulation suite. Default is "Cabayol23".
                Raises a ValueError if the simulation suite is not available.
            kp_Mpc (None or float): Optional. Pivot point used in linear power parameters.
                If specified, the parameters will be recomputed in the archive. Default is None.

        Returns:
            None

        """

        ## check input

        if isinstance(postproc, str) == False:
            raise TypeError("postproc must be a string")
        postproc_all = ["Pedersen21", "Cabayol23", "768_768"]
        if postproc not in postproc_all:
            msg = "Invalid postproc value. Available options:"
            raise ExceptionList(msg, postproc_all)

        if isinstance(kp_Mpc, (int, float, type(None))) == False:
            raise TypeError("kp_Mpc must be a number or None")
        self.kp_Mpc = kp_Mpc

        ## done check input

        ## sets list simulations available for this suite
        # list of especial simulations
        self.list_sim_test = [
            "mpg_central",
            "mpg_seed",
            "mpg_growth",
            "mpg_neutrinos",
            "mpg_curved",
            "mpg_running",
            "mpg_reio",
        ]
        # list of hypercube simulations
        self.list_sim_cube = []
        for ii in range(30):
            self.list_sim_cube.append("mpg_" + str(ii))
        # list all simulations
        self.list_sim = self.list_sim_cube + self.list_sim_test
        ## done set simulation list

        # get relevant flags for post-processing
        self._set_info_postproc(postproc)

        # load power spectrum measurements
        self._load_data()

        # extract indexes from data
        self._set_labels()

    def _set_info_postproc(self, postproc):
        """
        Set information about the post-processing suite and set corresponding attributes.

        Args:
            postproc (str): Name of the simulation suite.

        Returns:
            None

        Raises:
            AssertionError: If the environment variable "LACE_REPO" is not set.
        """

        self.postproc = postproc

        if postproc == "Pedersen21":
            # directory of the post-processing within LaCE
            self.basedir = "/lace/emulator/sim_suites/Australia20/"
            # number of simulation phases (fix-and-paired IC)
            self.n_phases = 2
            # number of simulation axes in the post-processing
            self.n_axes = 1
            # internal labels specifying the name of the post-processing files
            self.p1d_label = "p1d"
            self.sk_label = "Ns500_wM0.05"
            # Reading parameters describing the simulations from here. This is only important for
            # Cabayol23, as it reads these parameters from the Pedersen21 post-processing.
            # It was implemented this way because the code used to compute these values,
            # fake_spectra, changed between Pedersen21 and Cabayol23
            self.basedir_params = "/lace/emulator/sim_suites/Australia20/"
            self.p1d_label_params = self.p1d_label
            self.sk_label_params = "Ns500_wM0.05"
            # if files include P3D measurements
            self.also_P3D = False
            self.tag_param = "parameter_redundant.json"
            # available scalings for each simulation
            self.scalings_avail = [0.9, 1, 1.1]
            # training options
            self.training_average = "both"
            self.training_val_scaling = 1
            self.training_z_max = 10
            # testing options
            self.testing_val_scaling = 1
            self.testing_z_max = 10
        elif postproc == "Cabayol23":
            self.basedir = "/lace/emulator/sim_suites/post_768/"
            self.n_phases = 2
            self.n_axes = 3
            self.p1d_label = "p1d_stau"
            self.sk_label = "Ns768_wM0.05"
            self.basedir_params = "/lace/emulator/sim_suites/Australia20/"
            self.p1d_label_params = "p1d"
            self.sk_label_params = "Ns500_wM0.05"
            self.also_P3D = True
            self.tag_param = "parameter_redundant.json"
            self.scalings_avail = [0.9, 0.95, 1, 1.05, 1.1]
            self.training_average = "axes_phases_both"
            self.training_val_scaling = "all"
            self.training_z_max = 10
            self.testing_val_scaling = 1
            self.testing_z_max = 10
        elif postproc == "768_768":
            self.basedir = "/lace/emulator/sim_suites/post_768/"
            self.n_phases = 2
            self.n_axes = 3
            self.p1d_label = "p1d_stau"
            self.sk_label = "Ns768_wM0.05"
            self.basedir_params = "/lace/emulator/sim_suites/post_768/"
            self.p1d_label_params = self.p1d_label
            self.sk_label_params = "Ns768_wM0.05"
            self.also_P3D = True
            self.tag_param = "parameter.json"
            self.scalings_avail = [0.9, 0.95, 1, 1.05, 1.1]
            self.training_average = "axes_phases_both"
            self.training_val_scaling = "all"
            self.training_z_max = 10
            self.testing_val_scaling = 1
            self.testing_z_max = 10

        ## get path of the repo

        assert "LACE_REPO" in os.environ, "export LACE_REPO"
        repo = os.environ["LACE_REPO"] + "/"

        self.fulldir = repo + self.basedir
        self.fulldir_param = repo + self.basedir_params

        self.key_conv = {
            "mF": "mF",
            "sim_T0": "T0",
            "sim_gamma": "gamma",
            "sim_sigT_Mpc": "sigT_Mpc",
            "kF_Mpc": "kF_Mpc",
            "k_Mpc": "k_Mpc",
            "p1d_Mpc": "p1d_Mpc",
            "scale_tau": "val_scaling",
        }

        # parameters that must be in each element of the
        # training and testing set
        self.emu_params = [
            "Delta2_p",
            "n_p",
            "mF",
            "sigT_Mpc",
            "gamma",
            "kF_Mpc",
        ]

    def _sim2file_name(self, sim_label):
        """
        Convert simulation labels to file names.

        Args:
            sim_label (int or str): Selected simulation.

        Returns:
            tuple: A tuple containing the simulation file names and parameter file tag.

        """
        if self.postproc == "Pedersen21":
            dict_conv = {
                "mpg_central": "central",
                "mpg_seed": "diffSeed_sim",
                "mpg_growth": "h_sim",
                "mpg_neutrinos": "nu_sim",
                "mpg_curved": "curved_003",
                "mpg_running": "running_sim",
                "mpg_reio": "P18_sim",
            }
            dict_conv_params = dict_conv
        elif self.postproc == "Cabayol23":
            dict_conv = {
                "mpg_central": "sim_pair_30",
                "mpg_seed": "diffSeed",
                "mpg_growth": "sim_pair_h",
                "mpg_neutrinos": "nu_sim",
                "mpg_curved": "curved_003",
                "mpg_running": "running",
                "mpg_reio": "P18",
            }
            dict_conv_params = {
                "mpg_central": "central",
                "mpg_seed": "diffSeed_sim",
                "mpg_growth": "h_sim",
                "mpg_neutrinos": "nu_sim",
                "mpg_curved": "curved_003",
                "mpg_running": "running_sim",
                "mpg_reio": "P18_sim",
            }
        elif self.postproc == "768_768":
            dict_conv = {
                "mpg_central": "sim_pair_30",
                "mpg_seed": "diffSeed",
                "mpg_growth": "sim_pair_h",
                "mpg_neutrinos": "nu_sim",
                "mpg_curved": "curved_003",
                "mpg_running": "running",
                "mpg_reio": "P18",
            }
            dict_conv_params = dict_conv

        for ii in range(30):
            dict_conv["mpg_" + str(ii)] = "sim_pair_" + str(ii)
            dict_conv_params["mpg_" + str(ii)] = "sim_pair_" + str(ii)

        if sim_label in self.list_sim_test:
            tag_param = self.tag_param
        else:
            tag_param = "parameter.json"

        return dict_conv[sim_label], dict_conv_params[sim_label], tag_param

    def _get_nz_linP(self, sim_label):
        """
        Get the redshifts and linear power spectrum for the specified simulation.

        Args:
            sim_label (int or str): Selected simulation.

        Returns:
            tuple: A tuple containing the redshifts and linear power parameters.
                - zs (list): List of redshift values.
                - linP_zs (list): List of linear power spectrum at each redshift.

        """

        _, sim_name_param, tag_param = self._sim2file_name(sim_label)
        pair_dir = self.fulldir_param + "/" + sim_name_param

        # read zs values and compute/read linP_zs
        if (self.update_kp == False) & (sim_label in self.list_sim_cube):
            pair_json = pair_dir + "/" + tag_param

            with open(pair_json) as json_file:
                pair_data = json.load(json_file)
            zs = pair_data["zs"]
            linP_zs = pair_data["linP_zs"]
        else:
            # read gadget file
            gadget_fname = pair_dir + "/sim_plus/paramfile.gadget"
            sim_config = read_gadget.read_gadget_paramfile(gadget_fname)
            zs = read_gadget.snapshot_redshifts(sim_config)

            # compute linP_zs parameters
            # setup cosmology from GenIC file
            genic_fname = pair_dir + "/sim_plus/paramfile.genic"
            sim_cosmo_dict = read_genic.camb_from_genic(genic_fname)

            # setup CAMB object
            sim_cosmo = camb_cosmo.get_cosmology_from_dictionary(sim_cosmo_dict)

            # compute linear power parameters at each z (in Mpc units)
            linP_zs = fit_linP.get_linP_Mpc_zs(sim_cosmo, zs, self.kp_Mpc)
            linP_zs = list(linP_zs)

        return zs, linP_zs

    def _get_file_names(self, sim_label, ind_phase, ind_z, ind_axis):
        """
        Get the file names for the specified simulation parameters and snapshot.

        Args:
            sim_label (int or str): Selected simulation.
            ind_phase (int): Index of the simulation phase.
            ind_z (int): Index of the redshift.
            ind_axis (int): Index of the simulation axis.

        Returns:
            tuple: A tuple containing the file names for data and parameter JSON files.
                - data_json (str): File name for the data JSON file.
                - param_json (str): File name for the parameter JSON file.

        Notes:
            - The `sim_label` argument refers to the selected simulation.
            - The `ind_phase` argument refers to the index of the simulation phase.
            - The `ind_z` argument refers to the index of the redshift.
            - The `ind_axis` argument refers to the index of the simulation axis.
        """

        sim_name, sim_name_param, tag_param = self._sim2file_name(sim_label)

        if ind_phase == 0:
            tag_phase = "sim_plus"
        else:
            tag_phase = "sim_minus"

        if self.postproc == "Pedersen21":
            _sk_label_data = self.sk_label
            _sk_label_params = self.sk_label_params
            # different number of mF scalings for each post-processing
            n_it_files = 1
        else:
            _sk_label_data = self.sk_label + "_axis" + str(ind_axis + 1)
            if self.postproc == "Cabayol23":
                _sk_label_params = self.sk_label_params
            else:
                _sk_label_params = (
                    self.sk_label_params + "_axis" + str(ind_axis + 1)
                )
            # for these simulations with have one file with all scalings, for others 2
            sim_one_scaling = ["diffSeed", "P18", "running", "curved_003"]
            if sim_name in sim_one_scaling:
                n_it_files = 1
            else:
                n_it_files = 2

        # path to measurements
        data_json = []

        for it in range(n_it_files):
            if it == 0:
                p1d_label = self.p1d_label
            else:
                p1d_label = "p1d_setau"
            data_json.append(
                self.fulldir
                + "/"
                + sim_name
                + "/"
                + tag_phase
                + "/"
                + p1d_label
                + "_"
                + str(ind_z)
                + "_"
                + _sk_label_data
                + ".json"
            )

        # path to parameters
        param_json = (
            self.fulldir_param
            + "/"
            + sim_name_param
            + "/"
            + tag_phase
            + "/"
            + self.p1d_label_params
            + "_"
            + str(ind_z)
            + "_"
            + _sk_label_params
            + ".json"
        )

        return data_json, param_json

    def _get_sim(self, sim_label, ind_z, ind_axis):
        """
        Get the data and parameter information for the specified simulation parameters and snapshot.

        Args:
            self (object): The instance of the class containing this method.
            sim_label (str): Label of the simulation to retrieve.
            ind_z (int): Index of the redshift.
            ind_axis (int): Index of the simulation axis.

        Returns:
            tuple: A tuple containing the data and parameter information.
                - phase_data (list): List of dictionaries containing the data for each phase.
                - phase_params (list): List of dictionaries containing the parameter information for each phase.
                - arr_phase (list): List of phase indices corresponding to each data entry.

        Note:
            This function retrieves the data and parameter information for the specified simulation parameters and snapshot.
            The data is obtained by reading JSON files stored at specific paths.

        """
        phase_data = []
        phase_params = []
        arr_phase = []

        # open sim_plus and sim_minus (P1D + P3D & params)
        for ind_phase in range(self.n_phases):
            # get path to
            _ = self._get_file_names(sim_label, ind_phase, ind_z, ind_axis)
            path_data, path_param = _

            # read power spectrum
            for it in range(len(path_data)):
                with open(path_data[it]) as json_file:
                    phase_data.append(json.load(json_file))
                    arr_phase.append(ind_phase)

            # read params
            with open(path_param) as json_file:
                phase_params.append(json.load(json_file))

        return phase_data, phase_params, arr_phase

    def _load_data(self):
        """
        Setup the archive by gathering information from all measured power spectra in the simulations.

        Returns:
            None

        Notes:
        - This function reads and processes data from all measured power spectra in the simulations.
        - The function iterates over simulations, snapshots, and axes to retrieve the required data.
        - The retrieved data is stored in the `self.data` attribute as a list of dictionaries,
            with each dictionary representing a specific measurement and simulation configuration.

        """

        # All samples have an entry in this list. This entry is dictionary includes
        # P1D and P3D measurements and info about simulations
        self.data = []

        keys_in = list(self.key_conv.keys())

        ## read file containing information about simulation suite
        cube_json = self.fulldir + "/latin_hypercube.json"
        try:
            with open(cube_json) as json_file:
                self.cube_data = json.load(json_file)
        except FileNotFoundError:
            print(f"Error: Cube JSON file '{cube_json}' not found.")
        else:
            self.nsamples = self.cube_data["nsamples"]

            # read pivot point from simulation suite if not specified
            if self.kp_Mpc is None:
                n_star = self.cube_data["param_space"]["n_star"]
                self.kp_Mpc = n_star["kp_Mpc"]
                self.update_kp = False
            elif (
                self.kp_Mpc == self.cube_data["param_space"]["n_star"]["kp_Mpc"]
            ):
                ## If selected k_p is same as in the archive, do not recompute
                self.update_kp = False
            else:
                # will trigger slow code, could check that kp has indeed changed
                self.update_kp = True
        ## done

        ## read info from all sims, all snapshots, all rescalings
        # iterate over simulations
        for sim_label in self.list_sim:
            zs, linP_zs = self._get_nz_linP(sim_label)
            ### should we also save cosmology here?

            # iterate over snapshots
            for ind_z in range(len(zs)):
                # set linear power parameters describing snapshot
                linP_params = linP_zs[ind_z]
                snap_data = {}
                snap_data["Delta2_p"] = linP_params["Delta2_p"]
                snap_data["n_p"] = linP_params["n_p"]
                snap_data["alpha_p"] = linP_params["alpha_p"]
                snap_data["f_p"] = linP_params["f_p"]
                snap_data["z"] = zs[ind_z]

                # iterate over axes
                for ind_axis in range(self.n_axes):
                    # read data from simulation (different phases and rescalings)
                    _ = self._get_sim(sim_label, ind_z, ind_axis)
                    phase_data, phase_params, arr_phase = _

                    # store measurements
                    len_phase_data = len(phase_data)
                    for ind_phase in range(len_phase_data):
                        # iterate over scalings
                        n_scalings = len(phase_data[ind_phase]["p1d_data"])
                        if (len_phase_data == 2) | (ind_phase % 2 == 0):
                            floor_scaling = 0
                        elif (len_phase_data > 2) & (ind_phase % 2 != 0):
                            floor_scaling = len(phase_data[0]["p1d_data"])

                        for pp in range(n_scalings):
                            temp_data = phase_data[ind_phase]["p1d_data"][pp]
                            _ind_phase = arr_phase[ind_phase]
                            temp_param = phase_params[_ind_phase]["p1d_data"][0]

                            sim_data = copy.deepcopy(snap_data)
                            # identify simulation
                            sim_data["sim_label"] = sim_label
                            sim_data["ind_snap"] = ind_z
                            sim_data["ind_phase"] = _ind_phase
                            sim_data["ind_axis"] = ind_axis
                            sim_data["ind_rescaling"] = floor_scaling + pp

                            # iterate over properties
                            all_keys = list(self.key_conv.keys())
                            for key_in in all_keys:
                                key_out = self.key_conv[key_in]
                                if (key_in == "scale_tau") | (key_in == "mF"):
                                    sim_data[key_out] = temp_data[key_in]
                                elif (key_in == "p1d_Mpc") | (
                                    key_in == "k_Mpc"
                                ):
                                    sim_data[key_out] = np.array(
                                        temp_data[key_in]
                                    )
                                else:
                                    sim_data[key_out] = temp_param[key_in]

                                if self.also_P3D:
                                    sim_data["p3d_Mpc"] = np.array(
                                        temp_data["p3d_data"]["p3d_Mpc"]
                                    )
                                    sim_data["k3d_Mpc"] = np.array(
                                        temp_data["p3d_data"]["k_Mpc"]
                                    )
                                    sim_data["mu3d"] = np.array(
                                        temp_data["p3d_data"]["mu"]
                                    )

                            self.data.append(sim_data)
