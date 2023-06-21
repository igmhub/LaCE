import numpy as np
import copy
import sys
import os
import json
from itertools import product
from lace.setup_simulations import read_genic, read_gadget
from lace.cosmo import camb_cosmo
from lace.cosmo import fit_linP
from lace.emulator.utils import split_string


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


class archivePND(object):
    """
    Book-keeping of flux P1D & P3D measurements from a suite of simulations.

    Methods:
        __init__(self, sim_suite, linP_dir)
        _get_info_sim_suite(self, sim_suite)
        _get_sim_info(self, target_sim)
        _get_nz_linP(self, target_sim, pair_dir, tag_param, update_kp)
        _get_file_names(self, ind_axis, tag_phase, snap, tag_sample, tag_sample_params)
        _get_data(self, ind_axis, snap, tag_sample, tag_sample_params)
        _load_data(self, pick_sim, drop_sim, z_max, nsamples=None)
        _store_param_arrays(self)
        average_over_samples(self, flag="all")
        input_emulator(self, flag="all")
        print_entry(self, entry, fiducial_keys=True)
        plot_samples(self, param_1, param_2, tau_scalings=True, temp_scalings=True)
        plot_3D_samples(self, param_1, param_2, param_3, tau_scalings=True, temp_scalings=True)
    """

    def __init__(
        self,
        sim_suite="Cabayol23",
        pick_sim=None,
        drop_sim=None,
        nsamples=None,
        z_max=5.0,
        kp_Mpc=None,
        verbose=False,
    ):
        """
        Initialize the archivePND object.

        Args:
            sim_suite (str): Name of the simulation suite. Default is "Cabayol23".
                Raises a ValueError if the simulation suite is not available.
            pick_sim (None, int, or str): Optional. Simulation to pick from the available options.
                Raises a ValueError if the simulation is not available. Default is None.
            drop_sim (None, int, or str): Optional. Simulation to drop from the available options.
                Raises a ValueError if the simulation is not available. Default is None.
            nsamples (None or int): Optional. Maximum number of samples to load from each simulation.
                Default is None, which loads all samples.
            z_max (float): Optional. Maximum redshift value to consider when loading data. Default is 5.0.
            kp_Mpc (None or float): Optional. Pivot point used in linear power parameters.
                If specified, the parameters will be recomputed in the archive. Default is None.
            verbose (bool): Optional. Verbosity flag. If True, print additional information during loading.
                Default is False.

        Returns:
            None

        """

        ## check input

        # check if sim_suite available (To be modified for new suite)
        sim_suite_all = ["Pedersen21", "Cabayol23", "768_768"]
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

        # get list of simulations available for this suite
        _ = get_sim_option_list(sim_suite)
        self.sim_option_list, self.sim_especial_list, self.sim_option_dict = _

        # check if the value of pick_sim within simulations available
        if pick_sim is not None:
            try:
                if pick_sim in self.sim_option_list:
                    pass
                else:
                    print(
                        "Invalid pick_sim value. Available options: ",
                        self.sim_option_list,
                    )
                    raise
            except:
                print("An error occurred while checking the pick_sim value.")
                raise

        # check if the value of drop_sim within simulations available
        if drop_sim is not None:
            try:
                if drop_sim in self.sim_option_list:
                    pass
                else:
                    print(
                        "Invalid drop_sim value. Available options: ",
                        self.sim_option_list,
                    )
                    raise
            except:
                print("An error occurred while checking the drop_sim value.")
                raise

        ## get info from simulation suite
        self._get_info_sim_suite(sim_suite)

        # pivot point used in linP parameters
        self.kp_Mpc = kp_Mpc
        self.verbose = verbose

        ## load power spectrum measurements from suite
        self._load_data(pick_sim, drop_sim, z_max, nsamples)

        return

    def _get_keys_input_dict(self):
        """
        Get dictionary with the name of the target properties from the simulation suite.

        Returns:
            list: List of keys for the input dictionary.

        Note:
            To be modified for new suite.

        """
        if (
            (self.sim_suite == "Pedersen21")
            | (self.sim_suite == "Cabayol23")
            | (self.sim_suite == "768_768")
        ):
            # keys input file
            keys_copy_in = [
                "mF",
                "sim_T0",
                "sim_gamma",
                "sim_sigT_Mpc",
                "kF_Mpc",
                "k_Mpc",
                "p1d_Mpc",
                "scale_tau",
                "sim_scale_T0",
                "sim_scale_gamma",
            ]
            if self.also_P3D:
                keys_copy_in.append("p3d_data")
        return keys_copy_in

    def _get_keys_output_dict(self):
        """
        Get dictionary with the name of the properties to return from bookkeeping.

        Returns:
            list: List of keys for the output dictionary.

        Note:
            To be modified for new suite.

        """
        if (
            (self.sim_suite == "Pedersen21")
            | (self.sim_suite == "Cabayol23")
            | (self.sim_suite == "768_768")
        ):
            # keys input file
            keys_copy_out = [
                "mF",
                "T0",
                "gamma",
                "sigT_Mpc",
                "kF_Mpc",
                "k_Mpc",
                "p1d_Mpc",
                "scale_tau",
                "scale_T0",
                "scale_gamma",
            ]
            if self.also_P3D:
                keys_copy_out.append("k3d_Mpc")
                keys_copy_out.append("mu3d")
                keys_copy_out.append("p3d_Mpc")
        return keys_copy_out

    def _get_info_sim_suite(self, sim_suite):
        """
        Get information about the simulation suite and set corresponding attributes.

        Args:
            sim_suite (str): Name of the simulation suite.

        Returns:
            None

        Raises:
            AssertionError: If the environment variable "LACE_REPO" is not set.

        Note:
            To be modified for new suite.

        """

        self.sim_suite = sim_suite

        if sim_suite == "Pedersen21":
            # directory of the post-processing within LaCE
            self.basedir = "/lace/emulator/sim_suites/Australia20/"
            # number of simulation phases (fix-and-paired IC)
            self.n_phases = 2
            # number of simulation axes in the post-processing
            self.n_axes = 1
            # internal labels specifying the name of the post-processing files
            self.p1d_label = "p1d"
            self.sk_label = "Ns500_wM0.05"
            # Only important for LaCE post-processing. Reading parameters
            # describing the simulations from here. This is only important for
            # Cabayol23, as it reads these parameters from the Pedersen21 post-processing.
            # It was implemented this way because the code used to compute these values,
            # fake_spectra, changed between Pedersen21 and Cabayol23
            self.basedir_params = "/lace/emulator/sim_suites/Australia20/"
            self.p1d_label_params = self.p1d_label
            self.sk_label_params = "Ns500_wM0.05"
            # if files include P3D measurements
            self.also_P3D = False
        elif sim_suite == "Cabayol23":
            self.basedir = "/lace/emulator/sim_suites/post_768/"
            self.n_phases = 2
            self.n_axes = 3
            self.p1d_label = "p1d_stau"
            self.sk_label = "Ns768_wM0.05"
            self.basedir_params = "/lace/emulator/sim_suites/Australia20/"
            self.p1d_label_params = "p1d"
            self.sk_label_params = "Ns500_wM0.05"
            self.also_P3D = True
        elif sim_suite == "768_768":
            self.basedir = "/lace/emulator/sim_suites/post_768/"
            self.n_phases = 2
            self.n_axes = 3
            self.p1d_label = "p1d_stau"
            self.sk_label = "Ns768_wM0.05"
            self.basedir_params = "/lace/emulator/sim_suites/post_768/"
            self.p1d_label_params = self.p1d_label
            self.sk_label_params = "Ns768_wM0.05"
            self.also_P3D = True

        ## get path of the repo

        assert "LACE_REPO" in os.environ, "export LACE_REPO"
        repo = os.environ["LACE_REPO"] + "/"

        self.fulldir = repo + self.basedir
        self.fulldir_params = repo + self.basedir_params

    def _get_sim_info(self, target_sim):
        """
        Get simulation information based on the specified simulation suite and selected simulation option.

        Args:
            target_sim (int or str): Selected simulation.

        Returns:
            tuple: A tuple containing the simulation tags for the selected simulation option.
                - tag_sample (str): Simulation tag for the selected simulation option.
                - tag_sample_params (str): Simulation tag for the selected simulation option in parameter files.
                - tag_param (str): Tag for the parameter file.

        Note:
            To be modified for new suite.
        """
        if (
            (self.sim_suite == "Pedersen21")
            | (self.sim_suite == "Cabayol23")
            | (self.sim_suite == "768_768")
        ):
            if self.sim_suite == "Pedersen21":
                dict_conv = {
                    "central": "central",
                    "seed": "diffSeed_sim",
                    "growth": "h_sim",
                    "neutrinos": "nu_sim",
                    "curved": "curved_003",
                    "running": "running_sim",
                    "reionization": "P18_sim",
                }
                dict_conv_params = dict_conv
                tag_param = "parameter_redundant.json"
            elif self.sim_suite == "Cabayol23":
                dict_conv = {
                    "central": "sim_pair_30",
                    30: "sim_pair_30",
                    "seed": "diffSeed",
                    "growth": "sim_pair_h",
                    "neutrinos": "nu_sim",
                    "curved": "curved_003",
                    "running": "running",
                    "reionization": "P18",
                }
                dict_conv_params = {
                    "central": "central",
                    30: "central",
                    "seed": "diffSeed_sim",
                    "growth": "h_sim",
                    "neutrinos": "nu_sim",
                    "curved": "curved_003",
                    "running": "running_sim",
                    "reionization": "P18_sim",
                }
                tag_param = "parameter_redundant.json"
            else:
                dict_conv = {
                    "central": "sim_pair_30",
                    30: "sim_pair_30",
                    "seed": "diffSeed",
                    "growth": "sim_pair_h",
                    "neutrinos": "nu_sim",
                    "curved": "curved_003",
                    "running": "running",
                    "reionization": "P18",
                }
                dict_conv_params = dict_conv
                tag_param = "parameter.json"

            if (target_sim in self.sim_especial_list) or (target_sim == 30):
                tag_sample = dict_conv[target_sim]
                tag_sample_params = dict_conv_params[target_sim]
            else:
                tag_sample = "sim_pair_" + str(target_sim)
                tag_sample_params = tag_sample
                tag_param = "parameter.json"

        return tag_sample, tag_sample_params, tag_param

    def _get_nz_linP(self, target_sim, pair_dir, tag_param, update_kp):
        """
        Get the redshifts and linear power spectrum for the specified simulation.

        Args:
            target_sim (int or str): Selected simulation.
            pair_dir (str): Directory path of the simulation pair.
            tag_param (str): Tag for the parameter file.
            update_kp (bool): Flag indicating whether to update the linear power parameters.

        Returns:
            tuple: A tuple containing the redshifts and linear power parameters.
                - zs (list): List of redshift values.
                - linP_zs (list): List of linear power spectrum at each redshift.

        """

        # read zs values and compute/read linP_zs
        if (target_sim in self.sim_especial_list) | (target_sim == 30):
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
        else:
            pair_json = pair_dir + "/" + tag_param

            with open(pair_json) as json_file:
                pair_data = json.load(json_file)
            zs = pair_data["zs"]
            linP_zs = pair_data["linP_zs"]

        # overwrite linP parameters stored in parameter.json
        if update_kp:
            print("overwritting linP_zs in parameter.json")
            # setup cosmology from GenIC file
            genic_fname = pair_dir + "/sim_plus/paramfile.genic"
            print("read cosmology from GenIC", genic_fname)
            sim_cosmo_dict = read_genic.camb_from_genic(genic_fname)
            # setup CAMB object
            sim_cosmo = camb_cosmo.get_cosmology_from_dictionary(sim_cosmo_dict)
            # compute linear power parameters at each z (in Mpc units)
            linP_zs = fit_linP.get_linP_Mpc_zs(sim_cosmo, zs, self.kp_Mpc)
            print("update linP_zs", linP_zs)
            linP_zs = list(linP_zs)
        else:
            if self.verbose:
                print("Use linP_zs from parameter.json")

        return zs, linP_zs

    def _get_file_names(self, ind_axis, tag_phase, snap, tag_sample, tag_sample_params):
        """
        Get the file names for the specified simulation parameters and snapshot.

        Args:
            ind_axis (int): Index of the simulation axis.
            tag_phase (str): Tag for the simulation phase.
            snap (int): Snapshot number.
            tag_sample (str): Simulation tag for the selected simulation option.
            tag_sample_params (str): Simulation tag for the selected simulation option in parameter files.

        Returns:
            tuple: A tuple containing the file names for data and parameter JSON files.
                - data_json (list): List of file names for data JSON files.
                - param_json (str): File name for parameter JSON file.

        Note:
            To be modified for new suite.

        """

        if (
            (self.sim_suite == "Pedersen21")
            | (self.sim_suite == "Cabayol23")
            | (self.sim_suite == "768_768")
        ):
            if self.sim_suite == "Pedersen21":
                _sk_label_data = self.sk_label
                _sk_label_params = self.sk_label_params
                # different number of mF scalings for each post-processing
                n_it_files = 1
            else:
                _sk_label_data = self.sk_label + "_axis" + str(ind_axis + 1)
                if self.sim_suite == "Cabayol23":
                    _sk_label_params = self.sk_label_params
                else:
                    _sk_label_params = (
                        self.sk_label_params + "_axis" + str(ind_axis + 1)
                    )
                # for these simulations with have one file with all scalings, for others 2
                sim_one_scaling = ["diffSeed", "P18", "running", "curved_003"]
                if tag_sample in sim_one_scaling:
                    n_it_files = 1
                else:
                    n_it_files = 2

            data_json = []

            for it in range(n_it_files):
                if it == 0:
                    p1d_label = self.p1d_label
                else:
                    p1d_label = "p1d_setau"
                data_json.append(
                    self.fulldir
                    + "/"
                    + tag_sample
                    + tag_phase
                    + p1d_label
                    + "_"
                    + str(snap)
                    + "_"
                    + _sk_label_data
                    + ".json"
                )

            param_json = (
                self.fulldir_params
                + "/"
                + tag_sample_params
                + tag_phase
                + self.p1d_label_params
                + "_"
                + str(snap)
                + "_"
                + _sk_label_params
                + ".json"
            )

        return data_json, param_json

    def _get_data(self, ind_axis, snap, tag_sample, tag_sample_params):
        """
        Get the data and parameter information for the specified simulation parameters and snapshot.

        Args:
            ind_axis (int): Index of the simulation axis.
            snap (int): Snapshot number.
            tag_sample (str): Simulation tag for the selected simulation option.
            tag_sample_params (str): Simulation tag for the selected simulation option in parameter files.

        Returns:
            tuple: A tuple containing the data and parameter information.
                - phase_data (list): List of dictionaries containing the data for each phase.
                - phase_params (list): List of dictionaries containing the parameter information for each phase.
                - arr_phase (list): List of phase indices corresponding to each data entry.


        Note:
            To be modified for new suite.

        """
        phase_data = []
        phase_params = []
        arr_phase = []

        if (
            (self.sim_suite == "Pedersen21")
            | (self.sim_suite == "Cabayol23")
            | (self.sim_suite == "768_768")
        ):
            # open sim_plus and sim_minus (P1D + P3D & params)
            for _ind_phase in range(self.n_phases):
                if _ind_phase == 0:
                    tag_phase = "/sim_plus/"
                else:
                    tag_phase = "/sim_minus/"

                data_json, param_json = self._get_file_names(
                    ind_axis, tag_phase, snap, tag_sample, tag_sample_params
                )

                # open file with 1D and 3D power measured
                for it in range(len(data_json)):
                    with open(data_json[it]) as json_file:
                        phase_data.append(json.load(json_file))
                        arr_phase.append(_ind_phase)
                with open(param_json) as json_file:
                    phase_params.append(json.load(json_file))

        return phase_data, phase_params, arr_phase

    def _load_data(self, pick_sim, drop_sim, z_max, nsamples=None):
        """
        Setup the archive by gathering information from all measured power spectra in the simulations.

        Args:
            pick_sim (int or str): The simulation index or special simulation identifier to start the loop from. Default is None.
            drop_sim (int or str): The simulation index or special simulation identifier to skip during the loop. Default is None.
            z_max (float): The maximum redshift value to consider for the power spectra.
            nsamples (int, optional): The number of samples in the simulation suite. If not specified, it is read from the simulation suite. Default is None.

        Returns:
            None

        """

        # All samples have an entry in this list. This entry is dictionary includes
        # P1D and P3D measurements and info about simulations
        self.data = []

        keys_copy_in = self._get_keys_input_dict()
        keys_copy_out = self._get_keys_output_dict()

        # read file containing information about simulation suite
        cube_json = self.fulldir + "/latin_hypercube.json"
        try:
            with open(cube_json) as json_file:
                self.cube_data = json.load(json_file)
        except FileNotFoundError:
            print(f"Error: Cube JSON file '{cube_json}' not found.")
        else:
            # read nsamples from simulation suite if not specified
            if nsamples is None:
                self.nsamples = self.cube_data["nsamples"]
                if self.verbose:
                    print("simulation suite has %d samples" % self.nsamples)
            else:
                self.nsamples = nsamples

            # read pivot point from simulation suite if not specified
            if self.kp_Mpc is None:
                n_star = self.cube_data["param_space"]["n_star"]
                self.kp_Mpc = n_star["kp_Mpc"]
                update_kp = False
            elif self.kp_Mpc == self.cube_data["param_space"]["n_star"]["kp_Mpc"]:
                ## If selected k_p is same as in the archive, do not recompute
                update_kp = False
            else:
                # will trigger slow code, could check that kp has indeed changed
                update_kp = True

        # if pick_sim selected, start the loop at
        if pick_sim is not None:
            if np.issubdtype(type(pick_sim), np.integer):
                start = pick_sim  # pick_sim (if integer)
                self.nsamples = pick_sim + 1
            else:
                start = 0  # pick_sim (if str because it is an special simulation)
                self.nsamples = 1
        else:
            start = 0

        ## read data
        # read info from all sims, all snapshots, all rescalings
        # iterate over simulations
        for ind_sim in range(start, self.nsamples):
            if ind_sim is drop_sim:
                continue
            # we want to drop central simulation when reading hypercube
            elif ind_sim == 30:
                continue

            if pick_sim is not None:
                _ = self._get_sim_info(pick_sim)
                out_ind_sim = self.sim_option_dict[pick_sim]
            else:
                _ = self._get_sim_info(ind_sim)
                out_ind_sim = ind_sim
            tag_sample, tag_sample_params, tag_param = _
            pair_dir = self.fulldir_params + "/" + tag_sample_params
            zs, linP_zs = self._get_nz_linP(pick_sim, pair_dir, tag_param, update_kp)

            # iterate over snapshots
            for snap in range(len(zs)):
                if zs[snap] > z_max:
                    continue
                # get linear power parameters describing snapshot
                linP_params = linP_zs[snap]
                snap_p1d_data = {}
                snap_p1d_data["Delta2_p"] = linP_params["Delta2_p"]
                snap_p1d_data["n_p"] = linP_params["n_p"]
                snap_p1d_data["alpha_p"] = linP_params["alpha_p"]
                snap_p1d_data["f_p"] = linP_params["f_p"]
                snap_p1d_data["z"] = zs[snap]

                # iterate over axes
                for ind_axis in range(self.n_axes):
                    # extract power spectrum measurements
                    # we iterate now to extract data from

                    # read data from simulations
                    _ = self._get_data(ind_axis, snap, tag_sample, tag_sample_params)
                    phase_data, phase_params, arr_phase = _

                    # iterate over phases
                    len_phase_data = len(phase_data)
                    for ind_phase in range(len_phase_data):
                        # iterate over scalings
                        n_scalings = len(phase_data[ind_phase]["p1d_data"])
                        # TO BE UPDATED IF MORE THAN TWO FILES WITH TAU-SCALINGS FOR
                        # EACH SIMULATION
                        if (len_phase_data == 2) | (ind_phase % 2 == 0):
                            floor_scaling = 0
                        elif (len_phase_data > 2) & (ind_phase % 2 != 0):
                            floor_scaling = len(phase_data[0]["p1d_data"])

                        for pp in range(n_scalings):
                            temp_data = phase_data[ind_phase]["p1d_data"][pp]
                            _ind_phase = arr_phase[ind_phase]
                            temp_param = phase_params[_ind_phase]["p1d_data"][0]

                            # deep copy of dictionary (thread safe, why not)
                            p1d_data = json.loads(json.dumps(snap_p1d_data))
                            # identify simulation
                            p1d_data["ind_sim"] = out_ind_sim
                            p1d_data["ind_z"] = snap
                            p1d_data["ind_phase"] = _ind_phase
                            p1d_data["ind_axis"] = ind_axis
                            p1d_data["ind_tau"] = floor_scaling + pp

                            # iterate over properties
                            for ind_key in range(len(keys_copy_in)):
                                key_in = keys_copy_in[ind_key]
                                key_out = keys_copy_out[ind_key]
                                if (
                                    (key_in == "sim_scale_T0")
                                    | (key_in == "sim_scale_gamma")
                                    | (key_in == "scale_tau")
                                    | (key_in == "mF")
                                ):
                                    if key_in in temp_data:
                                        p1d_data[key_out] = temp_data[key_in]
                                elif key_in == "p3d_data":
                                    # all axes
                                    p1d_data["p3d_Mpc"] = np.array(
                                        temp_data["p3d_data"]["p3d_Mpc"]
                                    )
                                    p1d_data["k3d_Mpc"] = np.array(
                                        temp_data["p3d_data"]["k_Mpc"]
                                    )
                                    p1d_data["mu3d"] = np.array(
                                        temp_data["p3d_data"]["mu"]
                                    )
                                elif (key_in == "p1d_Mpc") | (key_in == "k_Mpc"):
                                    p1d_data[key_out] = np.array(temp_data[key_in])
                                else:
                                    p1d_data[key_out] = temp_param[key_in]

                            self.data.append(p1d_data)

        # create 1D arrays with all entries for a given parameter
        self._store_param_arrays()

    def _store_param_arrays(self):
        """
        Create 1D arrays with all entries for a given parameter.

        Returns:
            None

        """

        list_keys = [
            "ind_sim",
            "ind_tau",
            "ind_z",
            "ind_phase",
            "ind_axis",
        ]

        # put measurements in arrays
        _dict = {}
        N = len(self.data)
        for key in list_keys:
            if key not in self.data[0]:
                continue
            elif (
                (key != "k_Mpc")
                & (key != "p1d_Mpc")
                & (key != "k3d_Mpc")
                & (key != "mu3d")
                & (key != "p3d_Mpc")
            ):
                _dict[key] = np.zeros(N)
            else:
                _dict[key] = np.zeros((N, *self.data[0][key].shape))

            for ii in range(N):
                if (
                    (key != "k_Mpc")
                    & (key != "p1d_Mpc")
                    & (key != "k3d_Mpc")
                    & (key != "mu3d")
                    & (key != "p3d_Mpc")
                ):
                    _dict[key][ii] = self.data[ii][key]
                else:
                    _dict[key][ii] = self.data[ii][key][: _dict[key][ii].shape[0]]

        for key in _dict.keys():
            setattr(self, key, _dict[key])

    def average_over_samples(self, flag="all"):
        """
        Compute averages over either phases, axes, or all.

        Args:
            flag (str): Flag indicating the type of averaging. Valid options are:
                - "all": Compute averages over all phases, axes, and simulations.
                - "phases": Compute averages over phases and simulations, keeping axes fixed.
                - "axes": Compute averages over axes and simulations, keeping phases fixed.
                (default: "all")

        Returns:
            None

        Note:
            This method attaches a new attribute named "data_av_{flag}" to the class,
            containing the computed averages.

        """

        keys_same = [
            "ind_sim",
            "ind_tau",
            "ind_z",
            "ind_phase",
            "ind_axis",
            "Delta2_p",
            "n_p",
            "alpha_p",
            "f_p",
            "z",
            "scale_tau",
            "scale_T0",
            "scale_gamma",
        ]

        keys_merge = [
            "mF",
            "T0",
            "gamma",
            "sigT_Mpc",
            "kF_Mpc",
            "k_Mpc",
            "p1d_Mpc",
        ]

        if self.also_P3D:
            keys_merge.append("k3d_Mpc")
            keys_merge.append("mu3d")
            keys_merge.append("p3d_Mpc")

        # get number of simulations, scalings, and redshifts
        n_sims = np.unique(self.ind_sim).shape[0]
        n_tau = np.unique(self.ind_tau).shape[0]
        n_z = np.unique(self.ind_z).shape[0]

        # tot_nsam: number of samples per cosmology
        if flag == "all":
            tot_nsam = self.n_phases * self.n_axes
            loop = list(
                product(
                    np.unique(self.ind_sim).astype(int),
                    np.unique(self.ind_tau).astype(int),
                    np.unique(self.ind_z).astype(int),
                )
            )
        elif flag == "phases":
            tot_nsam = self.n_phases
            loop = list(
                product(
                    np.unique(self.ind_sim).astype(int),
                    np.unique(self.ind_tau).astype(int),
                    np.unique(self.ind_z).astype(int),
                    np.arange(self.n_axes),
                )
            )
        elif flag == "axes":
            tot_nsam = self.n_axes
            loop = list(
                product(
                    np.unique(self.ind_sim).astype(int),
                    np.unique(self.ind_tau).astype(int),
                    np.unique(self.ind_z).astype(int),
                    np.arange(self.n_phases),
                )
            )

        nloop = len(loop)

        list_new = []
        for ind_loop in range(nloop):
            if flag == "all":
                ind_sim, ind_tau, ind_z = loop[ind_loop]
                ind_merge = np.argwhere(
                    (self.ind_sim == ind_sim)
                    & (self.ind_tau == ind_tau)
                    & (self.ind_z == ind_z)
                )[:, 0]
            elif flag == "phases":
                ind_sim, ind_tau, ind_z, ind_axis = loop[ind_loop]
                ind_merge = np.argwhere(
                    (self.ind_sim == ind_sim)
                    & (self.ind_tau == ind_tau)
                    & (self.ind_z == ind_z)
                    & (self.ind_axis == ind_axis)
                )[:, 0]
            elif flag == "axes":
                ind_sim, ind_tau, ind_z, ind_phase = loop[ind_loop]
                ind_merge = np.argwhere(
                    (self.ind_sim == ind_sim)
                    & (self.ind_tau == ind_tau)
                    & (self.ind_z == ind_z)
                    & (self.ind_phase == ind_phase)
                )[:, 0]

            dict_av = {}
            for key in keys_same:
                dict_av[key] = self.data[ind_merge[0]][key]

            for key in keys_merge:
                if (
                    (key == "mF")
                    | (key == "T0")
                    | (key == "gamma")
                    | (key == "sigT_Mpc")
                    | (key == "kF_Mpc")
                ):
                    mean = 0
                else:
                    mean = np.zeros_like(self.data[ind_merge[0]][key])

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

            list_new.append(dict_av)

        setattr(self, "data_av_" + flag, list_new)

    def input_emulator(self, flag="all"):
        """
        Generate an input emulator based on the specified flag.

        Old function, it is superseeded by get_training_data

        Args:
            flag (str): Flag indicating the type of input emulator to generate. Valid options are:
                - "all": Combine both the original data and averaged data over all phases, axes, and simulations.
                - "phases": Combine the averaged data over phases and the original data.
                - "axes": Combine the averaged data over axes and the original data.
                (default: "all")

        Returns:
            None

        Note:
            This method attaches a new attribute named "data_input_{flag}" to the class,
            containing the generated input emulator.

        """
        archive_both = []
        if flag == "all":
            for ii in range(len(self.data)):
                archive_both.append(self.data[ii])
            for ii in range(len(self.data_av_all)):
                archive_both.append(self.data_av_all[ii])
        elif flag == "phases":
            for ii in range(len(self.data_av_phases)):
                archive_both.append(self.data_av_phases[ii])
            for ii in range(len(self.data_av_all)):
                archive_both.append(self.data_av_all[ii])
        elif flag == "axes":
            for ii in range(len(self.data_av_axes)):
                archive_both.append(self.data_av_axes[ii])
            for ii in range(len(self.data_av_all)):
                archive_both.append(self.data_av_all[ii])

        setattr(self, "data_input_" + flag, archive_both)

    def get_training_data(self, average=None, tau_scaling=None):
        """
        Retrieves the training data based on the provided flag.

        Parameters:
            self (object): The object instance.
            average (str, optional): The flag indicating the desired training data. Defaults to "axes_phases_all".

        Returns:
            None.

        Raises:
            AssertionError: If the flag is not a string or if it contains an invalid operation.

        Notes:
            The retrieved training data is stored in the 'training_data' attribute of the parent class.

        """

        ## check average keyword makes sense
        if average is None:
            try:
                if self.sim_suite == "Cabayol23":
                    average = "axes_phases_all"
                elif self.sim_suite == "768_768":
                    average = "axes_phases_all"
                elif self.sim_suite == "Pedersen21":
                    average = "all"
                else:
                    print(self.sim_suite + " not implemented")
                    raise
            except:
                print("An error occurred when checking the value of flag")
                raise

        # ensures try is a string
        assert isinstance(average, str), "Variable is not a string."

        possible_operations = ["axes", "phases", "all", "individual"]
        operations = split_string(average)

        for operation in operations:
            err = operation + " is not within allowed flags: "
            for op in possible_operations:
                err += op + " "
            assert operation in possible_operations, err

        ## check tau_scaling keyword makes sense
        if tau_scaling is None:
            try:
                if self.sim_suite == "Cabayol23":
                    tau_scaling = None
                elif self.sim_suite == "768_768":
                    tau_scaling = None
                elif self.sim_suite == "Pedersen21":
                    tau_scaling = 1
                else:
                    print(self.sim_suite + " not implemented")
                    raise
            except:
                print("An error occurred when checking the value of flag")
                raise
        else:
            assert isinstance(tau_scaling, (int, float)), "Variable is not a number."
            try:
                if self.sim_suite == "Cabayol23":
                    possible_scalings = [0.9, 0.95, 1, 1.05, 1.1]
                elif self.sim_suite == "768_768":
                    possible_scalings = [0.9, 0.95, 1, 1.05, 1.1]
                elif self.sim_suite == "Pedersen21":
                    possible_scalings = [0.9, 1, 1.1]
                else:
                    print(self.sim_suite + " not implemented")
                    raise
            except:
                print("An error occurred when checking the value of sim_suite")
                raise

            # check that the selecting scaling is available
            err = (
                "tau_scaling: "
                + str(tau_scaling)
                + " not implemented for "
                + self.sim_suite
                + "\n the list of possible tau_scalings is: "
            )
            for sc in possible_scalings:
                err += str(sc) + " "
            assert tau_scaling in possible_scalings, err

        ## put training points here
        training_data = []

        if "axes" in operations:
            # include average over axes
            self.average_over_samples(flag="axes")
            for ii in range(len(self.data_av_axes)):
                if (tau_scaling is None) | (
                    self.data_av_axes[ii]["scale_tau"] == tau_scaling
                ):
                    _ = self.data_av_axes[ii]
                    _["ind_axis"] = 999
                    training_data.append(_)

        if "phases" in operations:
            # include average over phases
            self.average_over_samples(flag="phases")
            for ii in range(len(self.data_av_phases)):
                if (tau_scaling is None) | (
                    self.data_av_phases[ii]["scale_tau"] == tau_scaling
                ):
                    _ = self.data_av_phases[ii]
                    _["ind_phase"] = 999
                    training_data.append(_)

        if "all" in operations:
            # include average over axes and phases
            self.average_over_samples(flag="all")
            for ii in range(len(self.data_av_all)):
                if (tau_scaling is None) | (
                    self.data_av_all[ii]["scale_tau"] == tau_scaling
                ):
                    _ = self.data_av_all[ii]
                    _["ind_axis"] = 999
                    _["ind_phase"] = 999
                    training_data.append(_)

        if "individual" in operations:
            # include individual measurements
            for ii in range(len(self.data)):
                if (tau_scaling is None) | (self.data[ii]["scale_tau"] == tau_scaling):
                    training_data.append(self.data[ii])

        setattr(self, "training_data", training_data)

    def get_testing_data(self, tau_scaling=1):
        """
        Retrieves the testing data based on the provided flag.

        Parameters:
            self (object): The object instance.
            flag (str, optional): The flag indicating the desired training data. Defaults to "axes_phases_all".

        Returns:
            None.

        Raises:
            AssertionError: If the flag is not a string or if it contains an invalid operation.

        Notes:
            The retrieved training data is stored in the 'training_data' attribute of the parent class.

        """

        # ensures tau_scaling is a number
        assert isinstance(tau_scaling, (int, float)), "Variable is not a number."

        # get list of available scalings for each sim_suite
        try:
            if self.sim_suite == "Cabayol23":
                possible_scalings = [0.9, 0.95, 1, 1.05, 1.1]
            elif self.sim_suite == "768_768":
                possible_scalings = [0.9, 0.95, 1, 1.05, 1.1]
            elif self.sim_suite == "Pedersen21":
                possible_scalings = [0.9, 1, 1.1]
            else:
                print(self.sim_suite + " not implemented")
                raise
        except:
            print("An error occurred when checking the value of sim_suite")
            raise

        # check that the selecting scaling is available
        err = (
            "tau_scaling: "
            + str(tau_scaling)
            + " not implemented for "
            + self.sim_suite
            + "\n the list of possible tau_scalings is: "
        )
        for sc in possible_scalings:
            err += str(sc) + " "
        assert tau_scaling in possible_scalings, err

        # to contain all points used in training
        testing_data = []

        # compute average over axes and phases
        self.average_over_samples(flag="all")
        for ii in range(len(self.data_av_all)):
            if self.data_av_all[ii]["scale_tau"] == tau_scaling:
                _ = self.data_av_all[ii]
                _["ind_axis"] = 999
                _["ind_phase"] = 999
                training_data.append(_)

        setattr(self, "testing_data", testing_data)
