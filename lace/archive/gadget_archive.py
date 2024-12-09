import numpy as np
import json
from pathlib import Path

from lace.setup_simulations import read_genic, read_gadget
from lace.archive.base_archive import BaseArchive
from lace.utils.exceptions import ExceptionList
from lace.emulator.constants import PROJ_ROOT
from lace.archive.constants import (
    Postprocessings,
    PEDERSEN21_CONFIG,
    CABAYOL23_CONFIG,
    CONFIG_768_768,
    PEDERSEN21_MAPPINGS,
    CABAYOL23_MAPPINGS,
    CABAYOL23_PARAM_MAPPINGS,
    LIST_SIM_REDSHIFTS_MPG,
    LIST_SIM_MPG_CUBE,
    LIST_SIM_MPG_TEST,
    LIST_ALL_SIMS_MPG
)



class GadgetArchive(BaseArchive):
    """
    Bookkeeping of Lya flux P1D & P3D measurements from a suite of Gadget simulations.

    Methods:
        __init__(self, postproc, linP_dir)
        _set_info_postproc(self, postproc)
        _sim2file_name(self, sim_label)
        _get_emu_cosmo(self, sim_label, force_recompute_linP_paramsi=False)
        _get_file_names(self, sim_label, ind_phase, ind_z, ind_axis)
        _get_sim(self, sim_label, ind_z, ind_axis)
        _load_data(self, force_recompute_linP_params=False)
    """

    def __init__(
        self,
        postproc: str = "Cabayol23",
        kp_Mpc: int | float | None = None,
        force_recompute_linP_params: bool = False,
        verbose: bool = False,
        z_star: float = 3,
        kp_kms: float = 0.009,
    ) -> None:
        """Initialize the archive object.

        Args:
            postproc (str): Specify post-processing run. Default is "Cabayol23".
                Raises a ValueError if the postproc is not available
            kp_Mpc (None or float): Optional. Pivot point used in linear power parameters.
                If specified, the parameters will be recomputed in the archive. Default is None.
            force_recompute_linP_params (bool): If set, it will recompute linear power parameters even if kp_Mpc match. Default is False.
            verbose (bool): Whether to print verbose output. Default is False.
            z_star (float): Reference redshift. Default is 3.
            kp_kms (float): Pivot point in velocity units. Default is 0.009.

        Returns:
            None

        """

        ## check input
        if postproc not in Postprocessings:
            raise ExceptionList("Invalid postproc value. Available options:", list(Postprocessings))
        
        if force_recompute_linP_params and kp_Mpc is None:
            raise TypeError("kp_Mpc must be a number if force_recompute_linP_params == True")

        self.kp_Mpc = kp_Mpc
        self.z_star = z_star
        self.kp_kms = kp_kms
        self.verbose = verbose

        # Set available simulations
        self.list_sim_test = LIST_SIM_MPG_TEST
        self.list_sim_cube = LIST_SIM_MPG_CUBE
        self.list_sim = LIST_ALL_SIMS_MPG
        self.list_sim_redshifts = LIST_SIM_REDSHIFTS_MPG

        # Set available axes based on postproc
        self.list_sim_axes = {
            "Pedersen21": [0],
            "Cabayol23": [0, 1, 2],
            "768_768": [0, 1, 2]
        }[postproc]

        self._set_info_postproc(postproc)
        self._load_data(force_recompute_linP_params)
        self._set_labels()

    def _set_info_postproc(self, postproc):
        """
        Set information about the post-processing suite and set corresponding attributes.

        Args:
            postproc (str): Name of the simulation suite.

        Returns:
            None
        """
        self.postproc = postproc
        
        config_map = {
            "Pedersen21": PEDERSEN21_CONFIG,
            "Cabayol23": CABAYOL23_CONFIG,
            "768_768": CONFIG_768_768
        }
        config = config_map.get(postproc)

        if config:
            for key, value in config.items():
                setattr(self, key, value)

        self.basedir = Path(str(self.basedir).lstrip('/'))
        self.basedir_params = Path(str(self.basedir_params).lstrip('/'))

        self.fulldir = PROJ_ROOT / self.basedir
        self.fulldir_param = PROJ_ROOT / self.basedir_params

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

        self.scaling_cov = {
            1: 0,
            0.90: 1,
            0.95: 2,
            1.05: 3,
            1.1: 4,
        }

    def _sim2file_name(self, sim_label):
        """
        Convert simulation labels to file names.

        Args:
            sim_label (int or str): Selected simulation.

        Returns:
            tuple: Tuple containing the conversion dictionaries for the simulation and parameters.
        """
        
        mapping_configs = {
            "Pedersen21": (PEDERSEN21_MAPPINGS, PEDERSEN21_MAPPINGS),
            "Cabayol23": (CABAYOL23_MAPPINGS, CABAYOL23_PARAM_MAPPINGS),
            "768_768": (CABAYOL23_MAPPINGS, CABAYOL23_MAPPINGS)
        }
        
        dict_conv, dict_conv_params = mapping_configs[self.postproc]

        # Add dynamic mappings
        for ii in range(30):
            dict_conv["mpg_" + str(ii)] = "sim_pair_" + str(ii)
            dict_conv_params["mpg_" + str(ii)] = "sim_pair_" + str(ii)

        tag_param = self.tag_param if sim_label in self.list_sim_test else "parameter.json"

        return dict_conv[sim_label], dict_conv_params[sim_label], tag_param

    def _get_emu_cosmo(self, sim_label, force_recompute_linP_params=False):
        """Get cosmology and linear power spectrum parameters."""
        compute_linP_params = force_recompute_linP_params

        if not compute_linP_params:
            fname = self.fulldir / "mpg_emu_cosmo.npy"
            try:
                file_cosmo = np.load(fname, allow_pickle=True).item()
                if sim_label not in file_cosmo:
                    if self.verbose:
                        file_error = (
                            "The file "
                            + fname
                            + " does not contain "
                            + sim_label
                            + ". To speed up calculations, "
                            + " you can recompute the file by running "
                            + "lace/scripts/developers/compute_nyx_emu_cosmo.py"
                            )   
                        print(file_error)
                    compute_linP_params = True
                else:
                    if self.kp_Mpc is None:
                        self.kp_Mpc = file_cosmo[sim_label]["linP_params"]["kp_Mpc"]
                    if self.kp_Mpc != file_cosmo[sim_label]["linP_params"]["kp_Mpc"]:
                        if self.verbose:
                            print(f"Recomputing kp_Mpc at {self.kp_Mpc}")
                        compute_linP_params = True
                    else:
                        cosmo_params = file_cosmo[sim_label]["cosmo_params"]
                        linP_params = file_cosmo[sim_label]["linP_params"]
                        star_params = file_cosmo[sim_label]["star_params"]
            except:
                raise IOError(f"File {fname} does not exist")

        if compute_linP_params:
            from lace.cosmo import camb_cosmo, fit_linP

            _, sim_name_param, tag_param = self._sim2file_name(sim_label)
            pair_dir = self.fulldir_param / sim_name_param

            gadget_fname = pair_dir / "sim_plus/paramfile.gadget"
            gadget_cosmo = read_gadget.read_gadget_paramfile(gadget_fname)
            zs = read_gadget.snapshot_redshifts(gadget_cosmo)

            genic_fname = pair_dir / "sim_plus/paramfile.genic"
            cosmo_params = read_genic.camb_from_genic(genic_fname)

            # setup CAMB object
            sim_cosmo = camb_cosmo.get_cosmology_from_dictionary(cosmo_params)

            # compute linear power parameters at each z (in Mpc units)
            linP_zs = fit_linP.get_linP_Mpc_zs(sim_cosmo, zs, self.kp_Mpc)

            # compute linear power parameters (in kms units)
            star_params = fit_linP.parameterize_cosmology_kms(
                sim_cosmo, None, self.z_star, self.kp_kms
            )
            # compute conversion from Mpc to km/s using cosmology
            dkms_dMpc_zs = camb_cosmo.dkms_dMpc(sim_cosmo, z=np.array(zs))

            linP_params = {
                "kp_Mpc": self.kp_Mpc,
                "z": zs,
                "dkms_dMpc": dkms_dMpc_zs,
                "Delta2_p": np.array([lp["Delta2_p"] for lp in linP_zs]),
                "n_p": np.array([lp["n_p"] for lp in linP_zs]),
                "alpha_p": np.array([lp["alpha_p"] for lp in linP_zs]),
                "f_p": np.array([lp["f_p"] for lp in linP_zs])
            }

        return cosmo_params, linP_params, star_params

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
        tag_phase = "sim_plus" if ind_phase == 0 else "sim_minus"

        if self.postproc == "Pedersen21":
            sk_label_data = self.sk_label
            sk_label_params = self.sk_label_params
            n_it_files = 1
        else:
            sk_label_data = f"{self.sk_label}_axis{ind_axis + 1}"
            sk_label_params = (f"{self.sk_label_params}_axis{ind_axis + 1}" 
                             if self.postproc != "Cabayol23" else self.sk_label_params)
            n_it_files = 1 if sim_name in ["diffSeed", "P18", "running", "curved_003"] else 2

        data_json = [
            str(self.fulldir / sim_name / tag_phase / 
                f"{'p1d_setau' if it else self.p1d_label}_{ind_z}_{sk_label_data}.json")
            for it in range(n_it_files)
        ]

        param_json = str(
            self.fulldir_param / sim_name_param / tag_phase / 
            f"{self.p1d_label_params}_{ind_z}_{sk_label_params}.json"
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
            path_data, path_param = self._get_file_names(sim_label, ind_phase, ind_z, ind_axis)

            for path in path_data:
                with open(path) as f:
                    phase_data.append(json.load(f))
                    arr_phase.append(ind_phase)

            with open(path_param) as f:
                phase_params.append(json.load(f))

        return phase_data, phase_params, arr_phase

    def _load_data(self, force_recompute_linP_params):
        """
        Setup the archive by gathering information from all measured power spectra in the simulations.

        Args:
            force_recompute_linP_params: recompute linP even if kp_Mpc matches

        Returns:
            None

        Notes:
        - This function reads and processes data from all measured power spectra in the simulations.
        - The function iterates over simulations, snapshots, and axes to retrieve the required data.
        - The retrieved data is stored in the `self.data` attribute as a list of dictionaries,
            with each dictionary representing a specific measurement and simulation configuration.

        """

        # All samples have an entry in this list, a dictionary that includes
        # P1D and P3D measurements and info about simulations
        self.data = []

        try:
            with open(self.fulldir / "latin_hypercube.json") as f:
                self.cube_data = json.load(f)
                self.nsamples = self.cube_data["nsamples"]
        except FileNotFoundError:
            print(f"Error: latin_hypercube.json not found in {self.fulldir}")

        ## read info from all sims, all snapshots, all rescalings
        # iterate over simulations
        for sim_label in self.list_sim:
            cosmo_params, linP_params, star_params = self._get_emu_cosmo(
                sim_label, force_recompute_linP_params
            )

            for ind_z in range(len(linP_params["z"])):
                snap_data = {
                    "cosmo_params": cosmo_params,
                    "star_params": star_params,
                    **{k: v[ind_z] if isinstance(v, np.ndarray) else v 
                       for k, v in linP_params.items()}
                }

                # iterate over axes
                for ind_axis in range(self.n_axes):
                    phase_data, phase_params, arr_phase = self._get_sim(
                        sim_label, ind_z, ind_axis
                    )

                    for ind_phase, (data, phase) in enumerate(zip(phase_data, arr_phase)):
                        n_scalings = len(data["p1d_data"])
                        floor_scaling = (len(phase_data[0]["p1d_data"]) 
                                      if len(phase_data) > 2 and ind_phase % 2 
                                      else 0)

                        for pp in range(n_scalings):
                            temp_data = phase_data[ind_phase]["p1d_data"][pp]
                            _ind_phase = arr_phase[ind_phase]
                            temp_param = phase_params[_ind_phase]["p1d_data"][0]

                            sim_data = {
                                **snap_data,
                                "sim_label": sim_label,
                                "ind_snap": ind_z,
                                "ind_phase": phase,
                                "ind_axis": ind_axis,
                                "ind_rescaling": self.scaling_cov[temp_data["scale_tau"]],
                                **{self.key_conv[k]: (
                                    np.array(temp_data[k]) if k in ["p1d_Mpc", "k_Mpc"]
                                    else temp_data[k] if k in ["mF", "scale_tau"]
                                    else temp_param[k]
                                ) for k in self.key_conv}
                            }

                            if self.also_P3D:
                                p3d_data = temp_data["p3d_data"]
                                sim_data.update({
                                    "p3d_Mpc": np.array(p3d_data["p3d_Mpc"]),
                                    "k3d_Mpc": np.array(p3d_data["k_Mpc"]),
                                    "mu3d": np.array(p3d_data["mu"])
                                })

                            self.data.append(sim_data)
