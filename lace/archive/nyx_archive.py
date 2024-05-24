import numpy as np
import os
import h5py

from lace.cosmo.thermal_broadening import thermal_broadening_kms
from lace.archive.base_archive import BaseArchive
from lace.utils.misc import split_string


def get_attrs(h5py_data):
    dict_params = {}
    for ipar in h5py_data.attrs.keys():
        dict_params[ipar] = h5py_data.attrs[ipar]
    return dict_params


class NyxArchive(BaseArchive):
    """
    Bookkeeping of Lya flux P1D & P3D measurements from a suite of Nyx simulations.
    """

    def __init__(
        self,
        nyx_version="Oct2023",
        nyx_file=None,
        kp_Mpc=None,
        force_recompute_linP_params=False,
        verbose=False,
        nfiles=18,
    ):
        ## check input
        if isinstance(nyx_version, str) == False:
            raise TypeError("nyx_version must be a string")
        self.nyx_version = nyx_version

        if isinstance(nyx_file, (str, type(None))) == False:
            raise TypeError("nyx_file must be a string or None")

        if isinstance(force_recompute_linP_params, bool) == False:
            raise TypeError("update_kp must be boolean")

        if force_recompute_linP_params:
            if isinstance(kp_Mpc, (int, float)) == False:
                raise TypeError(
                    "kp_Mpc must be a number if force_recompute_linP_params == True"
                )
        else:
            if isinstance(kp_Mpc, (int, float, type(None))) == False:
                raise TypeError("kp_Mpc must be a number or None")
        self.kp_Mpc = kp_Mpc

        if isinstance(verbose, bool) == False:
            raise TypeError("verbose must be boolean")
        self.verbose = verbose
        ## done check input

        ## sets list simulations available for this suite
        # list of especial simulations
        self.list_sim_test = ["nyx_central", "nyx_seed", "nyx_wdm"]
        # list of hypercube simulations
        self.list_sim_cube = []
        # simulation 14 was identified as problematic by the Nyx team
        for ii in range(nfiles):
            self.list_sim_cube.append("nyx_" + str(ii))
        # list all simulations
        self.list_sim = self.list_sim_cube + self.list_sim_test
        ## done set simulation list

        self.list_sim_axes = [0, 1, 2]

        # get relevant flags for post-processing
        self._set_info_sim(nfiles)

        # load power spectrum measurements
        self._load_data(
            nyx_file, force_recompute_linP_params
        )

        # extract indexes from data
        self._set_labels()

    def _set_info_sim(self, nfiles):
        # number of simulation phases (IC)
        self.n_phases = 1
        # number of simulation axes in the post-processing
        self.n_axes = 3
        # P3D configuration for fine binning
        self.also_P3D = True
        self.mubins = 20
        self.kbins = 2047
        # available scalings at best for each simulation
        self.scalings_avail = list(np.arange(22))
        # training options
        self.training_average = "both"
        self.training_val_scaling = "all"
        self.training_z_min = 0
        self.training_z_max = 10
        # testing options
        self.testing_ind_rescaling = 0
        self.testing_z_min = 0
        self.testing_z_max = 10

        self.key_conv = {
            "Fbar": "mF",
            "T_0": "T0",
            "gamma": "gamma",
            "tau_rescale_factor": "val_scaling",
            "lambda_P": "kF_Mpc",
        }
        # sigT_Mpc missing after gamma

        # conversion between names of the simulations in Nyx
        # file and lace bookkeeping
        self.sim_conv = {
            "fiducial": "nyx_central",
            "bar_ic_grid_3": "nyx_seed",
            "wdm_3.5kev_grid_1": "nyx_wdm",
        }
        for ii in range(nfiles):
            self.sim_conv["cosmo_grid_" + str(ii)] = "nyx_" + str(ii)

        # list of available redshifts at Nyx (not all sims have them)
        # self.list_sim_redshifts=np.append(np.arange(2.0,4.5,0.2),
        #        np.arange(4.6,5.5,0.4))
        self.list_sim_redshifts = [
            2.0,
            2.2,
            2.4,
            2.6,
            2.8,
            3.0,
            3.2,
            3.4,
            3.6,
            3.8,
            4.0,
            4.2,
            4.4,
            4.6,
            5.0,
            5.4,
        ]

        # conversion between nyx and lace snapshot names
        self.z_conv = {}
        for ii in range(len(self.list_sim_redshifts)):
            flag = "redshift_" + str(self.list_sim_redshifts[ii])
            self.z_conv[flag] = ii

        # conversion between scaling names
        self.scaling_conv = {
            "native_parameters": 0,
            "rescale_Fbar_fiducial": 1,
        }
        for ii in range(15):
            self.scaling_conv["thermal_grid_" + str(ii)] = ii + 2
        for ii in range(5):
            self.scaling_conv["new_thermal_lhd_" + str(ii)] = ii + 17

        self.axis_conv = {
            "x": 0,
            "y": 1,
            "z": 2,
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

    def _get_emu_cosmo(
        self, nyx_data, sim_label, force_recompute_linP_params=False
    ):
        """
        Get the cosmology and parameters describing linear power spectrum from simulation.

        Args:
            nyx_data: file containing nyx data
            sim_label: selected simulation
            force_recompute_linP_params: compute linP even if kp_Mpc match

        Returns:
            tuple: A tuple containing the following info:
                - cosmo_params (dict): contains cosmlogical parameters
                - linP_params (dict): contains parameters describing linear power spectrum

        """

        isim = self.sim_conv[sim_label]

        # figure out whether we need to compute linP params
        compute_linP_params = False

        if force_recompute_linP_params:
            compute_linP_params = True
        else:
            # open file with precomputed values to check kp_Mpc
            try:
                file_cosmo = np.load(self.file_cosmo, allow_pickle=True)
            except Exception as e:
                raise e

            sim_in_file = False
            for ii in range(len(file_cosmo)):
                if file_cosmo[ii]["sim_label"] == isim:
                    sim_in_file = True
                    # if kp_Mpc not defined, use precomputed value
                    if self.kp_Mpc is None:
                        self.kp_Mpc = file_cosmo[ii]["linP_params"]["kp_Mpc"]

                    # if kp_Mpc different from precomputed value, compute
                    if self.kp_Mpc != file_cosmo[ii]["linP_params"]["kp_Mpc"]:
                        if self.verbose:
                            print("Recomputing kp_Mpc at " + str(self.kp_Mpc))
                        compute_linP_params = True
                    else:
                        cosmo_params = file_cosmo[ii]["cosmo_params"]
                        linP_params = file_cosmo[ii]["linP_params"]
                    break
            if sim_in_file == False:
                file_error = (
                    "The file "
                    + file_cosmo
                    + " does not contain "
                    + isim
                    + ". To speed up calculations, "
                    + " you can recompute the file by running "
                    + "lace/scripts/compute_nyx_emu_cosmo.py"
                )
                if self.verbose:
                    print(file_error)
                compute_linP_params = True

        if compute_linP_params == True:
            # this is the only place where you need CAMB
            if self.verbose:
                print("We are using CAMB")
            from lace.cosmo import camb_cosmo, fit_linP

            cosmo_params = get_attrs(nyx_data[sim_label])
            if isim == "nyx_central":
                cosmo_params["A_s"] = 2.10e-9
                cosmo_params["n_s"] = 0.966
                cosmo_params["h"] = cosmo_params["H_0"] / 100
            # setup CAMB object
            sim_cosmo = camb_cosmo.get_Nyx_cosmology(cosmo_params)

            # compute linear power parameters at each z (in Mpc units)
            linP_zs = fit_linP.get_linP_Mpc_zs(
                sim_cosmo, self.list_sim_redshifts, self.kp_Mpc
            )
            zs = np.array(self.list_sim_redshifts)
            # compute conversion from Mpc to km/s using cosmology
            dkms_dMpc_zs = camb_cosmo.dkms_dMpc(sim_cosmo, z=zs)

            linP_params = {}
            linP_params["kp_Mpc"] = self.kp_Mpc
            labels = ["z", "dkms_dMpc", "Delta2_p", "n_p", "alpha_p", "f_p"]
            for lab in labels:
                linP_params[lab] = np.zeros(zs.shape[0])
                for ii in range(zs.shape[0]):
                    if lab == "z":
                        linP_params[lab][ii] = zs[ii]
                    elif lab == "dkms_dMpc":
                        linP_params[lab][ii] = dkms_dMpc_zs[ii]
                    else:
                        linP_params[lab][ii] = linP_zs[ii][lab]

        return cosmo_params, linP_params

    def _load_data(self, nyx_file=None, force_recompute_linP_params=False):
        # set nyx_file if not provided
        if nyx_file is None:
            if "NYX_PATH" not in os.environ:
                error_text = (
                    "If nyx_file is not provided, you must define"
                    + "the environ variable NYX_PATH pointing to the folder containing"
                    + "the hdf5 file containing Nyx data"
                )
                raise ValueError(error_text)
            nyx_file = (
                os.environ["NYX_PATH"]
                + "/models_Nyx_"
                + self.nyx_version
                + ".hdf5"
            )

        try:
            ff = h5py.File(nyx_file, "r")
        except Exception as e:
            raise e
        else:
            # set self.file_cosmo
            if "NYX_PATH" not in os.environ:
                folder = os.path.dirname(nyx_file)
            else:
                folder = os.environ["NYX_PATH"]
            self.file_cosmo = (
                folder + "/nyx_emu_cosmo_" + self.nyx_version + ".npy"
            )

        # store each measurement as an entry of the following list
        # each entry is a dictionary containing all relevant info
        self.data = []
        # loop over simulations. always check if avail because there are some
        # configurations missing here and there
        sim_avail = list(ff.keys())
        for isim in sim_avail:
            # this simulation seems to have issues
            if isim == "cosmo_grid_14":
                continue

            if self.verbose:
                print("read Nyx sim", isim)

            # read cosmo information and linear power parameters
            # (will only need CAMB if pivot point kp_Mpc is changed)
            cosmo_params, linP_params = self._get_emu_cosmo(
                ff, isim, force_recompute_linP_params
            )

            # loop over redshifts
            z_avail = list(ff[isim].keys())
            for iz in z_avail:
                zval = float(split_string(iz)[1])

                # find redshift index to read linear power parameters
                ind_z = np.where(
                    np.isclose(self.list_sim_redshifts, zval, 1e-10)
                )[0][0]

                scalings_avail = list(ff[isim][iz].keys())
                # loop over scalings
                for iscaling in scalings_avail:
                    if iscaling == "full_box_stats":
                        continue
                    # loop over axes
                    axes_avail = list(
                        ff[isim][iz][iscaling]["individual_axes"].keys()
                    )
                    for iaxis in axes_avail:
                        # dictionary containing measurements and relevant info
                        _arch = {}
                        _arch["cosmo_params"] = cosmo_params
                        for lab in linP_params.keys():
                            if lab == "kp_Mpc":
                                _arch[lab] = linP_params[lab]
                            else:
                                _arch[lab] = linP_params[lab][ind_z]

                        _arch["sim_label"] = self.sim_conv[isim]
                        _arch["ind_snap"] = self.z_conv[iz]
                        # no multiple phases for any simulation
                        _arch["ind_phase"] = 0
                        _arch["ind_axis"] = self.axis_conv[iaxis]
                        _arch["ind_rescaling"] = self.scaling_conv[iscaling]

                        ## store IGM parameters
                        # store temperature parameters
                        _igm = get_attrs(ff[isim][iz][iscaling])
                        for key in _igm.keys():
                            _arch[self.key_conv[key]] = _igm[key]

                        # compute thermal broadening in Mpc
                        sigma_T_kms = thermal_broadening_kms(_arch["T0"])
                        _arch["sigT_Mpc"] = (
                            sigma_T_kms / linP_params["dkms_dMpc"][ind_z]
                        )

                        # store pressure parameters
                        # not available in bar_ic_grid_3 and wdm_3.5kev_grid_1
                        # should it be the same as nyx_3 and nyx_1?
                        _pressure = get_attrs(ff[isim][iz])
                        key = "lambda_P"
                        if key in _pressure.keys():
                            _arch[self.key_conv[key]] = 1 / (
                                1e-3 * _pressure[key]
                            )
                        ## done store IGM parameters

                        ## store P1D and P3D measurements (if available)
                        _data = ff[isim][iz][iscaling]["individual_axes"][iaxis]
                        p1d = np.array(_data["1d power"])
                        _arch["k_Mpc"] = p1d["k"]
                        _arch["p1d_Mpc"] = p1d["Pk1d"]
                        if "3d power kmu" in _data:
                            p3d = np.array(
                                _data["3d power kmu"]["fine binning"]
                            )
                            _arch["k3d_Mpc"] = p3d["k"].reshape(
                                self.kbins, self.mubins
                            )
                            _arch["mu3d"] = p3d["mu"].reshape(
                                self.kbins, self.mubins
                            )
                            _arch["p3d_Mpc"] = p3d["Pkmu"].reshape(
                                self.kbins, self.mubins
                            )
                        self.data.append(_arch)
