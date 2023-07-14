import numpy as np
import os
import h5py

from lace.cosmo import camb_cosmo, fit_linP
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

    def __init__(self, file_name=None, kp_Mpc=0.7, verbose=False, zmax=None):
        ## check input
        if isinstance(file_name, (str, type(None))) == False:
            raise TypeError("file_name must be a string or None")

        if isinstance(kp_Mpc, (int, float, type(None))) == False:
            raise TypeError("kp_Mpc must be a number or None")
        self.kp_Mpc = kp_Mpc
        self.verbose = verbose
        ## done check input

        ## sets list simulations available for this suite
        # list of especial simulations
        self.list_sim_test = ["nyx_central", "nyx_seed", "nyx_wdm"]
        # list of hypercube simulations
        self.list_sim_cube = []
        # simulation 14 was identified as problematic by the Nyx team
        for ii in range(14):
            self.list_sim_cube.append("nyx_" + str(ii))
        # list all simulations
        self.list_sim = self.list_sim_cube + self.list_sim_test
        ## done set simulation list

        # list of available redshifts at Nyx (not all sims have them)
        self.list_sim_redshifts = np.append(
            np.arange(2.0, 4.5, 0.2), np.arange(4.6, 5.5, 0.4)
        )

        # get relevant flags for post-processing
        self._set_info_sim()

        # load power spectrum measurements
        self._load_data(file_name, zmax)

        # extract indexes from data
        self._set_labels()

    def _set_info_sim(self):
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
        self.training_z_max = 10
        # testing options
        self.testing_ind_rescaling = 0
        self.testing_z_max = 10

        self.key_conv = {
            "Fbar": "mF",
            "T_0": "T0",
            "gamma": "gamma",
            "tau_rescale_factor": "val_scaling",
            "lambda_P": "lambda_P",
        }
        # sigT_Mpc missing after gamma

        self.sim_conv = {
            "cosmo_grid_0": "nyx_0",
            "cosmo_grid_1": "nyx_1",
            "cosmo_grid_2": "nyx_2",
            "cosmo_grid_3": "nyx_3",
            "cosmo_grid_4": "nyx_4",
            "cosmo_grid_5": "nyx_5",
            "cosmo_grid_6": "nyx_6",
            "cosmo_grid_7": "nyx_7",
            "cosmo_grid_8": "nyx_8",
            "cosmo_grid_9": "nyx_9",
            "cosmo_grid_10": "nyx_10",
            "cosmo_grid_11": "nyx_11",
            "cosmo_grid_12": "nyx_12",
            "cosmo_grid_13": "nyx_13",
            "cosmo_grid_14": "nyx_14",
            "fiducial": "nyx_central",
            "bar_ic_grid_3": "nyx_seed",
            "wdm_3.5kev_grid_1": "nyx_wdm",
        }

        self.z_conv = {
            "redshift_2.0": 0,
            "redshift_2.2": 1,
            "redshift_2.4": 2,
            "redshift_2.6": 3,
            "redshift_2.8": 4,
            "redshift_3.0": 5,
            "redshift_3.2": 6,
            "redshift_3.4": 7,
            "redshift_3.6": 8,
            "redshift_3.8": 9,
            "redshift_4.0": 10,
            "redshift_4.2": 11,
            "redshift_4.4": 12,
            "redshift_4.6": 13,
            "redshift_5.0": 14,
            "redshift_5.4": 15,
        }

        self.scaling_conv = {
            "native_parameters": 0,
            "rescale_Fbar_fiducial": 1,
            "thermal_grid_0": 2,
            "thermal_grid_1": 3,
            "thermal_grid_2": 4,
            "thermal_grid_3": 5,
            "thermal_grid_4": 6,
            "thermal_grid_5": 7,
            "thermal_grid_6": 8,
            "thermal_grid_7": 9,
            "thermal_grid_8": 10,
            "thermal_grid_9": 11,
            "thermal_grid_10": 12,
            "thermal_grid_11": 13,
            "thermal_grid_12": 14,
            "thermal_grid_13": 15,
            "thermal_grid_14": 16,
            "new_thermal_lhd_0": 17,
            "new_thermal_lhd_1": 18,
            "new_thermal_lhd_2": 19,
            "new_thermal_lhd_3": 20,
            "new_thermal_lhd_4": 21,
        }

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
            "lambda_P",
        ]

    def _load_data(self, file_name, zmax):
        # if file_name was not provided, search for default one
        if file_name is None:
            if "NYX_PATH" not in os.environ:
                raise ValueError("Define NYX_PATH variable or pass file_name")
            file_name = os.environ["NYX_PATH"] + "/models.hdf5"

        # read data
        ff = h5py.File(file_name, "r")

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

            # setup CAMB object from sim_params
            sim_params = get_attrs(ff[isim])
            if isim == "fiducial":
                sim_params["A_s"] = 2.10e-9
                sim_params["n_s"] = 0.966
                sim_params["h"] = sim_params["H_0"] / 100
            sim_cosmo = camb_cosmo.get_Nyx_cosmology(sim_params)

            # compute linear power parameters at each z (will call CAMB)
            linP_zs = fit_linP.get_linP_Mpc_zs(
                sim_cosmo, self.list_sim_redshifts, self.kp_Mpc
            )
            # compute conversion from Mpc to km/s using cosmology
            dkms_dMpc_zs = camb_cosmo.dkms_dMpc(
                sim_cosmo, z=self.list_sim_redshifts
            )

            # loop over redshifts
            z_avail = list(ff[isim].keys())
            for iz in z_avail:
                zval = np.float(split_string(iz)[1])

                if zmax:
                    if zval > zmax:
                        if self.verbose:
                            print("do not read snapshot at", zval)
                        continue

                # find redshift index to read linear power parameters
                ind_z = np.where(
                    np.isclose(self.list_sim_redshifts, zval, 1e-10)
                )[0][0]
                linP_iz = linP_zs[ind_z]
                dkms_dMpc_iz = dkms_dMpc_zs[ind_z]

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

                        _arch["cosmo_pars"] = sim_params

                        # set linp params
                        for key in linP_iz.keys():
                            _arch[key] = linP_iz[key]
                        _arch["z"] = zval

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
                        _arch["sigT_Mpc"] = sigma_T_kms / dkms_dMpc_iz

                        # store pressure parameters
                        # not available in bar_ic_grid_3 and wdm_3.5kev_grid_1
                        # should it be the same as nyx_3 and nyx_1?
                        _pressure = get_attrs(ff[isim][iz])
                        key = "lambda_P"
                        if key in _pressure.keys():
                            _arch[self.key_conv[key]] = _pressure[key]
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
