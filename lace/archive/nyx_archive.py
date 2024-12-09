import numpy as np
import os
import h5py

from lace.cosmo.thermal_broadening import thermal_broadening_kms
from lace.archive.base_archive import BaseArchive
from lace.utils.misc import split_string
from lace.archive.constants import (
    LIST_SIM_NYX_TEST,
    LIST_SIM_NYX_CUBE,
    LIST_ALL_SIMS_NYX,
    LIST_SIM_REDSHIFTS_NYX
)


class NyxArchive(BaseArchive):
    """Bookkeeping of Lya flux P1D & P3D measurements from a suite of Nyx simulations."""

    def __init__(
        self,
        nyx_version: str = "Oct2023",
        nyx_file: str | None = None,
        include_central: bool = False,
        kp_Mpc: int | float | None = None,
        force_recompute_linP_params: bool = False,
        verbose: bool = False,
        nfiles: int = 18,
        z_star: float = 3,
        kp_kms: float = 0.009,
    ) -> None:
        # Check input
        self.nyx_version = nyx_version

        if force_recompute_linP_params and kp_Mpc is None:
            raise TypeError("kp_Mpc must be a number if force_recompute_linP_params == True")

        self.kp_Mpc = kp_Mpc
        self.z_star = z_star
        self.kp_kms = kp_kms
        self.verbose = verbose

        # Set list of available simulations
        self.list_sim_test = LIST_SIM_NYX_TEST
        self.list_sim_cube = LIST_SIM_NYX_CUBE

        if include_central:
            self.list_sim_cube.append("nyx_central")
            self.list_sim_test.remove("nyx_central")

        self.list_sim = LIST_ALL_SIMS_NYX
        self.list_sim_axes = [0, 1, 2]

        # Initialize simulation info and load data
        self._set_info_sim(nfiles)
        self._load_data(nyx_file, force_recompute_linP_params)
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
        self.scalings_avail = list(range(22))
        # training options
        self.training_average = "both"
        self.training_val_scaling = "all"
        self.training_z_min = 0
        self.training_z_max = 10
        
        # Testing parameters
        self.testing_ind_rescaling = 0
        self.testing_z_min = 0
        self.testing_z_max = 10

        # Key name conversions
        self.key_conv = {
            "Fbar": "mF",
            "T_0": "T0", 
            "gamma": "gamma",
            "tau_rescale_factor": "val_scaling",
            "lambda_P": "kF_Mpc",
        }

        # Simulation name conversions
        self.sim_conv = {
            "fiducial": "nyx_central",
            "bar_ic_grid_3": "nyx_3_ic",
            "CGAN_4096_base": "nyx_seed",
            "wdm_3.5kev_grid_1": "nyx_wdm",
        }
        for ii in range(nfiles):
            self.sim_conv[f"cosmo_grid_{ii}"] = f"nyx_{ii}"

        # list of available redshifts at Nyx (not all sims have them)
        self.list_sim_redshifts = LIST_SIM_REDSHIFTS_NYX

        # conversion between nyx and lace snapshot names
        self.z_conv = {
            f"redshift_{z}": i 
            for i, z in enumerate(self.list_sim_redshifts)
        }

        # conversion between scaling names
        self.scaling_conv = {
            "native_parameters": 0,
            "rescale_Fbar_fiducial": 1,
        }
        self.scaling_conv.update({
            f"thermal_grid_{i}": i + 2 
            for i in range(15)
        })
        self.scaling_conv.update({
            f"new_thermal_lhd_{i}": i + 17 
            for i in range(5)
        })

        # Axis name conversions
        self.axis_conv = {"x": 0, "y": 1, "z": 2}

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

    def _get_attrs(self, h5py_data):
        return dict(h5py_data.attrs)

    def _get_emu_cosmo(
        self, nyx_data, sim_label, force_recompute_linP_params=False
    ):
        """Get cosmology and linear power spectrum parameters from simulation.

        Args:
            nyx_data: File containing Nyx data
            sim_label: Selected simulation
            force_recompute_linP_params: Compute linP even if kp_Mpc matches

        Returns:
            tuple: (cosmo_params, linP_params, star_params) containing:
                - cosmo_params (dict): Cosmological parameters
                - linP_params (dict): Linear power spectrum parameters
                - star_params (dict): Star parameters
        """
        isim = self.sim_conv[sim_label]
        compute_linP_params = force_recompute_linP_params

        if not compute_linP_params:
            try:
                file_cosmo = np.load(self.file_cosmo, allow_pickle=True).item()
            except:
                raise IOError(f"The file {self.file_cosmo} does not exist")

            if isim not in file_cosmo:
                msg = (
                    f"The file {self.file_cosmo} does not contain {isim}. "
                    "To speed up calculations, you can recompute the file by running "
                    "lace/scripts/developers/compute_nyx_emu_cosmo.py"
                )
                if self.verbose:
                    print(msg)
                compute_linP_params = True
            else:
                # if kp_Mpc not defined, use precomputed value
                if self.kp_Mpc is None:
                    self.kp_Mpc = file_cosmo[isim]["linP_params"]["kp_Mpc"]

                # if kp_Mpc different from precomputed value, compute
                if self.kp_Mpc != file_cosmo[isim]["linP_params"]["kp_Mpc"]:
                    if self.verbose:
                        print(f"Recomputing kp_Mpc at {self.kp_Mpc}")
                    compute_linP_params = True
                else:
                    cosmo_params = file_cosmo[isim]["cosmo_params"]
                    linP_params = file_cosmo[isim]["linP_params"]
                    star_params = file_cosmo[isim]["star_params"]

        if compute_linP_params:
            if self.verbose:
                print("We are using CAMB")
            from lace.cosmo import camb_cosmo, fit_linP

            cosmo_params = self._get_attrs(nyx_data[sim_label])
            
            if isim == "nyx_central":
                cosmo_params.update({
                    "A_s": 2.10e-9,
                    "n_s": 0.966,
                    "h": cosmo_params["H_0"] / 100
                })
            elif isim == "nyx_seed":
                cosmo_params.update({
                    "A_s": 2.15865e-9,
                    "n_s": 0.96,
                    "h": cosmo_params["H_0"] / 100
                })

            sim_cosmo = camb_cosmo.get_Nyx_cosmology(cosmo_params)
            linP_zs = fit_linP.get_linP_Mpc_zs(
                sim_cosmo, self.list_sim_redshifts, self.kp_Mpc
            )
            star_params = fit_linP.parameterize_cosmology_kms(
                sim_cosmo, None, self.z_star, self.kp_kms
            )
            
            zs = np.array(self.list_sim_redshifts)
            dkms_dMpc_zs = camb_cosmo.dkms_dMpc(sim_cosmo, z=zs)

            linP_params = {
                "kp_Mpc": self.kp_Mpc,
                "z": zs,
                "dkms_dMpc": dkms_dMpc_zs
            }
            
            for lab in ["Delta2_p", "n_p", "alpha_p", "f_p"]:
                linP_params[lab] = np.array([linP_zs[i][lab] for i in range(len(zs))])

        return cosmo_params, linP_params, star_params

    def _load_data(self, nyx_file=None, force_recompute_linP_params=False):
        if nyx_file is None:
            if "NYX_PATH" not in os.environ:
                raise ValueError(
                    "If nyx_file is not provided, you must define NYX_PATH "
                    "environment variable pointing to the Nyx data folder"
                )
            nyx_file = os.path.join(
                os.environ["NYX_PATH"],
                f"models_Nyx_{self.nyx_version}.hdf5"
            )

        try:
            ff = h5py.File(nyx_file, "r")
        except Exception as e:
            ff.close()
            raise e

        folder = os.environ.get("NYX_PATH", os.path.dirname(nyx_file))
        self.file_cosmo = os.path.join(
            folder, f"nyx_emu_cosmo_{self.nyx_version}.npy"
        )

        # store each measurement as an entry of the following list
        # each entry is a dictionary containing all relevant info
        self.data = []
        # loop over simulations. always check if avail because there are some
        # configurations missing here and there
        sim_avail = list(ff.keys())
        
        for isim in sim_avail:
            # Skip problematic simulations
            if isim in ["cosmo_grid_14", "CGAN_4096_val"]:
                continue

            if self.verbose:
                print(f"Reading Nyx sim {isim}")

            # read cosmo information and linear power parameters
            # (will only need CAMB if pivot point kp_Mpc is changed)
            cosmo_params, linP_params, star_params = self._get_emu_cosmo(
                ff, isim, force_recompute_linP_params
            )

            for iz in ff[isim].keys():
                zval = float(split_string(iz)[1])

                # find redshift index to read linear power parameters
                try:
                    ind_z = np.where(np.isclose(self.list_sim_redshifts, zval, 1e-10))[0][0]
                except IndexError:
                    print(f"Could not find redshift {zval} in {isim}")
                    continue

                for iscaling in ff[isim][iz].keys():
                    if iscaling == "full_box_stats":
                        continue

                    for iaxis in ff[isim][iz][iscaling]["individual_axes"].keys():
                        _arch = {
                            "cosmo_params": cosmo_params,
                            "star_params": star_params,
                            "sim_label": self.sim_conv[isim],
                            "ind_snap": self.z_conv[iz],
                            "ind_phase": 0,
                            "ind_axis": self.axis_conv[iaxis],
                            "ind_rescaling": self.scaling_conv[iscaling]
                        }

                        # Add linear power parameters
                        for lab, val in linP_params.items():
                            _arch[lab] = val[ind_z] if lab != "kp_Mpc" else val

                        # Add IGM parameters
                        _igm = self._get_attrs(ff[isim][iz][iscaling])
                        for key, new_key in self.key_conv.items():
                            if key in _igm:
                                _arch[new_key] = _igm[key]

                        # compute thermal broadening in Mpc
                        sigma_T_kms = thermal_broadening_kms(_arch["T0"])
                        _arch["sigT_Mpc"] = sigma_T_kms / linP_params["dkms_dMpc"][ind_z]

                        # store pressure parameters
                        # not available in bar_ic_grid_3 and wdm_3.5kev_grid_1
                        # should it be the same as nyx_3 and nyx_1?
                        _pressure = self._get_attrs(ff[isim][iz])
                        if "lambda_P" in _pressure:
                            _arch["kF_Mpc"] = 1 / (1e-3 * _pressure["lambda_P"])

                        ## store P1D and P3D measurements (if available)
                        _data = ff[isim][iz][iscaling]["individual_axes"][iaxis]
                        p1d = np.array(_data["1d power"])
                        _arch.update({
                            "k_Mpc": p1d["k"],
                            "p1d_Mpc": p1d["Pk1d"]
                        })

                        # Add P3D if available
                        if "3d power kmu" in _data:
                            if "fine binning" in _data["3d power kmu"]:
                                p3d = np.array(_data["3d power kmu"]["fine binning"])
                                _arch.update({
                                    "k3d_Mpc": p3d["k"].reshape(self.kbins, self.mubins),
                                    "mu3d": p3d["mu"].reshape(self.kbins, self.mubins),
                                    "p3d_Mpc": p3d["Pkmu"].reshape(self.kbins, self.mubins)
                                })

                        self.data.append(_arch)
        ff.close()
