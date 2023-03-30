import numpy as np
import copy
import sys
import os
import json
from itertools import product
import matplotlib.pyplot as plt
from lace_manager.setup_simulations import read_gadget
from lace.setup_simulations import read_genic
from lace.cosmo import camb_cosmo
from lace.cosmo import fit_linP


class archivePD(object):
    """Book-keeping of flux P1D & P3D measurements from a suite of simulations."""

    def __init__(
        self,
        basedir="/lace/emulator/sim_suites/post_768/",
        skewers_label="Ns768_wM0.05",
        p1d_label=None,
        max_archive_size=None,
        undersample_z=1,
        verbose=False,
        no_skewers=False,
        pick_sim=None,
        drop_sim=None,
        z_max=5.0,
        nsamples=None,
        undersample_cube=1,
        kp_Mpc=None,
        multiple_axes=True,
        n_phases=2,
        old_params=True,
    ):
        """Load archive from base sim directory and (optional) label
        identifying skewer configuration (number, width).
        If kp_Mpc is specified, recompute linP params in archive"""

        # SHOULD UPDATE DOCSTRING WITH ALL THESE ARGUMENTS

        assert "LACE_REPO" in os.environ, "export LACE_REPO"
        repo = os.environ["LACE_REPO"] + "/"

        self.basedir = basedir
        self.fulldir = repo + basedir
        if p1d_label:
            self.p1d_label = p1d_label
        else:
            self.p1d_label = "p1d"
        if skewers_label:
            self.skewers_label = skewers_label
        else:
            self.skewers_label = "Ns768_wM0.05"
        self.verbose = verbose
        self.z_max = z_max
        self.undersample_cube = undersample_cube
        self.drop_sim = drop_sim
        # pivot point used in linP parameters
        self.kp_Mpc = kp_Mpc
        self.n_phases = n_phases
        self.multiple_axes = multiple_axes
        self.old_params = old_params

        if self.old_params == False:
            self.fulldir_params = self.fulldir
        else:
            self.fulldir_params = repo + "/lace/emulator/sim_suites/Australia20/"

        if multiple_axes == False:
            self.n_axes = 1
        else:
            self.n_axes = 3

        self._load_data(
            max_archive_size,
            undersample_z,
            no_skewers,
            pick_sim,
            self.drop_sim,
            z_max,
            undersample_cube,
            nsamples,
        )

        return

    def _load_data(
        self,
        max_archive_size,
        undersample_z,
        no_skewers,
        pick_sim,
        drop_sim,
        z_max,
        undersample_cube,
        nsamples=None,
    ):
        """Setup archive by looking at all measured power spectra in sims"""

        # each measured power will have a dictionary, stored here
        # we store the power of each phase and axis separately
        self.data = []

        # keys in output dictionary
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
        if self.basedir != "lace/emulator/sim_suites/Australia20/":
            keys_copy_in.append("p3d_data")
            keys_copy_out.append("k3_Mpc")
            keys_copy_out.append("mu3")
            keys_copy_out.append("p3d_Mpc")

        # read file containing information about latin hyper-cube
        cube_json = self.fulldir + "/latin_hypercube.json"
        with open(cube_json) as json_file:
            self.cube_data = json.load(json_file)
        if self.verbose:
            print("latin hyper-cube data", self.cube_data)
        if nsamples is None:
            self.nsamples = self.cube_data["nsamples"]
        else:
            self.nsamples = nsamples
        if self.verbose:
            print("simulation suite has %d samples" % self.nsamples)

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

        if pick_sim is not None:
            if np.issubdtype(type(pick_sim), np.integer):
                start = pick_sim
                self.nsamples = pick_sim_number + 1
            else:
                start = 0
                self.nsamples = 1
        else:
            start = 0

        # read info from all sims, all snapshots, all rescalings
        for sample in range(start, self.nsamples, undersample_cube):
            if sample is drop_sim:
                continue
            # store parameters for simulation pair / model
            sim_params = self.cube_data["samples"]["%d" % sample]
            if self.verbose:
                print(sample, "sample has sim params =", sim_params)

            # read number of snapshots (should be the same in all sims)
            ind_sim = sample
            if pick_sim is not None:
                if np.issubdtype(type(pick_sim), np.integer) == False:
                    if pick_sim == "h":
                        tag_sample = "sim_pair_h"
                        ind_sim = 100
                    elif pick_sim == "nu":
                        tag_sample = "nu_sim"
                        ind_sim = 101
                    else:
                        raise ValueError("pick_sim must be a number, h or nu")
            elif sample < 30:
                tag_sample = "sim_pair_" + str(sample)
            else:
                tag_sample = "sim_pair_" + str(sample)
                tag_sample_p = "central"

            if (tag_sample == "sim_pair_h") | (tag_sample == "nu_sim"):
                # read zs values
                pair_dir = self.fulldir_params + "/" + tag_sample
                file = pair_dir + "/sim_plus/paramfile.gadget"
                sim_config = read_gadget.read_gadget_paramfile(file)
                zs = read_gadget.snapshot_redshifts(sim_config)
                Nz = len(zs)
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
                if sample < 30:
                    pair_dir = self.fulldir_params + "/" + tag_sample
                    pair_json = pair_dir + "/parameter.json"
                else:
                    pair_dir = self.fulldir_params + "/" + tag_sample_p
                    pair_json = pair_dir + "/parameter_redundant.json"

                with open(pair_json) as json_file:
                    pair_data = json.load(json_file)
                zs = pair_data["zs"]
                Nz = len(zs)
                linP_zs = pair_data["linP_zs"]
                if self.verbose:
                    print("simulation has %d redshifts" % Nz)
                    print("undersample_z =", undersample_z)

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

            # to make lighter emulators, we might undersample redshifts
            for snap in range(0, Nz, undersample_z):
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

                # make sure that we have skewers for this snapshot (z < zmax)
                for ind_axis in range(self.n_axes):
                    if self.basedir == "lace/emulator/sim_suites/Australia20/":
                        temp_skewers_label = self.skewers_label
                    else:
                        temp_skewers_label = (
                            self.skewers_label + "_axis" + str(ind_axis + 1)
                        )

                    # check if we have extracted skewers yet
                    if no_skewers:
                        self.data.append(snap_p1d_data)
                        continue

                    # we get parameters from Chris' post-processing (ii = 1)
                    for ii in range(2):
                        if ii == 0:
                            pair_dir = self.fulldir + "/" + tag_sample
                            p1d_label = self.p1d_label
                            _temp_skewers_label = temp_skewers_label
                        else:
                            pair_dir = self.fulldir_params + "/" + tag_sample
                            if self.old_params == False:
                                p1d_label = self.p1d_label
                                _temp_skewers_label = temp_skewers_label
                            else:
                                if sample == 30:
                                    pair_dir = self.fulldir_params + "/" + tag_sample_p

                                p1d_label = "p1d"
                                _temp_skewers_label = "Ns500_wM0.05"

                        # open sim_plus and sim_minus (P1D + P3D & params)
                        for jj in range(self.n_phases):
                            if jj == 0:
                                _phase = "/sim_plus/"
                            else:
                                _phase = "/sim_minus/"

                            phase_p1d_json = (
                                pair_dir
                                + _phase
                                + p1d_label
                                + "_"
                                + str(snap)
                                + "_"
                                + _temp_skewers_label
                                + ".json"
                            )
                            # print(phase_p1d_json)
                            if not os.path.isfile(phase_p1d_json):
                                if self.verbose:
                                    print(phase_p1d_json, "snapshot does not have p1d")
                                continue
                            # open file with 1D and 3D power measured in snapshot for sim_plus
                            with open(phase_p1d_json) as json_file:
                                _data = json.load(json_file)

                            if (ii == 0) & (jj == 0):
                                plus_data = copy.deepcopy(_data)
                            elif (ii == 0) & (jj == 1):
                                minus_data = copy.deepcopy(_data)
                            elif (ii == 1) & (jj == 0):
                                plus_param = copy.deepcopy(_data)
                            elif (ii == 1) & (jj == 1):
                                minus_param = copy.deepcopy(_data)

                    # number of post-process rescalings for each snapshot
                    Npp = len(plus_data["p1d_data"])
                    # read info for each post-process
                    for pp in range(Npp):
                        # check if both phases use the same kbins
                        _flag = np.allclose(
                            plus_data["p1d_data"][pp]["k_Mpc"],
                            minus_data["p1d_data"][pp]["k_Mpc"],
                        )
                        if _flag == False:
                            print(tag_sample, snap, pp)
                            raise ValueError("different k_Mpc in minus/plus")

                        # iterate over phases
                        for ind_phase in range(self.n_phases):
                            if ind_phase == 0:
                                temp_data = plus_data["p1d_data"][pp]
                                temp_param = plus_param["p1d_data"][pp]
                            else:
                                temp_data = minus_data["p1d_data"][pp]
                                temp_param = minus_param["p1d_data"][pp]

                            # deep copy of dictionary (thread safe, why not)
                            p1d_data = json.loads(json.dumps(snap_p1d_data))
                            # identify simulation
                            p1d_data["ind_sim"] = ind_sim
                            p1d_data["ind_z"] = snap
                            p1d_data["ind_phase"] = ind_phase
                            p1d_data["ind_axis"] = ind_axis
                            p1d_data["ind_tau"] = pp

                            # iterate over properties
                            for ind_key in range(len(keys_copy_in)):
                                key_in = keys_copy_in[ind_key]
                                key_out = keys_copy_out[ind_key]
                                if (
                                    (key_in == "sim_scale_T0")
                                    | (key_in == "sim_scale_gamma")
                                    | (key_in == "sim_scale_tau")
                                ):
                                    if key_in in temp_data:
                                        p1d_data[key_out] = temp_data[key_in]
                                elif key_in == "p3d_data":
                                    # all axes
                                    p1d_data["p3d_Mpc"] = np.array(
                                        temp_data["p3d_data"]["p3d_Mpc"]
                                    )
                                    p1d_data["k3_Mpc"] = np.array(
                                        temp_data["p3d_data"]["k_Mpc"]
                                    )
                                    p1d_data["mu3"] = np.array(
                                        temp_data["p3d_data"]["mu"]
                                    )
                                elif (key_in == "p1d_Mpc") | (key_in == "k_Mpc"):
                                    p1d_data[key_out] = np.array(temp_data[key_in])
                                else:
                                    p1d_data[key_out] = temp_param[key_in]

                            self.data.append(p1d_data)

        if max_archive_size is not None:
            Ndata = len(self.data)
            if Ndata > max_archive_size:
                if self.verbose:
                    print("will keep only", max_archive_size, "entries")
                keep = np.random.randint(0, Ndata, max_archive_size)
                keep_data = [self.data[i] for i in keep]
                self.data = keep_data

        N = len(self.data)
        if self.verbose:
            print("archive setup, containing %d entries" % len(self.data))

        # create 1D arrays with all entries for a given parameter
        self._store_param_arrays()

        return

    def _store_param_arrays(self):
        """create 1D arrays with all entries for a given parameter."""

        list_keys = [
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
            "mF",
            "sigT_Mpc",
            "gamma",
            "kF_Mpc",
            "k_Mpc",
            "p1d_Mpc",
            "k3_Mpc",
            "mu3",
            "p3d_Mpc",
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
                & (key != "k3_Mpc")
                & (key != "mu3")
                & (key != "p3d_Mpc")
            ):
                _dict[key] = np.zeros(N)
            else:
                _dict[key] = np.zeros((N, *self.data[0][key].shape))

            for ii in range(N):
                if (
                    (key != "k_Mpc")
                    & (key != "p1d_Mpc")
                    & (key != "k3_Mpc")
                    & (key != "mu3")
                    & (key != "p3d_Mpc")
                ):
                    _dict[key][ii] = self.data[ii][key]
                else:
                    _dict[key][ii] = self.data[ii][key][: _dict[key][ii].shape[0]]

        for key in _dict.keys():
            setattr(self, key, _dict[key])

    def average_over_samples(self, flag="all"):
        """Compute averages over either phases, axes, or all

        Compute average of p1d, p3d, and mean flux
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
        if self.basedir != "lace/emulator/sim_suites/Australia20/":
            keys_merge.append("k3_Mpc")
            keys_merge.append("mu3")
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

    def print_entry(
        self,
        entry,
        keys=[
            "z",
            "Delta2_p",
            "n_p",
            "alpha_p",
            "f_p",
            "mF",
            "sigT_Mpc",
            "gamma",
            "kF_Mpc",
        ],
    ):
        """Print basic information about a particular entry in the archive"""

        if entry >= len(self.data):
            raise ValueError("{} entry does not exist in archive".format(entry))

        data = self.data[entry]
        info = "entry = {}".format(entry)
        for key in keys:
            info += ", {} = {:.4f}".format(key, data[key])
        print(info)
        return

    def plot_samples(self, param_1, param_2, tau_scalings=True, temp_scalings=True):
        """For parameter pair (param1,param2), plot each point in the archive"""

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

        return

    def plot_3D_samples(
        self, param_1, param_2, param_3, tau_scalings=True, temp_scalings=True
    ):
        """For parameter trio (param1,param2,param3), plot each point in the archive"""

        from mpl_toolkits import mplot3d

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

        return

    def sub_archive_mf(self, min_mf=0.0, max_mf=1.0):
        """Return copy of archive, with entries in a given mean flux range."""

        # make copy of archive
        copy_archive = copy.deepcopy(self)
        copy_archive.min_mf = min_mf
        copy_archive.max_mf = max_mf

        print(len(copy_archive.data), "initial entries")

        # select entries in a given mean flux range
        new_data = [
            d for d in copy_archive.data if (d["mF"] < max_mf and d["mF"] > min_mf)
        ]

        if self.verbose:
            print("use %d/%d entries" % (len(new_data), len(self.data)))

        # store new sub-data
        copy_archive.data = new_data

        # re-create 1D arrays with all entries for a given parameter
        copy_archive._store_param_arrays()

        return copy_archive

    def get_param_values(self, param, tau_scalings=True, temp_scalings=True):
        """Return values for a given parameter, including rescalings or not."""

        N = len(self.data)
        # mask post-process scalings (optional)
        if not tau_scalings:
            mask_tau = [x["scale_tau"] == 1.0 for x in self.data]
        else:
            mask_tau = [True] * N
        if not temp_scalings:
            mask_temp = [
                (x["scale_T0"] == 1.0) & (x["scale_gamma"] == 1.0) for x in self.data
            ]
        else:
            mask_temp = [True] * N

        # figure out values of param in archive
        values = np.array(
            [self.data[i][param] for i in range(N) if (mask_tau[i] & mask_temp[i])]
        )

        return values

    def get_simulation_cosmology(self, sim_num):
        """Get cosmology used in a given simulation in suite"""

        # setup cosmology from GenIC file
        dir_name = self.fulldir + "/sim_pair_" + str(sim_num)
        file_name = dir_name + "/sim_plus/paramfile.genic"
        sim_cosmo_dict = read_genic.camb_from_genic(file_name)
        sim_cosmo = camb_cosmo.get_cosmology_from_dictionary(sim_cosmo_dict)

        return sim_cosmo
