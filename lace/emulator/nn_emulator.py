import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import copy
import random
import time
from warnings import warn

# Torch related modules
import torch
from torch.utils.data import DataLoader, dataset, TensorDataset
from torch import nn, optim
from torch.optim import lr_scheduler

# LaCE modules
import lace
from lace.archive import gadget_archive, nyx_archive
from lace.emulator import nn_architecture, base_emulator
from lace.utils import poly_p1d

from scipy.spatial import Delaunay


class NNEmulator(base_emulator.BaseEmulator):
    """A class for training an emulator.

    Args:
        archive (class): Data archive used for training the emulator.
            Required when using a custom emulator.
        training_set: Specific training set.  Options are
            'Cabayol23'.
        emu_params (lsit): A list of emulator parameters.
        emulator_label (str): Specific emulator label. Options are
            'Cabayol23' and 'Nyx_vo'.
        kmax_Mpc (float): The maximum k in Mpc^-1 to use for training. Default is 3.5.
        nepochs (int): The number of epochs to train for. Default is 200.
        model_path (str): The path to a pretrained model. Default is None.
        train (bool): Wheather to train the emulator or not. Default True. If False, a model path must is required.
    """

    def __init__(
        self,
        archive=None,
        training_set=None,
        emu_params=["Delta2_p", "n_p", "mF", "sigT_Mpc", "gamma", "kF_Mpc"],
        emulator_label=None,
        kmax_Mpc=4,
        ndeg=5,
        nepochs=100,
        step_size=75,
        drop_sim=None,
        drop_z=None,
        train=True,
        save_path=None,
        model_path=None,
        weighted_emulator=True,
        nhidden=5,
        max_neurons=50,
        seed=32,
        fprint=print,
        lr0=1e-3,
        batch_size=100,
        weight_decay=1e-4
    ):
        # store emulator settings
        self.emulator_label = emulator_label
        self.training_set = training_set

        self.emu_params = emu_params
        self.kmax_Mpc = kmax_Mpc
        self.ndeg = ndeg
        self.nepochs = nepochs
        self.step_size = step_size
        # paths to save/load models
        self.save_path = save_path
        self.model_path = model_path
        repo = os.path.dirname(lace.__path__[0]) + "/"
        self.models_dir = os.path.join(repo, "data/")
        # CPU vs GPU
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # training data settings
        self.drop_sim = drop_sim
        self.drop_z = drop_z
        self.weighted_emulator = weighted_emulator
        self.nhidden = nhidden
        self.print = fprint
        self.lr0 = lr0
        self.max_neurons = max_neurons
        self.batch_size=batch_size
        self.weight_decay=weight_decay
        

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # check input #
        training_set_all = ["Pedersen21", "Cabayol23", "Nyx23_Oct2023"]
        emulator_label_all = [
            "Cabayol23",
            "nn_test",
            "Nyx_v0",
            "Cabayol23_extended",
            "Nyx_v0_extended",
        ]

        # check input

        ## set training_set
        if (archive is None) & (training_set is None):
            raise ValueError("Archive or training_set must be provided")

        elif (training_set is not None) & (archive is None):
            if training_set not in training_set_all:
                raise ValueError(
                    f"Invalid training_set value {training_set}. Available options:",
                    training_set_all,
                )

            self.print(f"Selected training set {training_set}")

            if training_set in ["Pedersen21", "Cabayol23"]:
                archive = gadget_archive.GadgetArchive(postproc=training_set)
            elif training_set[:5] in ["Nyx23"]:
                archive = nyx_archive.NyxArchive(nyx_version=training_set[6:])

            self.training_data = archive.get_training_data(
                emu_params=self.emu_params, drop_sim=self.drop_sim
            )

        elif (training_set is None) & (archive is not None):
            self.print(
                "Use custom archive provided by the user to train emulator"
            )
            self.training_data = archive.get_training_data(
                emu_params=self.emu_params,
                drop_sim=self.drop_sim,
                drop_z=self.drop_z,
            )

        elif (training_set is not None) & (archive is not None):
            if train:
                raise ValueError(
                    "Provide either archive or training set for training"
                )
            else:
                self.print(
                    "Using custom archive provided by the user to load emulator"
                )
                self.training_data = archive.get_training_data(
                    emu_params=self.emu_params,
                    drop_sim=self.drop_sim,
                    drop_z=self.drop_z,
                )

        self.print(f"Samples in training_set: {len(self.training_data)}")
        self.kp_Mpc = archive.kp_Mpc

        ## check emulator label
        if emulator_label is not None:
            if emulator_label in emulator_label_all:
                self.print(f"Selected emulator {emulator_label}")
            else:
                raise ValueError(
                    f"Invalid emulator_label value {emulator_label}. Available options:",
                    emulator_label_all,
                )
        else:
            if train:
                self.print("Selected custom emulator")
            else:
                raise ValueError("Provide emulator_label when loading emulator")

        # define emulator settings
        if emulator_label == "Cabayol23":
            self.print(
                r"Neural network emulating the optimal P1D of Gadget simulations "
                + "fitting coefficients to a 5th degree polynomial. It "
                + "goes to scales of 4Mpc^{-1} and z<=4.5. The parameters "
                + "passed to the emulator will be overwritten to match "
                + "these ones"
            )
            self.emu_params = [
                "Delta2_p",
                "n_p",
                "mF",
                "sigT_Mpc",
                "gamma",
                "kF_Mpc",
            ]
            self.emu_type = "polyfit"
            (
                self.kmax_Mpc,
                self.ndeg,
                self.nepochs,
                self.step_size,
                self.nhidden,
                self.max_neurons,
                self.lr0,
            ) = (4, 5, 100, 75, 5, 100, 1e-3)

        elif emulator_label == "Nyx_v0":
            self.print(
                r"Neural network emulating the optimal P1D of Nyx simulations "
                + "fitting coefficients to a 6th degree polynomial. It "
                + "goes to scales of 4Mpc^{-1} and z<=4.5. The parameters "
                + "passed to the emulator will be overwritten to match "
                + "these ones"
            )
            self.emu_params = [
                "Delta2_p",
                "n_p",
                "mF",
                "sigT_Mpc",
                "gamma",
                "kF_Mpc",
            ]
            self.emu_type = "polyfit"
            (
                self.kmax_Mpc,
                self.ndeg,
                self.nepochs,
                self.step_size,
                self.nhidden,
                self.max_neurons,
                self.lr0,
            ) = (4, 6, 800, 700, 5, 150, 5e-5)

        elif emulator_label == "Cabayol23_extended":
            self.print(
                r"Neural network emulating the optimal P1D of Gadget simulations "
                + "fitting coefficients to a 7th degree polynomial. It "
                + "goes to scales of 8Mpc^{-1} and z<=4.5. The parameters "
                + "passed to the emulator will be overwritten to match "
                + "these ones. This configuration does not downweight the "
                + "contribution of small scales."
            )
            self.emu_params = [
                "Delta2_p",
                "n_p",
                "mF",
                "sigT_Mpc",
                "gamma",
                "kF_Mpc",
            ]
            self.emu_type = "polyfit"
            (
                self.kmax_Mpc,
                self.ndeg,
                self.nepochs,
                self.step_size,
                self.nhidden,
                self.max_neurons,
                self.weighted_emulator,
                self.lr0,
            ) = (8, 7, 100, 75, 5, 100, False, 1e-3)

        elif emulator_label == "Nyx_v0_extended":
            self.print(
                r"Neural network emulating the optimal P1D of Nyx simulations "
                + "fitting coefficients to a 7th degree polynomial. It "
                + "goes to scales of 8Mpc^{-1} and z<=4.5. The parameters "
                + "passed to the emulator will be overwritten to match "
                + "these ones"
            )
            self.emu_params = [
                "Delta2_p",
                "n_p",
                "mF",
                "sigT_Mpc",
                "gamma",
                "kF_Mpc",
            ]
            self.emu_type = "polyfit"
            (
                self.kmax_Mpc,
                self.ndeg,
                self.nepochs,
                self.step_size,
                self.nhidden,
                self.weighted_emulator,
            ) = (8, 7, 1000, 750, 5, True)

        # check consistency between training data an emulator label
        if (emulator_label == "Cabayol23") | (
            emulator_label == "Cabayol23_extended"
        ):
            # make sure that input archive / training data are Gadget sims
            if self.training_data[0]["sim_label"][:3] != "mpg":
                raise ValueError(
                    f"Training data for {emulator_label} are not Gadget sims"
                )
        elif (emulator_label == "Nyx_v0") | (
            emulator_label == "Nyx_v0_extended"
        ):
            # make sure that input archive / training data are Nyx sims
            if self.training_data[0]["sim_label"][:3] != "nyx":
                raise ValueError(
                    f"Training data for {emulator_label} are not Nyx sims"
                )

        self.kmin_Mpc = self.training_data[0]["k_Mpc"][1]

        # decide whether to train emulator or read from file
        if train == False:
            if self.model_path is None:
                raise ValueError("If train==False, model path is required.")

            pretrained_model = torch.load(
                os.path.join(self.models_dir, self.model_path),
                map_location="cpu",
            )
            self.nn = nn_architecture.MDNemulator_polyfit(
                nhidden=self.nhidden,
                ndeg=self.ndeg,
                max_neurons=self.max_neurons,
                ninput=len(self.emu_params),
            )
            self.nn.load_state_dict(pretrained_model["emulator"])
            self.nn.to(self.device)
            self.print("Model loaded. No training needed")

            model_metadata = pretrained_model["metadata"]

            # check consistency between required settings and read ones
            training_set_loaded = model_metadata["training_set"]
            emulator_label_loaded = model_metadata["emulator_label"]
            drop_sim_loaded = model_metadata["drop_sim"]
            drop_z_loaded = model_metadata["drop_z"]
            if drop_z_loaded is not None:
                drop_z_loaded = float(drop_z_loaded)

            if emulator_label_loaded != emulator_label:
                raise ValueError(
                    f"Emulator label mismatch: Expected '{emulator_label}' but loaded '{emulator_label_loaded}'"
                )

            if training_set_loaded != training_set:
                raise ValueError(
                    f"Training set mismatch: Expected '{training_set}' but loaded '{training_set_loaded}'"
                )

            if drop_sim_loaded != self.drop_sim:
                raise ValueError(
                    f"drop_sim mismatch: Expected '{self.drop_sim}' but loaded '{drop_sim_loaded}'"
                )

            if drop_z_loaded != self.drop_z:
                raise ValueError(
                    f"drop_z mismatch: Expected '{self.drop_z}' but loaded '{drop_z_loaded}'"
                )

            if model_metadata["drop_sim"] is not None and self.drop_sim is None:
                warn(
                    f"Model trained without simulation {emulator_settings['drop_sim']}"
                )

            if model_metadata["drop_z"] is not None and self.drop_z is None:
                warn(
                    f"Model trained without redshift {emulator_settings['drop_z']}"
                )

            kMpc_train = self._obtain_sim_params()
            log_kMpc_train = torch.log10(kMpc_train).to(self.device)
            self.log_kMpc = log_kMpc_train

        else:
            self.train()

            if self.save_path is not None:
                # saves the model in the predefined path after training
                self.save_emulator()

    def _sort_dict(self, dct, keys):
        """
        Sort a list of dictionaries based on specified keys.

        Args:
            dct (list): List of dictionaries to be sorted.
            keys (list): List of keys to sort the dictionaries by.

        Returns:
            list: The sorted list of dictionaries.
        """
        for d in dct:
            sorted_d = {
                k: d[k] for k in keys
            }  # create a new dictionary with only the specified keys
            d.clear()  # remove all items from the original dictionary
            d.update(
                sorted_d
            )  # update the original dictionary with the sorted dictionary
        return dct

    def _obtain_sim_params(self):
        """
        Obtain simulation parameters.

        Returns:
            k_Mpc (np.ndarray): Simulation k values.
            Nz (int): Number of redshift values.
            Nk (int): Number of k values.
            k_Mpc_train (tensor): k values in the k training range
        """

        self.k_mask = [
            (self.training_data[i]["k_Mpc"] < self.kmax_Mpc)
            & (self.training_data[i]["k_Mpc"] > 0)
            for i in range(len(self.training_data))
        ]

        k_Mpc_train = self.training_data[0]["k_Mpc"][self.k_mask[0]]
        k_Mpc_train = torch.Tensor(k_Mpc_train)
        self.k_Mpc = k_Mpc_train
        Nk = len(k_Mpc_train)
        self.Nk = Nk

        data = [
            {
                key: value
                for key, value in self.training_data[i].items()
                if key in self.emu_params
            }
            for i in range(len(self.training_data))
        ]

        # Now, if 'A_UVB' is in emu_params, add it to each entry
        if "A_UVB" in self.emu_params:
            for i, d in enumerate(data):
                d["A_UVB"] = self.training_data[i]["cosmo_params"]["A_UVB"]

        data = self._sort_dict(
            data, self.emu_params
        )  # sort the data by emulator parameters
        data = [list(data[i].values()) for i in range(len(self.training_data))]
        data = np.array(data)

        self.paramLims = np.concatenate(
            (
                data.min(0).reshape(len(data.min(0)), 1),
                data.max(0).reshape(len(data.max(0)), 1),
            ),
            1,
        )

        training_label = [
            {
                key: value
                for key, value in self.training_data[i].items()
                if key in ["p1d_Mpc"]
            }
            for i in range(len(self.training_data))
        ]

        training_label = [
            training_label[i]["p1d_Mpc"][self.k_mask[i]].tolist()
            for i in range(len(self.training_data))
        ]
        training_label = np.array(training_label)
        self.yscalings = np.median(training_label)

        return k_Mpc_train

    def _get_training_data_nn(self):
        """
        Given an archive and key_av, it obtains the training data based on self.emu_params
        Sorts the training data according to self.emu_params and scales the data based on self.paramLims
        Finally, it returns the training data as a torch.Tensor object.
        """
        training_data = [
            {
                key: value
                for key, value in self.training_data[i].items()
                if key in self.emu_params
            }
            for i in range(len(self.training_data))
        ]

        # Now, if 'A_UVB' is in emu_params, add it to each entry
        if "A_UVB" in self.emu_params:
            for i, data in enumerate(training_data):
                data["A_UVB"] = self.training_data[i]["cosmo_params"]["A_UVB"]

        training_data = self._sort_dict(training_data, self.emu_params)
        training_data = [
            list(training_data[i].values())
            for i in range(len(self.training_data))
        ]

        training_data = np.array(training_data)
        training_data = (training_data - self.paramLims[:, 0]) / (
            self.paramLims[:, 1] - self.paramLims[:, 0]
        ) - 0.5
        training_data = torch.Tensor(training_data)

        return training_data

    def _get_training_pd1_nn(self):
        """
        Method to get the p1d_Mpc values from the training data in a format that the NN emulator can ingest
        It gets the P1D from the archive and scales it.
        Finally, it returns the scaled values as a torch.Tensor object along with the scaling factor.
        """
        training_label = [
            {
                key: value
                for key, value in self.training_data[i].items()
                if key in ["p1d_Mpc"]
            }
            for i in range(len(self.training_data))
        ]
        training_label = [
            list(training_label[i].values())[0][self.k_mask[i]].tolist()
            for i in range(len(self.training_data))
        ]

        training_label = np.array(training_label)
        # yscalings = np.median(training_label)
        training_label = np.log10(training_label / self.yscalings)
        training_label = torch.Tensor(training_label)

        # self.yscalings=yscalings

        return training_label  # , yscalings

    def _set_weights(self):
        """
        Method to set downscale the impact of small scales on the extended emulator setup.
        Studied by Emma Clarassó.
        """
        w = torch.ones(size=(self.Nk,))
        if (self.kmax_Mpc > 4) & (self.weighted_emulator == True):
            self.print("Exponential downweighting loss function at k>4")
            exponential_values = torch.linspace(
                0, 1.4, len(self.k_Mpc[self.k_Mpc > 4])
            )
            w[self.k_Mpc > 4] = torch.exp(-exponential_values)
        return w

    def train(self):
        """
        Trains the emulator with given key_list using the archive data.
        Args:
        key_list (list): List of keys to be used for training

        Returns:None
        """

        kMpc_train = self._obtain_sim_params()

        loss_function_weights = self._set_weights()
        loss_function_weights = loss_function_weights.to(self.device)

        log_kMpc_train = torch.log10(kMpc_train).to(self.device)

        self.log_kMpc = log_kMpc_train

        self.nn = nn_architecture.MDNemulator_polyfit(
            nhidden=self.nhidden,
            ndeg=self.ndeg,
            max_neurons=self.max_neurons,
            ninput=len(self.emu_params),
        )

        optimizer = optim.Adam(
            self.nn.parameters(), lr=self.lr0, weight_decay=self.weight_decay
        )  #
        scheduler = lr_scheduler.StepLR(optimizer, self.step_size, gamma=0.1)

        training_data = self._get_training_data_nn()
        training_label = self._get_training_pd1_nn()

        trainig_dataset = TensorDataset(training_data, training_label)
        loader_train = DataLoader(trainig_dataset, batch_size=self.batch_size, shuffle=True)

        self.nn.to(self.device)
        self.print(f"Training NN on {len(training_data)} points")
        t0 = time.time()

        for epoch in range(self.nepochs):
            for datain, p1D_true in loader_train:
                optimizer.zero_grad()

                coeffsPred, coeffs_logerr = self.nn(datain.to(self.device))  #
                coeffs_logerr = torch.clamp(coeffs_logerr, -10, 5)
                coeffserr = torch.exp(coeffs_logerr) ** 2

                powers = torch.arange(0, self.ndeg + 1, 1).to(self.device)
                P1Dpred = torch.sum(
                    coeffsPred[:, powers, None]
                    * (self.log_kMpc[None, :] ** powers[None, :, None]),
                    axis=1,
                )

                powers_err = torch.arange(0, self.ndeg * 2 + 1, 2).to(
                    self.device
                )
                P1Derr = torch.sqrt(
                    torch.sum(
                        coeffserr[:, powers, None]
                        * (self.log_kMpc[None, :] ** powers_err[None, :, None]),
                        axis=1,
                    )
                )

                P1Dlogerr = torch.log(P1Derr)

                log_prob = ((P1Dpred - p1D_true.to(self.device)) / P1Derr).pow(
                    2
                ) + 2 * P1Dlogerr  #

                log_prob = loss_function_weights[None, :] * log_prob

                loss = torch.nansum(log_prob, 1)
                loss = torch.nanmean(loss, 0)

                loss.backward()
                optimizer.step()

            scheduler.step()

        self.print(f"NN optimised in {time.time()-t0} seconds")

    def save_emulator(self):
        model_state_dict = self.nn.state_dict()

        # Define your metadata
        metadata = {
            "training_set": self.training_set,
            "emulator_label": self.emulator_label,
            "drop_sim": self.drop_sim,
            "drop_z": self.drop_z,
        }

        # Combine model_state_dict and metadata into a single dictionary
        model_with_metadata = {
            "emulator": model_state_dict,
            "metadata": metadata,
        }

        torch.save(model_with_metadata, self.save_path)

    def emulate_p1d_Mpc(self, model, k_Mpc, return_covar=False, z=None):
        """Emulates the p1d_Mpc at a given set of k_Mpc values"""

        if np.max(k_Mpc) > self.kmax_Mpc:
            warn(
                f"Some of the requested k's are higher than the maximum training value k={self.kmax_Mpc}",
            )
        elif np.min(k_Mpc) < self.kmin_Mpc:
            warn(
                f"Some of the requested k's are lower than the minimum training value k={self.kmin_Mpc}"
            )

        k_Mpc = torch.Tensor(k_Mpc)
        log_kMpc = torch.log10(k_Mpc).to(self.device)

        with torch.no_grad():
            emu_call = {}
            for param in self.emu_params:
                if param not in model:
                    raise (ValueError(param + " not in input model"))
                emu_call[param] = model[param]

            # emu_call = {k: emu_call[k] for k in self.emu_params}
            # emu_call = list(emu_call.values())
            # emu_call = np.array(emu_call)
            emu_call = [emu_call[param] for param in self.emu_params]

            emu_call = (emu_call - self.paramLims[:, 0]) / (
                self.paramLims[:, 1] - self.paramLims[:, 0]
            ) - 0.5
            emu_call = torch.Tensor(emu_call).unsqueeze(0)

            # ask emulator to emulate P1D (and its uncertainty)
            coeffsPred, coeffs_logerr = self.nn(emu_call.to(self.device))

            powers = torch.arange(0, self.ndeg + 1, 1).to(self.device)
            emu_p1d = torch.sum(
                coeffsPred[:, powers, None]
                * (log_kMpc[None, :] ** powers[None, :, None]),
                axis=1,
            )

            emu_p1d = emu_p1d.detach().cpu().numpy().flatten()

            emu_p1d = 10 ** (emu_p1d) * self.yscalings

        if return_covar == True:
            coeffs_logerr = torch.clamp(coeffs_logerr, -10, 5)
            coeffserr = torch.exp(coeffs_logerr) ** 2
            powers_err = torch.arange(0, self.ndeg * 2 + 1, 2).to(self.device)
            emu_p1derr = torch.sqrt(
                torch.sum(
                    coeffserr[:, powers, None]
                    * (log_kMpc[None, :] ** powers_err[None, :, None]),
                    axis=1,
                )
            )
            emu_p1derr = emu_p1derr.detach().cpu().numpy().flatten()
            emu_p1derr = (
                10 ** (emu_p1d) * np.log(10) * emu_p1derr * self.yscalings
            )
            covar = np.outer(emu_p1derr, emu_p1derr)
            return emu_p1d, covar

        else:
            return emu_p1d

    def emulate_arr_p1d_Mpc(
        self, emu_calls, log_kMpc, return_covar=False, z=None
    ):
        log_kMpc = torch.Tensor(log_kMpc).to(self.device)

        with torch.no_grad():
            emu_calls = (emu_calls - self.paramLims[None, :, 0]) / (
                self.paramLims[None, :, 1] - self.paramLims[None, :, 0]
            ) - 0.5
            emu_calls = torch.Tensor(emu_calls)

            # ask emulator to emulate P1D (and its uncertainty)
            coeffsPred, coeffs_logerr = self.nn(emu_calls.to(self.device))

            powers = torch.arange(0, self.ndeg + 1, 1).to(self.device)
            emu_p1d = torch.sum(
                coeffsPred[:, :, None]
                * (log_kMpc[:, None, :] ** powers[None, :, None]),
                axis=1,
            )

            emu_p1d = emu_p1d.detach().cpu().numpy()

            emu_p1d = 10**emu_p1d * self.yscalings

        if return_covar == True:
            coeffs_logerr = torch.clamp(coeffs_logerr, -10, 5)
            coeffserr = torch.exp(coeffs_logerr) ** 2
            powers_err = torch.arange(0, self.ndeg * 2 + 1, 2).to(self.device)
            emu_p1derr = torch.sqrt(
                torch.sum(
                    coeffserr[:, :, None]
                    * (log_kMpc[:, None, :] ** powers_err[None, :, None]),
                    axis=1,
                )
            )
            emu_p1derr = emu_p1derr.detach().cpu().numpy()
            emu_p1derr = (
                10 ** (emu_p1d) * np.log(10) * emu_p1derr * self.yscalings
            )
            covar = np.zeros(
                (emu_p1derr.shape[0], emu_p1derr.shape[1], emu_p1derr.shape[1])
            )
            for ii in range(emu_p1derr.shape[0]):
                covar[ii] = np.outer(emu_p1derr[ii], emu_p1derr[ii])
            return emu_p1d, covar

        else:
            return emu_p1d

    def check_hull(self):
        training_points = [
            d for d in self.training_data if d["ind_axis"] == "average"
        ]
        training_points = [
            d for d in training_points if d["ind_phase"] == "average"
        ]

        training_points = [
            {
                key: value
                for key, value in training_points[i].items()
                if key in self.emu_params
            }
            for i in range(len(training_points))
        ]
        training_points = self._sort_dict(training_points, self.emu_params)
        training_points = [
            list(training_points[i].values())
            for i in range(len(training_points))
        ]
        training_points = np.array(training_points)

        self.hull = Delaunay(training_points)

    def test_hull(self, model_test):
        test_point = [model_test[param] for param in self.emu_params]
        test_point = np.array(test_point)
        return self.hull.find_simplex(test_point) >= 0
