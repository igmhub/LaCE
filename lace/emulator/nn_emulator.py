import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import copy
import random
import time
from warnings import warn
import json

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
from scipy.interpolate import interp1d


class NNEmulator(base_emulator.BaseEmulator):
    """A class for training an emulator.

    Args:
        archive (class, optional): 
            Data archive used for training the emulator. Required when using a custom emulator. If not provided, defaults to None.
        training_set (str): 
            Specific training set. Options are 'Cabayol23'.
        emu_params (list): 
            A list of emulator parameters.
        emulator_label (str): 
            Specific emulator label. Options are 'Cabayol23' and 'Nyx_v0'.
        kmax_Mpc (float, optional): 
            The maximum k in Mpc^-1 to use for training. Defaults to 3.5.
        nepochs (int, optional): 
            The number of epochs to train for. Defaults to 200.
        model_path (str, optional): 
            The path to a pretrained model. Defaults to None.
        train (bool, optional): 
            Whether to train the emulator or not. Defaults to True. If False, a model path must be provided.

    Attributes:
        emulator: The trained emulator instance.
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
        nyx_file=None,
        weighted_emulator=True,
        nhidden=5,
        max_neurons=50,
        seed=32,
        fprint=print,
        lr0=1e-3,
        batch_size=100,
        weight_decay=1e-4,
        amsgrad=True,
        z_max=10,
        use_gpu_for_evaluation=False,
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
        # training data settings
        self.drop_sim = drop_sim
        self.drop_z = drop_z
        self.z_max = z_max
        self.weighted_emulator = weighted_emulator
        self.nhidden = nhidden
        self.print = fprint
        self.lr0 = lr0
        self.max_neurons = max_neurons
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.amsgrad = True

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # check input #
        training_set_all = ["Pedersen21", "Cabayol23", "Nyx23_Oct2023"]
        emulator_label_all = [
            "Cabayol23",
            "Cabayol23+",
            "Nyx_v0",
            "Nyx_alphap",
            "Nyx_alphap_extended",
            "Cabayol23_extended",
            "Nyx_v0_extended",
            "Cabayol23+_extended",
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
                archive = nyx_archive.NyxArchive(
                    nyx_version=training_set[6:], nyx_file=nyx_file
                )

            self.training_data = archive.get_training_data(
                emu_params=self.emu_params,
                drop_sim=self.drop_sim,
                z_max=self.z_max,
            )

        elif (training_set is None) & (archive is not None):
            self.print(
                "Use custom archive provided by the user to train emulator"
            )
            self.training_data = archive.get_training_data(
                emu_params=self.emu_params,
                drop_sim=self.drop_sim,
                drop_z=self.drop_z,
                z_max=self.z_max,
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
                    z_max=self.z_max,
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
                self.weight_decay,
                self.batch_size,
                self.amsgrad,
            ) = (4, 5, 100, 75, 5, 100, 1e-3, 1e-4, 100, False)

        if emulator_label == "Cabayol23+":
            self.print(
                r"Neural network emulating the optimal P1D of Gadget simulations "
                + "fitting coefficients to a 5th degree polynomial. It "
                + "goes to scales of 4Mpc^{-1} and z<=4.5. The parameters "
                + "passed to the emulator will be overwritten to match "
                + "these ones. This option is an updated on wrt to the one in the Cabayol+23 paper."
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
                self.weight_decay,
                self.batch_size,
                self.amsgrad,
            ) = (4, 5, 510, 500, 5, 250, 7e-4, 9.6e-3, 100, True)

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
                self.weight_decay,
                self.batch_size,
                self.amsgrad,
            ) = (4, 6, 800, 700, 5, 150, 5e-5, 1e-4, 100, True)

        elif emulator_label == "Nyx_alphap":
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
                "alpha_p",
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
                self.weight_decay,
                self.batch_size,
                self.amsgrad,
            ) = (4, 6, 600, 500, 6, 400, 2.5e-4, 8e-3, 100, True)

            
        elif emulator_label == "Nyx_alphap_extended":
            self.print(
                r"Neural network emulating the optimal P1D of Nyx simulations "
                + "fitting coefficients to a 8th degree polynomial. It "
                + "goes to scales of 8Mpc^{-1} and z<=4.5. The parameters "
                + "passed to the emulator will be overwritten to match "
                + "these ones"
            )
            self.emu_params = [
                "Delta2_p",
                "n_p",
                "alpha_p",
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
                self.weight_decay,
                self.batch_size,
                self.amsgrad
            ) = (8, 8, 600, 500, 6, 400, 2.5e-4,8e-3,100,True)
            
            
        
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

        if emulator_label == "Cabayol23+_extended":
            self.print(
                r"Neural network emulating the optimal P1D of Gadget simulations "
                + "fitting coefficients to a 5th degree polynomial. It "
                + "goes to scales of 4Mpc^{-1} and z<=4.5. The parameters "
                + "passed to the emulator will be overwritten to match "
                + "these ones. This option is an updated on wrt to the one in the Cabayol+23 paper."
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
                self.weight_decay,
                self.batch_size,
                self.amsgrad,
                self.weighted_emulator,
            ) = (8, 7, 250, 200, 4, 250, 7.1e-4, 4.1e-3, 100, True, True)

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

        _ = self.training_data[0]["k_Mpc"] > 0
        self.kmin_Mpc = np.min(self.training_data[0]["k_Mpc"][_])
        self._calculate_normalization(archive)

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
            # CPU vs GPU
            if use_gpu_for_evaluation:
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
            else:
                self.device = torch.device("cpu")
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
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self.train()

            if self.save_path is not None:
                # saves the model in the predefined path after training
                self.save_emulator()

    def _sort_dict(self, dct, keys):
        """
        Sorts a list of dictionaries based on specified keys.

        Args:
            dct (list): A list of dictionaries to be sorted.
            keys (list): A list of keys to sort the dictionaries by.

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

    def _calculate_normalization(self, archive):
        """
        Calculates normalization parameters based on the training data.

        Args:
            archive (Archive): The archive containing the training data.

        Side Effects:
            - Sets the `self.paramLims` attribute to the calculated parameter limits.
        """
        training_data_all = archive.get_training_data(
            emu_params=self.emu_params
        )

        data = []
        for ii in range(len(training_data_all)):
            data_dict = {}
            for jj, param in enumerate(self.emu_params):
                try:
                    value = training_data_all[ii]["cosmo_params"][param]
                except:
                    value = training_data_all[ii][param]
                data_dict[param] = value
            data.append(data_dict)

        data = self._sort_dict(data, self.emu_params)
        # sort the data by emulator parameters
        data = [list(data[i].values()) for i in range(len(training_data_all))]
        data = np.array(data)

        self.paramLims = np.concatenate(
            (
                data.min(0).reshape(len(data.min(0)), 1),
                data.max(0).reshape(len(data.max(0)), 1),
            ),
            1,
        )

    def _obtain_sim_params(self):
        """
        Obtains simulation parameters including k_Mpc and redshift values.

        Returns:
            k_Mpc (np.ndarray): The simulation k values.

        Side Effects:
            - Sets the `self.k_Mpc`, `self.Nk`, and `self.yscalings` attributes.
        """
        self.k_mask = [
            (self.training_data[i]["k_Mpc"] < self.kmax_Mpc)
            & (self.training_data[i]["k_Mpc"] > 0)
            for i in range(len(self.training_data))
        ]

        k_Mpc_train = [
            self.training_data[i]["k_Mpc"][self.k_mask[i]]
            for i in range(len(self.training_data))
        ]
        k_Mpc_train = np.array(k_Mpc_train)
        k_Mpc_train = torch.Tensor(k_Mpc_train)
        self.k_Mpc = k_Mpc_train

        Nk = len(k_Mpc_train[0])
        self.Nk = Nk

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

        if not self.emulator_label in ["Cabayol23", "Cabayol23_extended"]:
            for ii, p1d in enumerate(training_label):
                fit_p1d = poly_p1d.PolyP1D(self.k_Mpc[ii], p1d, deg=self.ndeg)
                training_label[ii] = fit_p1d.P_Mpc(self.k_Mpc[ii])
            self.yscalings = np.median(np.log(training_label))

        else:
            self.yscalings = np.median(training_label)

        return k_Mpc_train

    def _get_training_data_nn(self):
        """
        Retrieves and normalizes training data for the neural network.

        Returns:
            torch.Tensor: The normalized training data as a tensor.
        """

        training_data = []
        for ii in range(len(self.training_data)):
            data_dict = {}
            for jj, param in enumerate(self.emu_params):
                try:
                    value = self.training_data[ii]["cosmo_params"][param]
                except:
                    value = self.training_data[ii][param]
                data_dict[param] = value
            training_data.append(data_dict)

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
        Retrieves and scales p1d_Mpc values from the training data for the neural network.

        Returns:
            torch.Tensor: The scaled p1d_Mpc values as a tensor.
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

        if not self.emulator_label in ["Cabayol23", "Cabayol23_extended"]:
            for ii, p1d in enumerate(training_label):
                fit_p1d = poly_p1d.PolyP1D(self.k_Mpc[ii], p1d, deg=self.ndeg)
                training_label[ii] = fit_p1d.P_Mpc(self.k_Mpc[ii])
            training_label = np.log(training_label) / self.yscalings**2

        else:
            training_label = np.log10(training_label / self.yscalings)

        training_label = torch.Tensor(training_label)

        return training_label

    def _set_weights(self):
        """
        Sets weights for downscaling the impact of small scales in the emulator.

        Returns:
            torch.Tensor: The weights for the loss function.
        """
        w = torch.ones(size=(self.Nk,))
        if (self.kmax_Mpc > 4) & (self.weighted_emulator == True):
            self.print("Exponential downweighting loss function at k>4")
            exponential_values = torch.linspace(
                0, 1.4, len(self.k_Mpc[0][self.k_Mpc[0] > 4])
            )
            w[self.k_Mpc[0] > 4] = torch.exp(-exponential_values)
        return w

    def train(self):
        """
        Trains the emulator neural network with the given data and settings.

        Args:
            None

        Returns:
            None

        Side Effects:
            - Trains the neural network defined by `self.nn`.
            - Saves the trained model if `self.model_path` is specified.
        """

        kMpc_train = self._obtain_sim_params()

        loss_function_weights = self._set_weights()
        loss_function_weights = loss_function_weights.to(self.device)

        log_kMpc_train = torch.log10(kMpc_train).to(self.device)
        # log_kMpc_train = torch.log(kMpc_train).to(self.device)

        self.log_kMpc = log_kMpc_train  # [0]

        self.nn = nn_architecture.MDNemulator_polyfit(
            nhidden=self.nhidden,
            ndeg=self.ndeg,
            max_neurons=self.max_neurons,
            ninput=len(self.emu_params),
        )

        if self.model_path is not None:
            pretrained_model = torch.load(
                os.path.join(self.models_dir, self.model_path),
                map_location="cpu",
            )
            self.nn.load_state_dict(pretrained_model["emulator"])
            self.nn.to(self.device)
            self.print("Loading pretrained initial state.")

        if not self.emulator_label in ["Cabayol23", "Cabayol23_extended"]:
            optimizer = optim.AdamW(
                self.nn.parameters(),
                lr=self.lr0,
                weight_decay=self.weight_decay,
                amsgrad=self.amsgrad,
            )
        else:
            optimizer = optim.Adam(
                self.nn.parameters(),
                lr=self.lr0,
                weight_decay=self.weight_decay,
            )

        scheduler = lr_scheduler.StepLR(optimizer, self.step_size, gamma=0.1)

        training_data = self._get_training_data_nn()
        training_label = self._get_training_pd1_nn()

        trainig_dataset = TensorDataset(
            training_data, training_label, log_kMpc_train
        )
        loader_train = DataLoader(
            trainig_dataset, batch_size=self.batch_size, shuffle=True
        )

        self.nn.to(self.device)
        self.print(f"Training NN on {len(training_data)} points")
        t0 = time.time()

        for epoch in range(self.nepochs):
            for datain, p1D_true, logkP1D in loader_train:
                kP1D = 10**logkP1D

                p1D_true_scaled = (
                    p1D_true.to(self.device) * kP1D.to(self.device) / torch.pi
                )

                optimizer.zero_grad()

                coeffsPred, coeffs_logerr = self.nn(datain.to(self.device))  #
                coeffs_logerr = torch.clamp(coeffs_logerr, -10, 5)
                coeffserr = torch.exp(coeffs_logerr) ** 2

                powers = torch.arange(0, self.ndeg + 1, 1).to(self.device)
                P1Dpred = torch.sum(
                    coeffsPred[:, powers, None]
                    * (logkP1D[:, None, :] ** powers[None, :, None]),
                    axis=1,
                )

                powers_err = torch.arange(0, self.ndeg * 2 + 1, 2).to(
                    self.device
                )
                P1Derr = torch.sqrt(
                    torch.sum(
                        coeffserr[:, powers, None]
                        * (logkP1D[:, None, :] ** powers_err[None, :, None]),
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
        """Saves the current state of the emulator to a file.

        This method saves both the model state dictionary and metadata about the emulator 
        to a file specified by `self.save_path`.

        Metadata includes:
            - `training_set`: The training set used.
            - `emulator_label`: The label of the emulator.
            - `drop_sim`: The simulation to drop from the training set.
            - `drop_z`: The redshift value to drop from the training set.

        Side Effects:
            - Writes a file to `self.save_path` with the saved emulator state and metadata.
        """
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
        """Emulates the power spectrum P1D at a given set of k_Mpc values.

        Args:
            model (dict): A dictionary containing the model parameters required for emulation.
            k_Mpc (array-like): The k values in Mpc^-1 at which to emulate the P1D.
            return_covar (bool, optional): Whether to return the covariance matrix. Defaults to False.
            z (float, optional): The redshift value. Currently not used. Defaults to None.

        Returns:
            numpy.ndarray: Emulated P1D values at the provided k_Mpc.
            If `return_covar` is True:
                tuple:
                    - numpy.ndarray: Emulated P1D values.
                    - numpy.ndarray: Covariance matrix of the emulated P1D.

        Warnings:
            - If any `k_Mpc` values exceed the training range or are below the minimum training range, a warning will be issued.
        """
        
        logk_Mpc = torch.log10(torch.Tensor(k_Mpc)).to(self.device)

        if np.max(k_Mpc) > self.kmax_Mpc:
            warn(
                f"Some of the requested k's are higher than the maximum training value k={self.kmax_Mpc}",
            )
        elif np.min(k_Mpc) < self.kmin_Mpc:
            warn(
                f"Some of the requested k's are lower than the minimum training value k={self.kmin_Mpc}"
            )

        with torch.no_grad():
            emu_call = {}
            for param in self.emu_params:
                if param not in model:
                    raise ValueError(f"{param} not in input model")
                emu_call[param] = model[param]

            emu_call = [emu_call[param] for param in self.emu_params]

            emu_call = (emu_call - self.paramLims[:, 0]) / (
                self.paramLims[:, 1] - self.paramLims[:, 0]
            ) - 0.5

            emu_call = torch.Tensor(emu_call).unsqueeze(0)

            # ask emulator to emulate P1D (and its uncertainty)
            coeffsPred, coeffs_logerr = self.nn(emu_call.to(self.device))

            powers = torch.arange(0, self.ndeg + 1, 1).to(self.device)
            emu_p1d = torch.sum(
                coeffsPred[:, :, None]
                * (logk_Mpc[None, None, :] ** powers[None, :, None]),
                axis=1,
            )

            emu_p1d = emu_p1d.detach().cpu().numpy().flatten()

            if self.emulator_label in ["Cabayol23", "Cabayol23_extended"]:
                emu_p1d = 10 ** (emu_p1d) * self.yscalings
            else:
                emu_p1d = np.exp(emu_p1d * self.yscalings**2)

        if return_covar:
            coeffs_logerr = torch.clamp(coeffs_logerr, -10, 5)
            coeffserr = torch.exp(coeffs_logerr) ** 2
            powers_err = torch.arange(0, self.ndeg * 2 + 1, 2).to(self.device)
            emu_p1derr = torch.sqrt(
                torch.sum(
                    coeffserr[:, :, None]
                    * (logk_Mpc[None, None, :] ** powers_err[None, :, None]),
                    axis=1,
                )
            )

            emu_p1derr = emu_p1derr.detach().cpu().numpy().flatten()
            if self.emulator_label in ["Cabayol23", "Cabayol23_extended"]:
                emu_p1derr = (
                    10 ** (emu_p1d) * np.log(10) * emu_p1derr * self.yscalings
                )
            else:
                emu_p1derr = (
                    np.exp(emu_p1d * self.yscalings**2)
                    * self.yscalings**2
                    * emu_p1derr
                )

            covar = np.outer(emu_p1derr, emu_p1derr)
            return emu_p1d, covar

        else:
            return emu_p1d


    def emulate_arr_p1d_Mpc(self, emu_calls, k_Mpc, return_covar=False, z=None):
        """Emulates the power spectrum P1D for an array of emulator parameters.

        Args:
            emu_calls (array-like): An array of emulator parameter sets.
            k_Mpc (array-like): The k values in Mpc^-1 at which to emulate the P1D.
            return_covar (bool, optional): Whether to return the covariance matrices. Defaults to False.
            z (float, optional): The redshift value. Currently not used. Defaults to None.

        Returns:
            numpy.ndarray: Emulated P1D values for each set of parameters at the provided k_Mpc.
            If `return_covar` is True:
                tuple:
                    - numpy.ndarray: Emulated P1D values for each parameter set.
                    - numpy.ndarray: Covariance matrices for each parameter set.

        Warnings:
            - If any `k_Mpc` values exceed the training range or are below the minimum training range, 
              a warning will be issued.
        """
        logk_Mpc = torch.log10(torch.Tensor(k_Mpc)).to(self.device)

        emu_p1ds = np.zeros(shape=(len(emu_calls), k_Mpc.shape[1]))
        emu_p1d_interp = np.zeros(shape=(len(emu_calls), k_Mpc.shape[1]))
        covars = np.zeros(
            shape=(len(emu_calls), k_Mpc.shape[1], k_Mpc.shape[1])
        )

        with torch.no_grad():
            emu_calls = (emu_calls - self.paramLims[None, :, 0]) / (
                self.paramLims[None, :, 1] - self.paramLims[None, :, 0]
            ) - 0.5

            emu_calls = torch.Tensor(emu_calls)

            # ask emulator to emulate P1D (and its uncertainty)
            coeffsPred, coeffs_logerr = self.nn(emu_calls.to(self.device))

            powers = torch.arange(0, self.ndeg + 1, 1).to(self.device)

            emu_p1ds = torch.sum(
                coeffsPred[:, :, None]
                * (logk_Mpc[:, None, :] ** powers[None, :, None]),
                axis=1,
            )

            emu_p1ds = emu_p1ds.detach().cpu().numpy()

            if self.emulator_label in ["Cabayol23", "Cabayol23_extended"]:
                emu_p1ds = 10 ** (emu_p1ds) * self.yscalings
            else:
                emu_p1ds = np.exp(emu_p1ds * self.yscalings**2)

        if return_covar:
            coeffs_logerr = torch.clamp(coeffs_logerr, -10, 5)
            coeffserr = torch.exp(coeffs_logerr) ** 2
            powers_err = torch.arange(0, self.ndeg * 2 + 1, 2).to(self.device)

            emu_p1derr = torch.sqrt(
                torch.sum(
                    coeffserr[:, :, None]
                    * (logk_Mpc[:, None, :] ** powers_err[None, :, None]),
                    axis=1,
                )
            )
            emu_p1derrs = emu_p1derr.detach().cpu().numpy()

            if self.emulator_label in ["Cabayol23", "Cabayol23_extended"]:
                emu_p1derrs = (
                    10 ** (emu_p1ds) * np.log(10) * emu_p1derrs * self.yscalings
                )
            else:
                emu_p1derrs = (
                    np.exp(emu_p1ds * self.yscalings**2)
                    * self.yscalings**2
                    * emu_p1derrs
                )

            for ii, emu_p1derr in enumerate(emu_p1derrs):
                covars[ii] = np.outer(emu_p1derr, emu_p1derr)

            return emu_p1ds, covars

        else:
            return emu_p1ds


    def check_hull(self):
        """Checks and creates the convex hull of the training data in the emulator parameter space.

        This method filters the training data to include only those with "average" values for 
        both the axis and phase, sorts the data according to emulator parameters, and then computes 
        the convex hull using the Delaunay triangulation.

        Side Effects:
            - Sets the `self.hull` attribute to the computed Delaunay triangulation of the training data.
        """
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
        """Tests if a given parameter set is within the convex hull of the training data.

        Args:
            model_test (dict): A dictionary containing the parameter set to be tested.

        Returns:
            bool: True if the parameter set is inside or on the boundary of the convex hull, False otherwise.
        """
        test_point = [model_test[param] for param in self.emu_params]
        test_point = np.array(test_point)
        return self.hull.find_simplex(test_point) >= 0
