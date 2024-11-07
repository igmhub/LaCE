import numpy as np
import matplotlib.pyplot as plt
import os
import json
import random
import time
from warnings import warn
from tqdm import tqdm

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
from lace.emulator.constants import (
    TrainingSet,
    EmulatorLabel,
    EMULATOR_PARAMS,
    EMULATOR_DESCRIPTIONS,
    PROJ_ROOT,
    GADGET_LABELS,
    NYX_LABELS,
)
from lace.emulator.select_training import select_training


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
        z_max=10,
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
        self.models_dir = PROJ_ROOT / "data/"  # training data settings
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

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        training_set_all = list(TrainingSet)
        emulator_label_all = list(EmulatorLabel)
        self.GADGET_LABELS = GADGET_LABELS
        self.NYX_LABELS = NYX_LABELS

        if isinstance(emulator_label, str):
            try:
                self.emulator_label = EmulatorLabel(emulator_label)
            except ValueError:
                valid_labels = ", ".join(label.value for label in EmulatorLabel)
                raise ValueError(
                    f"Invalid emulator_label: '{emulator_label}'. "
                    f"Available options are: {valid_labels}"
                )

        if emulator_label in EMULATOR_PARAMS:
            self.print(
                EMULATOR_DESCRIPTIONS.get(
                    emulator_label, "No description available."
                )
            )

            params = EMULATOR_PARAMS[emulator_label]
            for key, value in params.items():
                setattr(self, key, value)

        self._check_consistency()

        self.archive, self.training_data = select_training(
            archive=archive,
            training_set=training_set,
            emu_params=self.emu_params,
            drop_sim=self.drop_sim,
            drop_z=self.drop_z,
            z_max=self.z_max,
            nyx_file=nyx_file,
            train=train,
            print_func=self.print,
        )

        self.print(f"Samples in training_set: {len(self.training_data)}")
        self.kp_Mpc = self.archive.kp_Mpc

        _ = self.training_data[0]["k_Mpc"] > 0
        self.kmin_Mpc = np.min(self.training_data[0]["k_Mpc"][_])
        self._calculate_normalization(self.archive)

        if not train:
            self._load_pretrained_model()
        else:
            self.train()
            if self.save_path is not None:
                self.save_emulator()

    def _load_pretrained_model(self):
        if self.model_path is None:
            raise ValueError("If train==False, model path is required.")

        pretrained_model = torch.load(
            self.models_dir / self.model_path,
            map_location="cpu",
        )
        self._setup_nn(pretrained_model["emulator"])
        self.print("Model loaded. No training needed")

        self._check_model_consistency(pretrained_model["metadata"])

    def _setup_nn(self, state_dict):
        self.nn = nn_architecture.MDNemulator_polyfit(
            nhidden=self.nhidden,
            ndeg=self.ndeg,
            max_neurons=self.max_neurons,
            ninput=len(self.emu_params),
        )
        self.nn.load_state_dict(state_dict)
        self.nn.to(self.device)
        kMpc_train = self._obtain_sim_params()

    def _check_model_consistency(self, metadata):
        expected_values = {
            "emulator_label": self.emulator_label,
            "training_set": self.training_set,
            "drop_sim": self.drop_sim,
            "drop_z": self.drop_z,
        }

        for key, expected in expected_values.items():
            loaded = metadata[key]
            if key == "drop_z" and loaded is not None:
                loaded = float(loaded)

            if loaded != expected:
                raise ValueError(
                    f"{key} mismatch: Expected '{expected}' but loaded '{loaded}'"
                )

    def _check_consistency(self):
        """Check consistency between training data and emulator label."""
        if self.emulator_label in self.GADGET_LABELS:
            if self.training_data[0]["sim_label"][:3] != "mpg":
                raise ValueError(
                    f"Training data for {self.emulator_label} are not Gadget sims"
                )
        elif self.emulator_label in self.NYX_LABELS:
            if self.training_data[0]["sim_label"][:3] != "nyx":
                raise ValueError(
                    f"Training data for {self.emulator_label} are not Nyx sims"
                )

    @staticmethod
    def _sort_dict(dct, keys):
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

        data = NNEmulator._sort_dict(data, self.emu_params)
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
            z = self.training_data[ii]["z"]

        training_data = NNEmulator._sort_dict(training_data, self.emu_params)
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

    def _get_training_cov(self):
        if self.emulator_label == "Nyx_alphap_cov":
            training_cov = []
            self.Y1_relerr = self._laod_DESIY1_err()
            for ii in range(len(self.training_data)):
                training_cov.append(
                    self.Y1_relerr[np.round(self.training_data[ii]["z"], 1)]
                )
            training_cov = np.array(training_cov)
            training_cov = torch.Tensor(training_cov)
        else:
            training_cov = np.zeros(
                shape=(len(self.training_data), len(self.k_Mpc[0]))
            )
            training_cov = torch.Tensor(training_cov)
        return training_cov

    def _laod_DESIY1_err(self):
        with open(
            self.models_dir / "DESI_cov/rerr_DESI_Y1.json", "r"
        ) as json_file:
            z_to_rel = json.load(json_file)
        z_to_rel = {float(z): rel_error for z, rel_error in z_to_rel.items()}

        return z_to_rel

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
        training_cov = self._get_training_cov()

        trainig_dataset = TensorDataset(
            training_data, training_label, training_cov, log_kMpc_train
        )
        loader_train = DataLoader(
            trainig_dataset, batch_size=self.batch_size, shuffle=True
        )

        self.nn.to(self.device)
        self.print(f"Training NN on {len(training_data)} points")
        t0 = time.time()

        for epoch in tqdm(range(self.nepochs), desc="Training epochs"):
            for datain, p1D_true, P1D_desi_err, logkP1D in loader_train:
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
                P1Dlogerr = torch.log(
                    torch.sqrt(P1Derr**2 + P1D_desi_err**2 * p1D_true**2)
                )

                log_prob = (P1Dpred - p1D_true.to(self.device)).pow(2) / (
                    P1Derr**2 + P1D_desi_err**2 * p1D_true**2
                ) + 2 * P1Dlogerr

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
        """
        Emulate P1D values for a given set of k values in Mpc units.

        Args:
            model (dictionary): Dictionary containing the model parameters.
            k_Mpc (np.ndarray): Array of k values in Mpc units.
            return_covar (bool, optional): Whether to return covariance. Defaults to False.
            z (float, optional): Redshift value. Defaults to None.

        Returns:
            np.ndarray: Emulated P1D values.
        """

        for param in self.emu_params:
            if param not in model:
                continue
                raise ValueError(f"{param} not in input model")

        try:
            length = len(model[self.emu_params[0]])
        except:
            length = 1

        if k_Mpc.ndim == 1:
            k_Mpc = np.repeat(k_Mpc[None, :], length, axis=0)

        if np.max(k_Mpc) > self.kmax_Mpc:
            warn(
                f"Some of the requested k's are higher than the maximum training value k={self.kmax_Mpc}",
            )
        elif np.min(k_Mpc) < self.kmin_Mpc:
            warn(
                f"Some of the requested k's are lower than the minimum training value k={self.kmin_Mpc}"
            )

        with torch.no_grad():
            emu_call = np.zeros((length, len(self.emu_params)))
            for ii, param in enumerate(self.emu_params):
                emu_call[:, ii] = model[param]

            emu_call = (emu_call - self.paramLims[None, :, 0]) / (
                self.paramLims[None, :, 1] - self.paramLims[None, :, 0]
            ) - 0.5

            emu_call = torch.Tensor(emu_call).unsqueeze(0)

            # ask emulator to emulate P1D (and its uncertainty)
            coeffsPred, coeffs_logerr = self.nn(emu_call.to(self.device))

            logk_Mpc = torch.log10(torch.Tensor(k_Mpc)).to(self.device)
            powers = torch.arange(0, self.ndeg + 1, 1).to(self.device)
            emu_p1d = torch.sum(
                coeffsPred[0, :, :, None]
                * (logk_Mpc[:, None, :] ** powers[None, :, None]),
                axis=1,
            )

            emu_p1d = emu_p1d.detach().cpu().numpy()

            if self.emulator_label in ["Cabayol23", "Cabayol23_extended"]:
                emu_p1d = 10 ** (emu_p1d) * self.yscalings
            else:
                emu_p1d = np.exp(emu_p1d * self.yscalings**2)

            if emu_p1d.shape[0] == 1:
                emu_p1d = emu_p1d[0, :]

        if return_covar:
            coeffs_logerr = torch.clamp(coeffs_logerr, -10, 5)
            coeffserr = torch.exp(coeffs_logerr) ** 2
            powers_err = torch.arange(0, self.ndeg * 2 + 1, 2).to(self.device)
            emu_p1derr = torch.sqrt(
                torch.sum(
                    coeffserr[0, :, :, None]
                    * (logk_Mpc[None, None, :] ** powers_err[None, :, None]),
                    axis=1,
                )
            )

            emu_p1derr = emu_p1derr.detach().cpu().numpy()
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
            if emu_p1derr.shape[0] == 1:
                emu_p1derr = emu_p1derr[0, :]
                covar = np.outer(emu_p1derr, emu_p1derr)
            else:
                covar = np.zeros((emu_p1derr.shape[0], len(k_Mpc), len(k_Mpc)))
                for ii in range(emu_p1derr.shape[0]):
                    covar[ii] = np.outer(emu_p1derr[ii], emu_p1derr[ii])
            return emu_p1d, covar

        else:
            return emu_p1d

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
        training_points = NNEmulator._sort_dict(
            training_points, self.emu_params
        )
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
