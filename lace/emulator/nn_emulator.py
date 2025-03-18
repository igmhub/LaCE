import numpy as np
import matplotlib.pyplot as plt
import os, sys
import json
import random
import time
from warnings import warn
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

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


def func_poly(x, a, b, c, d, e):
    return a + b * x**0.5 + c * x + d * x**2 + e * x**3


def func_poly_train(y, ypars):
    x = y[None, :]
    pars = ypars[:, :, None]
    return (
        pars[:, 0]
        + pars[:, 1] * x**0.5
        + pars[:, 2] * x
        + pars[:, 3] * x**2
        + pars[:, 4] * x**3
    )


def func_poly_evaluate(x, pars):
    return (
        pars[:, 0]
        + pars[:, 1] * x**0.5
        + pars[:, 2] * x
        + pars[:, 3] * x**2
        + pars[:, 4] * x**3
    )


# def func_poly(x, a, b, c, d, e, f, g):
#     return (
#         a * x**0.5
#         + b * x**0.75
#         + c * x
#         + d * x**2
#         + e * x**3
#         + f * x**4
#         + g * x**5
#     )


# def func_poly_train(y, ypars):
#     x = y[None, :]
#     pars = ypars[:, :, None]
#     res = (
#         pars[:, 0] * x**0.5
#         + pars[:, 1] * x**0.75
#         + pars[:, 2] * x
#         + pars[:, 3] * x**2
#         + pars[:, 4] * x**3
#         + pars[:, 5] * x**4
#         + pars[:, 6] * x**5
#     )
#     return res


# def func_poly_evaluate(x, pars):
#     res = (
#         pars[:, 0] * x**0.5
#         + pars[:, 1] * x**0.75
#         + pars[:, 2] * x
#         + pars[:, 3] * x**2
#         + pars[:, 4] * x**3
#         + pars[:, 5] * x**4
#         + pars[:, 6] * x**5
#     )
#     return res


# def func_norm_logmF():
#     pfit = np.array([0.25705317, 1.04475434, -0.76853821, 0.01896378])
#     return np.poly1d(pfit)


def init_xavier(m):
    """Initialization of the NN.
    This is quite important for a faster training
    """
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


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
        include_central=False,
        average="both",
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
        gamma_optimizer=0.1,
        max_neurons=50,
        seed=32,
        fprint=print,
        lr0=1e-3,
        batch_size=100,
        weight_decay=1e-4,
        z_max=10,
        init_xavier=True,
        pred_error=False,
    ):
        # store emulator settings
        self.emulator_label = emulator_label
        self.training_set = training_set
        self.emu_params = emu_params
        self.kmax_Mpc = kmax_Mpc
        self.ndeg = ndeg
        self.nepochs = nepochs
        self.step_size = step_size
        self.init_xavier = init_xavier
        self.pred_error = pred_error
        self.average = average
        # paths to save/load models
        self.save_path = save_path
        self.model_path = model_path
        self.models_dir = PROJ_ROOT / "data/"  # training data settings
        self.drop_sim = drop_sim
        self.drop_z = drop_z
        self.include_central = include_central
        self.z_max = z_max
        self.weighted_emulator = weighted_emulator
        self.nhidden = nhidden
        self.print = fprint
        self.lr0 = lr0
        self.gamma_optimizer = gamma_optimizer
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
                print(key, value)
                setattr(self, key, value)

        if emulator_label[:4] == "CH24":
            repo = os.path.dirname(lace.__path__[0])
            fname = os.path.join(repo, "data", "ff_mpgcen.npy")
            self.input_norm = np.load(fname, allow_pickle=True).item()
            self.norm_imF = interp1d(
                self.input_norm["mF"], self.input_norm["p1d_Mpc_mF"], axis=0
            )
            self.pred_error = False
        else:
            self.pred_error = True

        if (emulator_label[:4] != "CH24") | (
            (emulator_label[:4] == "CH24") & train
        ):
            archive, self.training_data = select_training(
                archive=archive,
                training_set=training_set,
                emu_params=self.emu_params,
                drop_sim=self.drop_sim,
                drop_z=self.drop_z,
                z_max=self.z_max,
                average=self.average,
                include_central=self.include_central,
                nyx_file=nyx_file,
                train=train,
                print_func=self.print,
            )
            self._calculate_normalization(archive)

        if not train:
            self._load_pretrained_model()
        else:
            self._check_consistency()

            self.print(f"Samples in training_set: {len(self.training_data)}")
            self.kp_Mpc = archive.kp_Mpc

            _ = self.training_data[0]["k_Mpc"] > 0
            self.kmin_Mpc = np.min(self.training_data[0]["k_Mpc"][_])

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
        self._setup_nn(
            pretrained_model["emulator"], pretrained_model["metadata"]
        )
        self.print("Model loaded. No training needed")

        self._check_model_consistency(pretrained_model["metadata"])

    def _setup_nn(self, state_dict, metadata):
        self.nn = nn_architecture.MDNemulator_polyfit(
            nhidden=self.nhidden,
            ndeg=self.ndeg,
            max_neurons=self.max_neurons,
            ninput=len(self.emu_params),
            pred_error=self.pred_error,
        )
        self.nn.load_state_dict(state_dict)
        self.nn.to(self.device)

        if self.emulator_label[:4] == "CH24":
            self.min_params = metadata["min_params"]
            self.max_params = metadata["max_params"]
            self.pred_error = metadata["pred_error"]
            self.kmax_Mpc = metadata["kmax_Mpc"]
            self.kmin_Mpc = metadata["kmin_Mpc"]
            self.kp_Mpc = metadata["kp_Mpc"]
        else:
            # self.yscalings = metadata["yscalings"]
            _ = self._obtain_sim_params()

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
                # temporal hack
                if (expected == "Nyx_alphap_cov_central") and (
                    loaded == "Nyx_alphap_cov"
                ):
                    continue
                if key == "training_set":
                    warn(
                        f"Training set mismatch: Expected '{expected}' but loaded '{loaded}'"
                    )
                else:
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

        self.min_params = np.min(data, axis=0)
        self.max_params = np.max(data, axis=0)

        # self.paramLims = np.concatenate(
        #     (
        #         data.min(0).reshape(len(data.min(0)), 1),
        #         data.max(0).reshape(len(data.max(0)), 1),
        #     ),
        #     1,
        # )

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
        self.kmin_Mpc = k_Mpc_train[0][0]
        Nk = len(k_Mpc_train[0])
        self.Nk = Nk
        self.k_Mpc = torch.Tensor(np.array(k_Mpc_train))

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

        if self.emulator_label in ["Cabayol23", "Cabayol23_extended"]:
            self.yscalings = np.median(training_label)
        elif self.emulator_label[:4] == "CH24":
            pass
        else:
            for ii, p1d in enumerate(training_label):
                fit_p1d = poly_p1d.PolyP1D(self.k_Mpc[ii], p1d, deg=self.ndeg)
                training_label[ii] = fit_p1d.P_Mpc(self.k_Mpc[ii])
            self.yscalings = np.median(np.log(training_label))

        return k_Mpc_train

    def _get_training_data_nn(self):
        """
        Retrieves and normalizes training data for the neural network.

        Returns:
            torch.Tensor: The normalized training data as a tensor.
        """

        # "x" in the emulator
        train_emu_params = np.zeros(
            (len(self.training_data), len(self.emu_params))
        )
        # "y" in the emulator
        k_Mpc = self.training_data[0]["k_Mpc"]
        mask = (k_Mpc > 0) & (k_Mpc < self.kmax_Mpc)
        k_Mpc_mask = k_Mpc[mask]
        x_fit = k_Mpc_mask / self.kmax_Mpc

        if self.emulator_label[:4] == "CH24":
            yscalings = np.zeros((len(self.training_data), len(x_fit)))
        train_p1d = np.zeros((len(self.training_data), len(x_fit)))

        # loop over archive to extract x and y data
        for ii, entry in enumerate(self.training_data):
            # extract x data
            for jj, param in enumerate(self.emu_params):
                # TODO check
                if param in entry["cosmo_params"]:
                    train_emu_params[ii, jj] = entry["cosmo_params"][param]
                elif param in entry:
                    train_emu_params[ii, jj] = entry[param]
                else:
                    raise ValueError(
                        f"Parameter '{param}' not found in training data"
                    )
            # extract y data
            mask = (entry["k_Mpc"] > 0) & (entry["k_Mpc"] < self.kmax_Mpc)
            if self.emulator_label[:4] == "CH24":
                # smooth version goes into likelihood
                # normalize and compute parameters of fit
                yscalings[ii] = np.interp(
                    k_Mpc_mask,
                    self.input_norm["k_Mpc"],
                    self.norm_imF(entry["mF"]),
                )
                # yscalings[ii] = self.func_norm(np.log(entry["mF"])) * norm_use
                store_fit, _ = curve_fit(
                    func_poly,
                    x_fit,
                    np.log(entry["p1d_Mpc"][mask] / yscalings[ii]),
                )
                # evaluate the fit
                train_p1d[ii] = yscalings[ii] * np.exp(
                    func_poly(x_fit, *store_fit)
                )

                # XXX
                # if ii == 0:
                #     print(store_fit)
                #     plt.plot(k_Mpc_mask, k_Mpc_mask * train_p1d[ii])
                #     plt.plot(k_Mpc_mask, k_Mpc_mask * entry["p1d_Mpc"][mask])

            else:
                # original version, no smoothing!
                train_p1d[ii] = entry["p1d_Mpc"][mask]

        # normalize P1Ds
        if self.emulator_label in ["Cabayol23", "Cabayol23_extended"]:
            self.yscalings = np.median(train_p1d, axis=0)
            train_p1d = np.log10(train_p1d / self.yscalings[None, :])
        elif self.emulator_label[:4] == "CH24":
            pass
        else:
            fit_p1d = np.zeros((len(self.training_data), len(x_fit)))
            for ii in range(len(train_p1d)):
                fit = poly_p1d.PolyP1D(k_Mpc_mask, train_p1d[ii], deg=self.ndeg)
                fit_p1d[ii] = fit.P_Mpc(k_Mpc_mask)
            self.yscalings = np.median(np.log(fit_p1d))
            train_p1d = np.log(train_p1d) / self.yscalings[None, :] ** 2

        # normalize x between -0.5 and 0.5
        self.min_params = np.min(train_emu_params, axis=0)
        self.max_params = np.max(train_emu_params, axis=0)
        train_emu_params = (train_emu_params - self.min_params) / (
            self.max_params - self.min_params
        ) - 0.5

        # k_Mpc tensor
        if self.emulator_label in ["Cabayol23", "Cabayol23_extended"]:
            x_train = torch.log(k_Mpc_mask).to(self.device)
        elif self.emulator_label[:4] == "CH24":
            x_train = torch.Tensor(k_Mpc_mask / self.kmax_Mpc).to(self.device)
        else:
            x_train = torch.log10(k_Mpc_mask).to(self.device)

        # tensorize
        train_emu_params = torch.Tensor(train_emu_params).to(self.device)
        train_p1d = torch.Tensor(train_p1d).to(self.device)
        k_Mpc_mask = torch.Tensor(k_Mpc_mask).to(self.device)

        if self.emulator_label[:4] == "CH24":
            return train_emu_params, x_train, train_p1d, k_Mpc_mask, yscalings
        else:
            return train_emu_params, x_train, train_p1d, k_Mpc_mask

    def _get_training_cov(self, k_Mpc):
        if self.emulator_label == "Nyx_alphap_cov":
            training_cov = []
            self.Y1_relerr = self._load_DESIY1_err()
            z_values = np.array(list(self.Y1_relerr.keys()))
            for ii in range(len(self.training_data)):
                z = np.round(self.training_data[ii]["z"], 2)
                # Find closest z value if exact match not found
                if z not in self.Y1_relerr:
                    closest_z = z_values[np.abs(z_values - z).argmin()]
                    z = closest_z
                training_cov.append(self.Y1_relerr[z])
            training_cov = torch.Tensor(np.array(training_cov))
        else:
            training_cov = np.zeros(shape=(len(self.training_data), len(k_Mpc)))
            training_cov = torch.Tensor(training_cov)
        return training_cov

    def _get_rescalings_weights(self):
        weights_rescalings = np.ones(len(self.training_data))
        if self.emulator_label == "Nyx_alphap_cov":
            weights_rescalings[
                np.where(
                    [
                        (
                            d["ind_rescaling"] not in [0, 1]
                            and d["z"] in [3, 3.2]
                        )
                        for d in self.training_data
                    ]
                )
            ] = 1
        else:
            pass
        return torch.Tensor(weights_rescalings)

    def _load_DESIY1_err(self):
        with open(
            self.models_dir / "DESI_cov/rerr_DESI_Y1.json", "r"
        ) as json_file:
            z_to_rel = json.load(json_file)
        z_to_rel = {
            np.round(float(z), 2): rel_error
            for z, rel_error in z_to_rel.items()
        }

        return z_to_rel

    def _set_weights(self, k_Mpc):
        """
        Sets weights for downscaling the impact of small scales in the emulator.

        Returns:
            torch.Tensor: The weights for the loss function.
        """
        w = torch.ones(size=(k_Mpc.shape[0],))
        if (self.kmax_Mpc > 4) & (self.weighted_emulator == True):
            self.print("Exponential downweighting loss function at k>4")
            exponential_values = torch.linspace(0, 1.4, len(k_Mpc[k_Mpc > 4]))
            w[k_Mpc > 4] = torch.exp(-exponential_values)
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

        # set the emulator
        self.nn = nn_architecture.MDNemulator_polyfit(
            nhidden=self.nhidden,
            ndeg=self.ndeg,
            max_neurons=self.max_neurons,
            ninput=len(self.emu_params),
            pred_error=self.pred_error,
        )
        # initialize it
        if self.init_xavier:
            self.nn.apply(init_xavier)
        # send to device
        self.nn.to(self.device)

        if self.emulator_label in ["Cabayol23", "Cabayol23_extended"]:
            optimizer = optim.Adam(
                self.nn.parameters(),
                lr=self.lr0,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = optim.AdamW(
                self.nn.parameters(),
                lr=self.lr0,
                weight_decay=self.weight_decay,
                amsgrad=self.amsgrad,
            )

        scheduler = lr_scheduler.StepLR(
            optimizer, self.step_size, gamma=self.gamma_optimizer
        )

        res = self._get_training_data_nn()
        if self.emulator_label[:4] == "CH24":
            emu_params, x_P1Ds, P1Ds, k_Mpc, yscalings = res
        else:
            emu_params, x_P1Ds, P1Ds, k_Mpc = res
        loss_function_weights = self._set_weights(k_Mpc).to(self.device)
        training_cov = self._get_training_cov(k_Mpc)
        weights_rescalings = self._get_rescalings_weights()

        if self.emulator_label[:4] == "CH24":
            trainig_dataset = TensorDataset(
                emu_params,
                P1Ds,
                training_cov,
                weights_rescalings,
                torch.Tensor(yscalings).to(self.device),
            )
        else:
            trainig_dataset = TensorDataset(
                emu_params,
                P1Ds,
                training_cov,
                weights_rescalings,
            )
        loader_train = DataLoader(
            trainig_dataset, batch_size=self.batch_size, shuffle=True
        )

        self.print(f"Training NN on {len(emu_params[:,0])} points")

        self.loss_arr = []
        t0 = time.time()

        for epoch in tqdm(range(self.nepochs), desc="Training epochs"):
            _loss_arr = []
            for entry in loader_train:
                datain = entry[0]
                p1D_true = entry[1]
                P1D_desi_err = entry[2]
                weights_rescalings = entry[3]
                if self.emulator_label[:4] == "CH24":
                    yscalings = entry[4]

                optimizer.zero_grad()

                if self.pred_error:
                    coeffsPred, coeffs_logerr = self.nn(datain.to(self.device))
                else:
                    coeffsPred = self.nn(datain.to(self.device))

                # transform back from coeff to P1D
                if self.emulator_label[:4] == "CH24":
                    P1Dpred = yscalings * torch.exp(
                        func_poly_train(x_P1Ds, coeffsPred)
                    )
                else:
                    powers = torch.arange(0, self.ndeg + 1, 1).to(self.device)
                    P1Dpred = torch.sum(
                        coeffsPred[:, powers, None]
                        * (x_P1Ds[None, None, :] ** powers[None, :, None]),
                        axis=1,
                    )

                if self.pred_error:
                    coeffs_logerr = torch.clamp(coeffs_logerr, -10, 5)
                    coeffserr = torch.exp(coeffs_logerr) ** 2
                    powers_err = torch.arange(0, self.ndeg * 2 + 1, 2).to(
                        self.device
                    )
                    P1Derr = torch.sqrt(
                        torch.sum(
                            coeffserr[:, powers, None]
                            * (x_P1Ds[:, None, :] ** powers_err[None, :, None]),
                            axis=1,
                        )
                    )
                    P1Dlogerr = torch.log(
                        torch.sqrt(
                            P1Derr**2
                            + P1D_desi_err.to(self.device) ** 2 * p1D_true**2
                        )
                    )

                    log_prob = (P1Dpred - p1D_true.to(self.device)).pow(2) / (
                        P1Derr**2
                        + P1D_desi_err.to(self.device) ** 2 * p1D_true**2
                    ) + 2 * P1Dlogerr

                else:
                    log_prob = (P1Dpred - p1D_true).pow(2)

                if self.emulator_label == "Nyx_alphap_cov":
                    log_prob = (
                        weights_rescalings.to(self.device)[:, None] * log_prob
                    )
                else:
                    log_prob = loss_function_weights[None, :] * log_prob

                loss = torch.nansum(log_prob, 1)
                loss = torch.nanmean(torch.sqrt(loss), 0)

                loss.backward()
                optimizer.step()

                _loss_arr.append(loss.item())

            self.loss_arr.append(np.mean(_loss_arr))
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
            "min_params": self.min_params,
            "max_params": self.max_params,
            "pred_error": self.pred_error,
            "kmax_Mpc": self.kmax_Mpc,
            "kmin_Mpc": self.kmin_Mpc,
            "kp_Mpc": self.kp_Mpc,
        }

        if self.emulator_label[:4] != "CH24":
            metadata["yscalings"] = self.yscalings

        # Combine model_state_dict and metadata into a single dictionary
        model_with_metadata = {
            "emulator": model_state_dict,
            "metadata": metadata,
        }

        if not self.emulator_label:
            save_path = self.save_path
        else:
            # Create model directory under data/NNmodels/{emulator_label}
            dir_path = PROJ_ROOT / "data" / "NNmodels" / self.emulator_label
            dir_path.mkdir(parents=True, exist_ok=True)
            self.print(f"Creating and using model directory: {dir_path}")

            # Build filename based on whether we're dropping simulations
            filename = f"{self.emulator_label}"
            if self.drop_sim is not None:
                filename += f"_drop_sim_{self.drop_sim}_{self.drop_z}"
            filename += ".pt"

            save_path = dir_path / filename

        torch.save(model_with_metadata, save_path)

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

        self.nn = self.nn.eval()

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
                if param == "mF":
                    mF = emu_call[:, ii]

            emu_call = (emu_call - self.min_params[None, :]) / (
                self.max_params[None, :] - self.min_params[None, :]
            ) - 0.5

            emu_call = torch.Tensor(emu_call).unsqueeze(0)

            # ask emulator to emulate P1D (and its uncertainty)
            res_call = self.nn(emu_call.to(self.device))
            if self.pred_error:
                coeffsPred, coeffs_logerr = res_call
            else:
                coeffsPred = res_call

            # k_Mpc tensor
            if self.emulator_label in ["Cabayol23", "Cabayol23_extended"]:
                x_P1Ds = torch.log(k_Mpc).to(self.device)
            elif self.emulator_label[:4] == "CH24":
                x_P1Ds = torch.Tensor(k_Mpc / self.kmax_Mpc).to(self.device)
                norm_use = np.zeros_like(k_Mpc)
                for ii in range(k_Mpc.shape[0]):
                    norm_use[ii] = np.interp(
                        k_Mpc,
                        self.input_norm["k_Mpc"],
                        self.norm_imF(mF[ii]),
                    )
                yscalings = torch.Tensor(norm_use).to(self.device)
            else:
                x_P1Ds = torch.log10(torch.Tensor(k_Mpc)).to(self.device)

            if self.emulator_label[:4] == "CH24":
                emu_p1d = func_poly_evaluate(x_P1Ds, coeffsPred[0])
            else:
                powers = torch.arange(0, self.ndeg + 1, 1).to(self.device)
                # TODO check next line!
                # print(x_train.shape, coeffsPred.shape, powers.shape)
                emu_p1d = torch.sum(
                    coeffsPred[0, :, :, None]
                    * (x_P1Ds[:, None, :] ** powers[None, :, None]),
                    axis=1,
                )

            if self.emulator_label in ["Cabayol23", "Cabayol23_extended"]:
                emu_p1d = 10 ** (emu_p1d) * self.yscalings
            elif self.emulator_label[:4] == "CH24":
                emu_p1d = yscalings * np.exp(emu_p1d)
            else:
                emu_p1d = np.exp(emu_p1d * self.yscalings**2)

            if emu_p1d.shape[0] == 1:
                emu_p1d = emu_p1d[0, :]

            emu_p1d = emu_p1d.detach().cpu().numpy()

        if self.pred_error and return_covar:
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
