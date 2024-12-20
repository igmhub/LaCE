from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import random
import time
from warnings import warn
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from torch.optim import lr_scheduler

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


class NNEmulator(base_emulator.BaseEmulator):
    """Neural network emulator for P1D power spectrum."""

    def __init__(
        self,
        archive: Optional[Any] = None,
        training_set: Optional[str] = None,
        emu_params: List[str] = ["Delta2_p", "n_p", "mF", "sigT_Mpc", "gamma", "kF_Mpc"],
        emulator_label: Optional[str] = None,
        include_central: bool = False,
        kmax_Mpc: float = 4.0,
        ndeg: int = 5,
        nepochs: int = 100,
        step_size: int = 75,
        drop_sim: Optional[str] = None,
        drop_z: Optional[float] = None,
        train: bool = True,
        save_path: Optional[Path] = None,
        model_path: Optional[Path] = None,
        nyx_file: Optional[str] = None,
        weighted_emulator: bool = True,
        filename_covariance: Optional[str] = "rerr_DESI_Y1.json",
        nhidden: int = 5,
        max_neurons: int = 50,
        seed: int = 32,
        fprint: callable = print,
        lr0: float = 1e-3,
        batch_size: int = 100,
        weight_decay: float = 1e-4,
        z_max: float = 10.0,
    ) -> None:
        """Initialize emulator."""
        self._init_attributes(
            emulator_label, training_set, emu_params, kmax_Mpc, ndeg, nepochs,
            step_size, save_path, model_path, drop_sim, drop_z, include_central,
            z_max, weighted_emulator, nhidden, fprint, lr0, max_neurons,
            batch_size, weight_decay, filename_covariance
        )

        self._set_random_seeds(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._validate_emulator_label()
        
        archive, training_data = select_training(
            archive=archive,
            training_set=training_set,
            emu_params=self.emu_params,
            drop_sim=self.drop_sim,
            drop_z=self.drop_z,
            z_max=self.z_max,
            include_central=self.include_central,
            nyx_file=nyx_file,
            train=train,
            print_func=self.print,
        )

        self._check_consistency(training_data)
        self._init_training_data(archive, training_data)

        if not train:
            self._load_pretrained_model(training_data)
        else:
            self.train(training_data)
            if self.save_path:
                self.save_emulator()

    def _init_attributes(self, *args) -> None:
        """Initialize class attributes."""
        (
            self.emulator_label, self.training_set, self.emu_params,
            self.kmax_Mpc, self.ndeg, self.nepochs, self.step_size,
            self.save_path, self.model_path, self.drop_sim, self.drop_z,
            self.include_central, self.z_max, self.weighted_emulator,
            self.nhidden, self.print, self.lr0, self.max_neurons,
            self.batch_size, self.weight_decay, self.filename_covariance
        ) = args
        
        self.models_dir = PROJ_ROOT / "data/"
        self.amsgrad = True
        self.GADGET_LABELS = GADGET_LABELS
        self.NYX_LABELS = NYX_LABELS

    def _set_random_seeds(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _validate_emulator_label(self) -> None:
        """Validate emulator label."""
        if isinstance(self.emulator_label, str):
            try:
                self.emulator_label = EmulatorLabel(self.emulator_label)
            except ValueError:
                valid_labels = ", ".join(label.value for label in EmulatorLabel)
                raise ValueError(
                    f"Invalid emulator_label: '{self.emulator_label}'. "
                    f"Available options are: {valid_labels}"
                )

        if self.emulator_label in EMULATOR_PARAMS:
            self.print(EMULATOR_DESCRIPTIONS.get(self.emulator_label, "No description available."))
            for key, value in EMULATOR_PARAMS[self.emulator_label].items():
                setattr(self, key, value)

    def _init_training_data(self, 
                            archive: gadget_archive.GadgetArchive,
                            training_data: List[Dict]) -> None:
        """Initialize training data."""
        self.print(f"Samples in training_set: {len(training_data)}")
        self.kp_Mpc = archive.kp_Mpc

        mask = training_data[0]["k_Mpc"] > 0
        self.kmin_Mpc = np.min(training_data[0]["k_Mpc"][mask])
        self._calculate_normalization(archive)

    def _load_pretrained_model(self,
                               training_data: List[Dict]) -> None:
        """Load pretrained model."""
        if self.model_path is None:
            raise ValueError("If train==False, model path is required.")

        pretrained_model = torch.load(self.models_dir / self.model_path, map_location="cpu")
        self._setup_nn(pretrained_model["emulator"], training_data)
        self.print("Model loaded. No training needed")
        self._check_model_consistency(pretrained_model["metadata"])

    def _setup_nn(self, 
                  state_dict: Dict[str, torch.Tensor],
                  training_data: List[Dict]) -> None:
        """Set up neural network."""
        self.nn = nn_architecture.MDNemulator_polyfit(
            nhidden=self.nhidden,
            ndeg=self.ndeg,
            max_neurons=self.max_neurons,
            ninput=len(self.emu_params),
        )
        self.nn.load_state_dict(state_dict)
        self.nn.to(self.device)
        self._obtain_sim_params(training_data)

    def _check_model_consistency(self, 
                                 metadata: Dict[str, Any]) -> None:
        """Check model consistency."""
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
                if expected == "Nyx_alphap_cov_central" and loaded == "Nyx_alphap_cov":
                    continue
                if key == "training_set":
                    warn(f"Training set mismatch: Expected '{expected}' but loaded '{loaded}'")
                else:
                    raise ValueError(f"{key} mismatch: Expected '{expected}' but loaded '{loaded}'")

    def _check_consistency(self, 
                           training_data: List[Dict]) -> None:
        """Check consistency between training data and emulator label."""
        sim_label = training_data[0]["sim_label"][:3]
        if self.emulator_label in self.GADGET_LABELS and sim_label != "mpg":
            warn(f"Training data for {self.emulator_label} are not Gadget sims")
        elif self.emulator_label in self.NYX_LABELS and sim_label != "nyx":
            warn(f"Training data for {self.emulator_label} are not Nyx sims")

    @staticmethod
    def _sort_dict(dct: List[Dict], keys: List[str]) -> List[Dict]:
        """Sort list of dictionaries by keys."""
        return [{k: d[k] for k in keys} for d in dct]

    def _calculate_normalization(self, archive: Any) -> None:
        """Calculate normalization parameters."""
        training_data_all = archive.get_training_data(emu_params=self.emu_params)

        data = []
        for entry in training_data_all:
            data_dict = {}
            for param in self.emu_params:
                try:
                    value = entry["cosmo_params"][param]
                except:
                    value = entry[param]
                data_dict[param] = value
            data.append(data_dict)

        data = self._sort_dict(data, self.emu_params)
        data = np.array([list(d.values()) for d in data])

        self.paramLims = np.column_stack((data.min(0), data.max(0)))

    def _obtain_sim_params(self,
                           training_data: List[Dict]) -> torch.Tensor:
        """Get simulation parameters."""
        self.k_mask = [(d["k_Mpc"] < self.kmax_Mpc) & (d["k_Mpc"] > 0) for d in training_data]
        k_Mpc_train = np.array([d["k_Mpc"][mask] for d, mask in zip(training_data, self.k_mask)])
        k_Mpc_train = torch.from_numpy(k_Mpc_train)
        self.k_Mpc = k_Mpc_train
        self.Nk = len(k_Mpc_train[0])

        training_label = np.array([d["p1d_Mpc"][mask].tolist() for d, mask in zip(training_data, self.k_mask)])

        if self.emulator_label not in ["Cabayol23", "Cabayol23_extended"]:
            for i, p1d in enumerate(training_label):
                fit_p1d = poly_p1d.PolyP1D(self.k_Mpc[i], p1d, deg=self.ndeg)
                training_label[i] = fit_p1d.P_Mpc(self.k_Mpc[i])
            self.yscalings = np.median(np.log(training_label))
        else:
            self.yscalings = np.median(training_label)

        return k_Mpc_train

    def _get_training_data_nn(self,
                              training_data: List[Dict]) -> torch.Tensor:
        """Get training data for neural network."""
        data = []
        for entry in training_data:
            data_dict = {}
            for param in self.emu_params:
                try:
                    value = entry["cosmo_params"][param]
                except:
                    value = entry[param]
                data_dict[param] = value
            data.append(data_dict)

        data = self._sort_dict(data, self.emu_params)
        data = np.array([list(d.values()) for d in data])
        
        data = (data - self.paramLims[:, 0]) / (self.paramLims[:, 1] - self.paramLims[:, 0]) - 0.5
        return torch.Tensor(data)

    def _get_training_cov(self,
                          training_data: List[Dict]) -> torch.Tensor:
        """Get training covariance matrix."""
        if self.emulator_label == "Nyx_alphap_cov":
            training_cov = []
            self.Y1_relerr = self._load_DESIY1_err()
            z_values = np.array(list(self.Y1_relerr.keys()))
            
            for entry in training_data:
                z = np.round(entry["z"], 1)
                if z not in self.Y1_relerr:
                    z = z_values[np.abs(z_values - z).argmin()]
                training_cov.append(self.Y1_relerr[z])
                
            training_cov = torch.Tensor(training_cov)
        else:
            training_cov = torch.zeros((len(training_data), len(self.k_Mpc[0])))
            
        return training_cov

    def _get_rescalings_weights(self,
                               training_data: List[Dict]) -> torch.Tensor:
        """Get rescaling weights."""
        weights = np.ones(len(training_data))
        if self.emulator_label == "Nyx_alphap_cov":
            mask = [(d['ind_rescaling'] not in [0,1] and d['z'] in [3,3.2]) for d in training_data]
            weights[np.where(mask)] = 1
        return torch.Tensor(weights)

    def _load_DESIY1_err(self) -> Dict[float, List[float]]:
        """Load DESI Y1 error data."""
        with open(self.models_dir / "DESI_cov" / self.filename_covariance) as f:
            return {float(z): rel_error for z, rel_error in json.load(f).items()}

    def _get_training_pd1_nn(self,
                             training_data: List[Dict]) -> torch.Tensor:
        """Get p1d_Mpc training data."""
        training_label = np.array([d["p1d_Mpc"][mask].tolist() for d, mask in zip(training_data, self.k_mask)])

        if self.emulator_label not in ["Cabayol23", "Cabayol23_extended"]:
            for i, p1d in enumerate(training_label):
                fit_p1d = poly_p1d.PolyP1D(self.k_Mpc[i], p1d, deg=self.ndeg)
                training_label[i] = fit_p1d.P_Mpc(self.k_Mpc[i])
            training_label = np.log(training_label) / self.yscalings**2
        else:
            training_label = np.log10(training_label / self.yscalings)

        return torch.Tensor(training_label)

    def _set_weights(self) -> torch.Tensor:
        """Set weights for loss function."""
        weights = torch.ones(self.Nk)
        if self.kmax_Mpc > 4 and self.weighted_emulator:
            self.print("Exponential downweighting loss function at k>4")
            exp_values = torch.linspace(0, 1.4, len(self.k_Mpc[0][self.k_Mpc[0] > 4]))
            weights[self.k_Mpc[0] > 4] = torch.exp(-exp_values)
        return weights

    def train(self, 
              training_data: List[Dict]) -> None:
        """Train the emulator."""
        k_Mpc_train = self._obtain_sim_params(training_data)
        weights = self._set_weights().to(self.device)
        log_k_Mpc = torch.log10(k_Mpc_train).to(self.device)
        self.log_kMpc = log_k_Mpc

        self.nn = nn_architecture.MDNemulator_polyfit(
            nhidden=self.nhidden,
            ndeg=self.ndeg,
            max_neurons=self.max_neurons,
            ninput=len(self.emu_params),
        )

        optimizer_class = optim.AdamW if self.emulator_label not in ["Cabayol23", "Cabayol23_extended"] else optim.Adam
        optimizer = optimizer_class(
            self.nn.parameters(),
            lr=self.lr0,
            weight_decay=self.weight_decay,
            amsgrad=getattr(self, 'amsgrad', False)
        )

        scheduler = lr_scheduler.StepLR(optimizer, self.step_size, gamma=0.1)

        training_data = self._get_training_data_nn(training_data)
        training_label = self._get_training_pd1_nn(training_data)
        training_cov = self._get_training_cov(training_data)
        weights_rescalings = self._get_rescalings_weights(training_data)

        dataset = TensorDataset(training_data, training_label, training_cov, weights_rescalings, log_k_Mpc)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.nn.to(self.device)
        self.print(f"Training NN on {len(training_data)} points")
        start_time = time.time()

        loss = torch.tensor(0.0)  # Initialize loss before the loop
        pbar = tqdm(range(self.nepochs), desc="Training epochs") 
        for epoch in pbar:
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            for batch in loader:
                optimizer.zero_grad()
                
                datain, p1D_true, P1D_desi_err, weights_rescalings, logkP1D = [x.to(self.device) for x in batch]

                coeffsPred, coeffs_logerr = self.nn(datain)
                coeffs_logerr = torch.clamp(coeffs_logerr, -10, 5)
                coeffserr = torch.exp(coeffs_logerr) ** 2

                powers = torch.arange(self.ndeg + 1).to(self.device)
                P1Dpred = torch.sum(
                    coeffsPred[:, powers, None] * (logkP1D[:, None, :] ** powers[None, :, None]),
                    axis=1
                )

                powers_err = torch.arange(0, self.ndeg * 2 + 1, 2).to(self.device)
                P1Derr = torch.sqrt(
                    torch.sum(
                        coeffserr[:, powers, None] * (logkP1D[:, None, :] ** powers_err[None, :, None]),
                        axis=1
                    )
                )

                P1Dlogerr = torch.log(torch.sqrt(P1Derr**2 + P1D_desi_err**2 * p1D_true**2))

                log_prob = (P1Dpred - p1D_true).pow(2) / (P1Derr**2 + P1D_desi_err**2 * p1D_true**2) + 2 * P1Dlogerr

                if self.emulator_label == "Nyx_alphap_cov":
                    log_prob = weights_rescalings[:, None] * log_prob
                else:
                    log_prob = weights[None, :] * log_prob

                loss = torch.nansum(log_prob, 1).nanmean()
                loss.backward()
                optimizer.step()

            scheduler.step()

        self.print(f"NN optimised in {time.time()-start_time:.1f} seconds")

    def save_emulator(self) -> None:
        """Save emulator to disk."""
        model_state = {
            "emulator": self.nn.state_dict(),
            "metadata": {
                "training_set": self.training_set,
                "emulator_label": self.emulator_label,
                "drop_sim": self.drop_sim,
                "drop_z": self.drop_z,
            }
        }

        if not self.emulator_label:
            save_path = self.save_path
        else:
            dir_path = PROJ_ROOT / "data" / "NNmodels" / self.emulator_label
            dir_path.mkdir(parents=True, exist_ok=True)
            self.print(f"Creating and using model directory: {dir_path}")
            
            filename = f"{self.emulator_label}"
            if self.drop_sim is not None:
                filename += f"_drop_sim_{self.drop_sim}_{self.drop_z}"
            filename += ".pt"
            
            save_path = dir_path / filename

        torch.save(model_state, save_path)

    def emulate_p1d_Mpc(
        self, 
        model: Dict[str, Union[float, np.ndarray]], 
        k_Mpc: np.ndarray,
        return_covar: bool = False,
        z: Optional[float] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Emulate P1D values.

        Args:
            model: Model parameters
            k_Mpc: k values in Mpc units
            return_covar: Whether to return covariance
            z: Redshift value

        Returns:
            Emulated P1D values and optionally covariance
        """
        self.nn.eval()

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
            warn(f"Some k values exceed maximum training value k={self.kmax_Mpc}")
        elif np.min(k_Mpc) < self.kmin_Mpc:
            warn(f"Some k values below minimum training value k={self.kmin_Mpc}")

        with torch.no_grad():
            emu_call = np.zeros((length, len(self.emu_params)))
            for i, param in enumerate(self.emu_params):
                emu_call[:, i] = model[param]

            emu_call = (emu_call - self.paramLims[None, :, 0]) / (
                self.paramLims[None, :, 1] - self.paramLims[None, :, 0]
            ) - 0.5

            emu_call = torch.Tensor(emu_call).unsqueeze(0).to(self.device)
            logk_Mpc = torch.log10(torch.Tensor(k_Mpc)).to(self.device)

            coeffsPred, coeffs_logerr = self.nn(emu_call)

            powers = torch.arange(self.ndeg + 1).to(self.device)
            emu_p1d = torch.sum(
                coeffsPred[0, :, :, None] * (logk_Mpc[:, None, :] ** powers[None, :, None]),
                axis=1
            )

            emu_p1d = emu_p1d.cpu().numpy()

            if self.emulator_label in ["Cabayol23", "Cabayol23_extended"]:
                emu_p1d = 10 ** emu_p1d * self.yscalings
            else:
                emu_p1d = np.exp(emu_p1d * self.yscalings**2)

            if emu_p1d.shape[0] == 1:
                emu_p1d = emu_p1d[0]

            if not return_covar:
                return emu_p1d

            coeffs_logerr = torch.clamp(coeffs_logerr, -10, 5)
            coeffserr = torch.exp(coeffs_logerr) ** 2
            
            powers_err = torch.arange(0, self.ndeg * 2 + 1, 2).to(self.device)
            emu_p1derr = torch.sqrt(
                torch.sum(
                    coeffserr[0, :, :, None] * (logk_Mpc[None, None, :] ** powers_err[None, :, None]),
                    axis=1
                )
            )

            emu_p1derr = emu_p1derr.cpu().numpy()

            if self.emulator_label in ["Cabayol23", "Cabayol23_extended"]:
                emu_p1derr = 10 ** emu_p1d * np.log(10) * emu_p1derr * self.yscalings
            else:
                emu_p1derr = np.exp(emu_p1d * self.yscalings**2) * self.yscalings**2 * emu_p1derr

            if emu_p1derr.shape[0] == 1:
                emu_p1derr = emu_p1derr[0]
                covar = np.outer(emu_p1derr, emu_p1derr)
            else:
                covar = np.array([np.outer(err, err) for err in emu_p1derr])

            return emu_p1d, covar

    def check_hull(self,
                   training_data: List[Dict]) -> None:
        """Create convex hull of training data."""
        training_points = [
            d for d in training_data 
            if d["ind_axis"] == "average" and d["ind_phase"] == "average"
        ]

        training_points = [
            {k: v for k, v in d.items() if k in self.emu_params}
            for d in training_points
        ]

        training_points = self._sort_dict(training_points, self.emu_params)
        training_points = np.array([list(d.values()) for d in training_points])

        self.hull = Delaunay(training_points)

    def test_hull(self, model_test: Dict) -> bool:
        """Test if point is in convex hull.
        
        Args:
            model_test: Parameter set to test

        Returns:
            Whether point is in hull
        """
        test_point = [model_test[param] for param in self.emu_params]
        return self.hull.find_simplex(test_point) >= 0
