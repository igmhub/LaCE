"""Precision diagnostics for LaCE one-dimensional power-spectrum emulators.

The functions in this module retain the numerical definitions used in the
``Precision_emulators`` notebook while making the resulting figures and their
underlying numerical data reusable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import curve_fit

from lace.configuration import get_path_repo
from lace.emulator.gp_emulator_multi import GPEmulator

IGM_PARAMETERS = ("mF", "sigT_Mpc", "gamma", "kF_Mpc")


def complete_igm_parameters(
    data: Sequence[dict[str, Any]],
    reference_data: Sequence[dict[str, Any]],
    *,
    parameter_names: Sequence[str] = IGM_PARAMETERS,
    redshift_tolerance: float = 0.05,
) -> list[dict[str, Any]]:
    """Return copied data with missing IGM parameters filled from a reference.

    A replacement is announced for each affected parameter. This is intended
    for preparing incomplete testing data before validation; plotting itself
    never mixes parameters from different simulations.
    """
    completed_data = []
    for entry in data:
        completed_entry = entry.copy()
        matching_entries = [
            reference_entry
            for reference_entry in reference_data
            if abs(entry["z"] - reference_entry["z"]) < redshift_tolerance
        ]
        for parameter_name in parameter_names:
            if np.isfinite(completed_entry.get(parameter_name, np.nan)):
                continue
            if not matching_entries:
                raise ValueError(
                    f"No reference entry within {redshift_tolerance} of "
                    f"z={entry['z']:.3f} to replace missing {parameter_name}."
                )
            reference_value = matching_entries[0].get(parameter_name, np.nan)
            if not np.isfinite(reference_value):
                raise ValueError(
                    f"Reference data at z={matching_entries[0]['z']:.3f} also "
                    f"lack a finite {parameter_name}."
                )
            print(
                f"Testing data z={entry['z']:.3f}: replacing missing "
                f"{parameter_name} with reference value {reference_value}."
            )
            completed_entry[parameter_name] = reference_value
        completed_data.append(completed_entry)
    return completed_data


class EmulatorPrecisionPlotter:
    """Calculate and plot validation diagnostics for a P1D emulator.

    Parameters
    ----------
    emulator
        Loaded emulator used for the testing and smoothing diagnostics.
    archive, emulator_label
        Required only for leave-one-out validation, where one emulator is
        constructed for every excluded simulation.
    """

    def __init__(
        self,
        emulator: Any,
        *,
        archive: Any | None = None,
        emulator_label: str | None = None,
    ) -> None:
        self.emulator = emulator
        self.archive = archive
        self.emulator_label = emulator_label

    @staticmethod
    def _smooth_power(emulator: Any, entry: dict[str, Any], k_Mpc: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Fit the polynomial smoothing model used by the emulator diagnostics."""
        normalization = np.interp(
            k_Mpc, emulator.input_norm["k_Mpc"], emulator.norm_imF(entry["mF"])
        )
        log_power = np.log(np.asarray(entry["p1d_Mpc"])[mask] / normalization)
        fitted_parameters, _ = curve_fit(
            emulator.func_poly, k_Mpc / emulator.kmax_Mpc, log_power
        )
        return normalization * np.exp(emulator.func_poly(k_Mpc / emulator.kmax_Mpc, *fitted_parameters))

    @staticmethod
    def _k_grid(data: Sequence[dict[str, Any]], kmax_Mpc: float) -> tuple[np.ndarray, np.ndarray]:
        raw_k = np.asarray(data[0]["k_Mpc"])
        mask = (raw_k > 0) & (raw_k < kmax_Mpc)
        return raw_k[mask], mask

    def compute_testing_precision(
        self,
        testing_data: Sequence[dict[str, Any]],
        *,
        kmax_Mpc: float = 4.0,
    ) -> dict[str, np.ndarray | str]:
        """Compute ``P1D_smooth / P1D_emulator - 1`` at each valid redshift."""
        k_Mpc, mask = self._k_grid(testing_data, kmax_Mpc)
        redshifts, differences = [], []
        for entry in testing_data:
            missing_parameters = [
                name for name in IGM_PARAMETERS if not np.isfinite(entry.get(name, np.nan))
            ]
            if missing_parameters:
                raise ValueError(
                    f"Testing data at z={entry['z']:.3f} lack {missing_parameters}. "
                    "Prepare them with complete_igm_parameters before plotting."
                )
            emulator_power = np.asarray(self.emulator.emulate_p1d_Mpc(entry, k_Mpc)).reshape(-1, k_Mpc.size)[0]
            if not np.all(np.isfinite(emulator_power)) or np.any(emulator_power == 0):
                continue
            smooth_power = self._smooth_power(self.emulator, entry, k_Mpc, mask)
            redshifts.append(entry["z"])
            differences.append(smooth_power / emulator_power - 1.0)
        return {
            "k_Mpc": k_Mpc,
            "redshift": np.asarray(redshifts),
            "relative_difference": np.asarray(differences),
            "relative_difference_definition": "P1D_smooth / P1D_emulator - 1",
        }

    def compute_smoothing_precision(
        self,
        training_data: Sequence[dict[str, Any]],
        *,
        kmax_Mpc: float = 4.0,
    ) -> dict[str, np.ndarray | str]:
        """Compute percentile bands of ``P1D_simulation / P1D_smooth - 1``."""
        k_Mpc, mask = self._k_grid(training_data, kmax_Mpc)
        differences = []
        for entry in training_data:
            if not np.isfinite(entry.get("kF_Mpc", np.nan)):
                continue
            smooth_power = self._smooth_power(self.emulator, entry.copy(), k_Mpc, mask)
            differences.append(np.asarray(entry["p1d_Mpc"])[mask] / smooth_power - 1.0)
        if not differences:
            raise ValueError("No valid training samples were available for smoothing precision.")
        levels = np.array([5, 16, 84, 95])
        return {
            "k_Mpc": k_Mpc,
            "percentile_levels": levels,
            "relative_difference_percentiles": np.percentile(differences, levels, axis=0),
            "relative_difference_definition": "P1D_simulation / P1D_smooth - 1",
        }

    def compute_leave_one_out_precision(
        self,
        *,
        model_path: str | Path,
        testing_prefix: str | None = None,
        stop_simulation: str | None = None,
    ) -> dict[str, np.ndarray | str]:
        """Compute L1O precision using models from ``model_path``."""
        if self.archive is None or self.emulator_label is None:
            raise ValueError("archive and emulator_label are required for leave-one-out precision.")
        prefix = testing_prefix or ("nyx" if self.emulator_label.startswith("CH24_nyx") else "mpg")
        first_data = self.archive.get_testing_data(f"{prefix}_0")
        k_Mpc, mask = self._k_grid(first_data, self.emulator.kmax_Mpc)
        redshifts = np.asarray(self.archive.list_sim_redshifts)
        differences = []
        for simulation in self.archive.list_sim_cube:
            if simulation == stop_simulation:
                break
            testing_data = self.archive.get_testing_data(simulation)
            left_out_emulator = GPEmulator(
                emulator_label=self.emulator_label,
                archive=self.archive,
                train=False,
                drop_sim=simulation,
                model_path=model_path,
            )
            for entry in testing_data:
                required = ("kF_Mpc", "sigT_Mpc", "gamma")
                if not all(np.isfinite(entry.get(name, np.nan)) for name in required):
                    continue
                if not np.any(np.abs(redshifts - entry["z"]) < 0.05):
                    continue
                local_entry = entry.copy()
                emulator_power = np.asarray(left_out_emulator.emulate_p1d_Mpc(local_entry, k_Mpc)).reshape(-1, k_Mpc.size)[0]
                if not np.all(np.isfinite(emulator_power)) or np.any(emulator_power == 0):
                    continue
                smooth_power = self._smooth_power(left_out_emulator, local_entry, k_Mpc, mask)
                differences.append(emulator_power / smooth_power - 1.0)
        if not differences:
            raise ValueError("No valid leave-one-out samples were available.")
        levels = np.array([5, 16, 84, 95])
        return {
            "k_Mpc": k_Mpc,
            "percentile_levels": levels,
            "relative_difference_percentiles": np.percentile(differences, levels, axis=0),
            "relative_difference_definition": "P1D_emulator / P1D_smooth - 1",
        }

    @staticmethod
    def _save_zenodo(plot_data: dict[str, np.ndarray | str], *, save_zenodo: bool, zenodo_filename: str | None, zenodo_directory: str | Path | None) -> None:
        if not save_zenodo:
            return
        if zenodo_filename is None:
            raise ValueError("zenodo_filename is required when save_zenodo=True.")
        directory = Path(zenodo_directory) if zenodo_directory is not None else get_path_repo() / "data" / "zenodo"
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / zenodo_filename, plot_data, allow_pickle=True)

    @staticmethod
    def _finalize(axis: plt.Axes, *, fontsize: int, ylabel: str, xlim: tuple[float, float] | None, legend_kwargs: dict[str, Any]) -> None:
        axis.axhline(0.0, linestyle=":", color="k")
        axis.axhline(0.01, linestyle="--", color="k")
        axis.axhline(-0.01, linestyle="--", color="k")
        axis.set_xscale("log")
        axis.set_xlabel(r"$k_\parallel\,\left[\mathrm{Mpc}^{-1}\right]$", fontsize=fontsize)
        axis.set_ylabel(ylabel, fontsize=fontsize)
        if xlim is not None:
            axis.set_xlim(*xlim)
        axis.tick_params(axis="both", which="major", labelsize=fontsize - 2)
        axis.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        axis.legend(**legend_kwargs)
        axis.figure.tight_layout()

    def plot_testing_precision(self, testing_data: Sequence[dict[str, Any]], *, kmax_Mpc: float = 4.0, ax: plt.Axes | None = None, fontsize: int = 24, save_zenodo: bool = False, zenodo_filename: str | None = None, zenodo_directory: str | Path | None = None) -> tuple[plt.Figure, plt.Axes, dict[str, np.ndarray | str]]:
        """Plot redshift-resolved precision for self-contained testing data."""
        plot_data = self.compute_testing_precision(testing_data, kmax_Mpc=kmax_Mpc)
        if ax is None:
            figure, ax = plt.subplots(figsize=(8, 6))
        else:
            figure = ax.figure
        for redshift, difference in zip(plot_data["redshift"], plot_data["relative_difference"], strict=True):
            ax.plot(plot_data["k_Mpc"], difference, lw=2, label=rf"$z={redshift:.2f}$")
        self._finalize(ax, fontsize=fontsize, ylabel=r"$P_\mathrm{1D}^\mathrm{smooth}/P_\mathrm{1D}^\mathrm{emu}-1$", xlim=(0.08, kmax_Mpc), legend_kwargs={"loc": "upper left", "fontsize": fontsize - 5, "ncol": 3})
        self._save_zenodo(plot_data, save_zenodo=save_zenodo, zenodo_filename=zenodo_filename, zenodo_directory=zenodo_directory)
        return figure, ax, plot_data

    def _plot_percentile_precision(self, plot_data: dict[str, np.ndarray | str], *, ylabel: str, xlim: tuple[float, float] | None, ax: plt.Axes | None, fontsize: int, save_zenodo: bool, zenodo_filename: str | None, zenodo_directory: str | Path | None) -> tuple[plt.Figure, plt.Axes, dict[str, np.ndarray | str]]:
        if ax is None:
            figure, ax = plt.subplots(figsize=(8, 6))
        else:
            figure = ax.figure
        percentiles = plot_data["relative_difference_percentiles"]
        k_Mpc = plot_data["k_Mpc"]
        ax.fill_between(k_Mpc, percentiles[0], percentiles[-1], label="5-95th percentiles", color="C1", alpha=0.4)
        ax.fill_between(k_Mpc, percentiles[1], percentiles[2], label="16-84th percentiles", color="C0", alpha=0.4)
        self._finalize(ax, fontsize=fontsize, ylabel=ylabel, xlim=xlim, legend_kwargs={"fontsize": fontsize - 2, "ncol": 1})
        self._save_zenodo(plot_data, save_zenodo=save_zenodo, zenodo_filename=zenodo_filename, zenodo_directory=zenodo_directory)
        return figure, ax, plot_data

    def plot_smoothing_precision(self, training_data: Sequence[dict[str, Any]], *, kmax_Mpc: float = 4.0, ax: plt.Axes | None = None, fontsize: int = 24, save_zenodo: bool = False, zenodo_filename: str | None = None, zenodo_directory: str | Path | None = None) -> tuple[plt.Figure, plt.Axes, dict[str, np.ndarray | str]]:
        """Plot smooth-fit percentile precision and optionally export its data."""
        data = self.compute_smoothing_precision(training_data, kmax_Mpc=kmax_Mpc)
        return self._plot_percentile_precision(data, ylabel=r"$P_\mathrm{1D}^\mathrm{sim}/P_\mathrm{1D}^\mathrm{smooth}-1$", xlim=None, ax=ax, fontsize=fontsize, save_zenodo=save_zenodo, zenodo_filename=zenodo_filename, zenodo_directory=zenodo_directory)

    def plot_leave_one_out_precision(self, *, model_path: str | Path, testing_prefix: str | None = None, stop_simulation: str | None = None, ax: plt.Axes | None = None, fontsize: int = 24, save_zenodo: bool = False, zenodo_filename: str | None = None, zenodo_directory: str | Path | None = None) -> tuple[plt.Figure, plt.Axes, dict[str, np.ndarray | str]]:
        """Plot leave-one-out percentile precision and optionally export its data."""
        data = self.compute_leave_one_out_precision(model_path=model_path, testing_prefix=testing_prefix, stop_simulation=stop_simulation)
        return self._plot_percentile_precision(data, ylabel=r"$P_\mathrm{1D}^\mathrm{emu}/P_\mathrm{1D}^\mathrm{smooth}-1$", xlim=(0.08, 4.0), ax=ax, fontsize=fontsize, save_zenodo=save_zenodo, zenodo_filename=zenodo_filename, zenodo_directory=zenodo_directory)
