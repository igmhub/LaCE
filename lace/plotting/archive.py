"""Reusable visualizations for simulation archives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PARAMETER_LABELS = {
    "Delta2_p": r"$\Delta_p^2$", "n_p": r"$n_p$", "alpha_p": r"$\alpha_p$",
    "mF": r"$\bar{F}$", "T0": r"$T_0$ [K]", "sigT_Mpc": r"$\sigma_T$ [Mpc]",
    "sigT_kms": r"$\sigma_T$ [km/s]", "gamma": r"$\gamma$",
    "kF_Mpc": r"$k_F$ [1/Mpc]", "kF_kms": r"$k_F$ [s/km]",
    "tau_eff": r"$\tau_{\rm eff}$", "z": r"$z$", "f_p": r"$f_p$",
}


def parameter_matrix(data: Sequence[Mapping[str, Any]], parameters: Sequence[str]) -> np.ndarray:
    """Return finite scalar archive parameters with requested column order.

    Parameters
    ----------
    data : sequence of mapping
        Archive entries.
    parameters : sequence of str
        Names of scalar parameters to extract.

    Returns
    -------
    numpy.ndarray
        Matrix of shape ``(len(data), len(parameters))``.
    """
    rows = []
    for index, entry in enumerate(data):
        row = []
        for parameter in parameters:
            if parameter not in entry:
                raise KeyError(f"Archive entry {index} has no '{parameter}' parameter")
            value = np.asarray(entry[parameter])
            if value.size != 1:
                raise ValueError(f"Archive entry {index} parameter '{parameter}' must be scalar")
            scalar = float(value.reshape(-1)[0])
            if not np.isfinite(scalar):
                raise ValueError(f"Archive entry {index} parameter '{parameter}' is not finite")
            row.append(scalar)
        rows.append(row)
    return np.asarray(rows, dtype=float).reshape(len(data), len(parameters))


def _label(parameter: str, labels: Mapping[str, str] | None = None) -> str:
    return (labels or PARAMETER_LABELS).get(parameter, parameter)


def _save_figure(figure: plt.Figure, save_path: str | Path | None) -> None:
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, bbox_inches="tight")


def _axes_grid(n_panels: int, nrows: int, axes: Any = None) -> tuple[plt.Figure, np.ndarray]:
    if axes is None:
        ncols = int(np.ceil(n_panels / nrows))
        figure, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    else:
        axes = np.asarray(axes)
        figure = axes.reshape(-1)[0].figure
    flat_axes = np.asarray(axes).reshape(-1)
    if flat_axes.size < n_panels:
        raise ValueError("Not enough axes for the requested panels")
    return figure, flat_axes


class ArchivePlotter:
    """Create consistent diagnostic figures for a LaCE simulation archive.

    Parameters
    ----------
    archive : object, optional
        Archive exposing ``get_training_data``. Explicit entry dictionaries can
        also be supplied directly to every training-domain plotting method.
    """

    def __init__(self, archive: Any | None = None):
        self.archive = archive

    def get_training_data(
        self, parameters: Sequence[str], *, average: str | None = None
    ) -> list[dict[str, Any]]:
        """Return training entries for ``parameters`` from the attached archive.

        ``average="both"`` averages phase and line-of-sight-axis repetitions.
        It is appropriate for diagnostic figures, whereas emulator training
        normally uses the full, unaveraged data set.
        """
        if self.archive is None:
            raise ValueError("An archive is required when data is not supplied")
        return self.archive.get_training_data(
            emu_params=list(parameters), average=average
        )

    def plot_parameter_pair(self, x_parameter: str, y_parameter: str, *, data: Sequence[Mapping[str, Any]] | None = None, color_parameter: str | None = "z", ax: plt.Axes | None = None, label: str | None = None, marker: str = "o", color: str | None = None, cmap: str = "viridis", alpha: float = 1.0, size: float = 8, add_colorbar: bool = True, save_path: str | Path | None = None) -> tuple[plt.Figure, plt.Axes]:
        """Plot one parameter pair, optionally coloured by a third parameter.

        No files are written unless ``save_path`` is supplied. The returned
        figure and axes can be further customised by callers.
        """
        required = [x_parameter, y_parameter]
        if color_parameter is not None and color is None:
            required.append(color_parameter)
        entries = list(data) if data is not None else self.get_training_data(required)
        values = parameter_matrix(entries, required)
        if ax is None:
            figure, ax = plt.subplots()
        else:
            figure = ax.figure
        kwargs = {"s": size, "marker": marker, "alpha": alpha, "label": label}
        if color is None and color_parameter is not None:
            color_values = values[:, 2]
            artist = ax.scatter(values[:, 0], values[:, 1], c=color_values, cmap=cmap, **kwargs)
            if add_colorbar and np.ptp(color_values) > 0:
                figure.colorbar(artist, ax=ax, label=_label(color_parameter))
        else:
            ax.scatter(values[:, 0], values[:, 1], color=color, **kwargs)
        ax.set_xlabel(_label(x_parameter))
        ax.set_ylabel(_label(y_parameter))
        if label is not None:
            ax.legend()
        figure.tight_layout()
        _save_figure(figure, save_path)
        return figure, ax

    def plot_parameter_sequence(self, parameters: Sequence[str], *, data: Sequence[Mapping[str, Any]] | None = None, labels: Mapping[str, str] | None = None, nrows: int = 3, color: str | None = None, alpha: float = 1.0, size: float = 2, axes: Any = None, save_path: str | Path | None = None) -> tuple[plt.Figure, np.ndarray]:
        """Plot adjacent projections through a parameter domain.

        The parameters must be finite scalar values. Unused panels are hidden;
        saving is opt-in through ``save_path``.
        """
        if len(parameters) < 2:
            raise ValueError("At least two parameters are required")
        entries = list(data) if data is not None else self.get_training_data(parameters)
        matrix = parameter_matrix(entries, parameters)
        figure, flat_axes = _axes_grid(len(parameters) - 1, nrows, axes)
        for index, axis in enumerate(flat_axes[: len(parameters) - 1]):
            axis.scatter(matrix[:, index], matrix[:, index + 1], s=size, color=color, alpha=alpha)
            axis.set_xlabel(_label(parameters[index], labels))
            axis.set_ylabel(_label(parameters[index + 1], labels))
        for axis in flat_axes[len(parameters) - 1:]:
            axis.set_visible(False)
        figure.tight_layout()
        _save_figure(figure, save_path)
        return figure, flat_axes

    def plot_parameter_triplet(self, parameters: Sequence[str], *, data=None, cmap: str = "viridis", size: float = 8, ax=None, save_path=None):
        """Plot three archive parameters in 3D, coloured by redshift."""
        if len(parameters) != 3:
            raise ValueError("parameters must contain exactly three names")
        entries = list(data) if data is not None else self.get_training_data([*parameters, "z"])
        values = parameter_matrix(entries, [*parameters, "z"])
        if ax is None:
            figure = plt.figure()
            ax = figure.add_subplot(projection="3d")
        else:
            figure = ax.figure
        artist = ax.scatter(values[:, 0], values[:, 1], values[:, 2], c=values[:, 3], cmap=cmap, s=size)
        figure.colorbar(artist, ax=ax, label=_label("z"))
        ax.set_xlabel(_label(parameters[0])); ax.set_ylabel(_label(parameters[1])); ax.set_zlabel(_label(parameters[2]))
        figure.tight_layout(); _save_figure(figure, save_path)
        return figure, ax

    def compare_parameter_sequences(self, datasets: Mapping[str, Sequence[Mapping[str, Any]]] | Sequence[Sequence[Mapping[str, Any]]], parameters: Sequence[str], *, labels: Sequence[str] | None = None, parameter_labels: Mapping[str, str] | None = None, colors: Sequence[str] | None = None, alphas: Sequence[float] | None = None, sizes: Sequence[float] | None = None, nrows: int = 3, axes: Any = None, save_path: str | Path | None = None) -> tuple[plt.Figure, np.ndarray]:
        """Overlay adjacent parameter projections from several data sets."""
        if isinstance(datasets, Mapping):
            names, data_sets = list(datasets), list(datasets.values())
        else:
            data_sets = list(datasets)
            names = list(labels) if labels is not None else [f"dataset {index + 1}" for index in range(len(data_sets))]
        if len(names) != len(data_sets) or len(parameters) < 2:
            raise ValueError("Provide matching datasets and labels and at least two parameters")
        colors = list(colors or [f"C{index}" for index in range(len(data_sets))])
        alphas = list(alphas or [0.5] * len(data_sets))
        sizes = list(sizes or [2.0] * len(data_sets))
        if not (len(colors) == len(alphas) == len(sizes) == len(data_sets)):
            raise ValueError("colors, alphas, and sizes must match the number of datasets")
        matrices = [parameter_matrix(entries, parameters) for entries in data_sets]
        figure, flat_axes = _axes_grid(len(parameters) - 1, nrows, axes)
        for index, axis in enumerate(flat_axes[: len(parameters) - 1]):
            for name, matrix, color, alpha, size in zip(names, matrices, colors, alphas, sizes, strict=True):
                axis.scatter(matrix[:, index], matrix[:, index + 1], s=size, color=color, alpha=alpha, label=name)
            axis.set_xlabel(_label(parameters[index], parameter_labels))
            axis.set_ylabel(_label(parameters[index + 1], parameter_labels))
        flat_axes[0].legend(markerscale=2)
        for axis in flat_axes[len(parameters) - 1:]:
            axis.set_visible(False)
        figure.tight_layout()
        _save_figure(figure, save_path)
        return figure, flat_axes

    def plot_p1d_dependence(
        self,
        parameter: str,
        *,
        data: Sequence[Mapping[str, Any]] | None = None,
        average: str | None = "both",
        k_min_Mpc: float = 0.0,
        k_max_Mpc: float = 10.0,
        log_y: bool = True,
        cmap: str = "viridis",
        alpha: float = 0.25,
        max_curves: int | None = 300,
        label_extrema: bool = True,
        add_colorbar: bool = True,
        ax: plt.Axes | None = None,
        save_path: str | Path | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the P1D training spectra coloured by one scalar parameter.

        Parameters
        ----------
        parameter
            Scalar archive parameter controlling the line colours.
        data
            Entries containing ``parameter``, ``k_Mpc``, and ``p1d_Mpc``.
            When omitted, data are requested from the attached archive.
        average
            Archive phase/axis averaging passed to ``get_training_data`` when
            ``data`` is omitted. The default ``"both"`` avoids plotting
            repeated spectra that are only needed during emulator training.
        k_min_Mpc, k_max_Mpc
            Open interval in comoving wavenumber, in ``1/Mpc``.
        log_y
            Use a logarithmic vertical axis as well as a logarithmic
            horizontal axis.
        max_curves
            Maximum number of spectra to draw. For larger archives, entries
            are sampled deterministically across the sorted parameter range,
            including both extrema. Set to ``None`` to draw every spectrum.
        label_extrema
            Label the spectra with the minimum and maximum parameter values.
        save_path
            Optional output path. No file is written by default.

        Returns
        -------
        matplotlib.figure.Figure, matplotlib.axes.Axes
            Figure and axes containing ``k P1D(k)``. This product is
            dimensionless for ``k`` in ``1/Mpc`` and ``P1D`` in ``Mpc``.
        """
        if k_min_Mpc < 0 or k_max_Mpc <= k_min_Mpc:
            raise ValueError("Require 0 <= k_min_Mpc < k_max_Mpc")
        entries = (
            list(data)
            if data is not None
            else self.get_training_data([parameter], average=average)
        )
        parameter_values = parameter_matrix(entries, [parameter])[:, 0]
        if max_curves is not None and max_curves < 2:
            raise ValueError("max_curves must be at least 2 or None")
        value_min = float(np.min(parameter_values))
        value_max = float(np.max(parameter_values))
        if value_min == value_max:
            normalization = plt.Normalize(value_min - 0.5, value_max + 0.5)
        else:
            normalization = plt.Normalize(value_min, value_max)
        colormap = plt.get_cmap(cmap)
        if ax is None:
            figure, ax = plt.subplots()
        else:
            figure = ax.figure

        extrema = {int(np.argmin(parameter_values)), int(np.argmax(parameter_values))}
        selected_indices = np.arange(len(entries))
        if max_curves is not None and len(entries) > max_curves:
            sorted_indices = np.argsort(parameter_values, kind="stable")
            sample_positions = np.linspace(
                0, len(sorted_indices) - 1, max_curves, dtype=int
            )
            selected_indices = sorted_indices[sample_positions]
        plotted = 0
        for index in selected_indices:
            entry = entries[index]
            value = parameter_values[index]
            if "k_Mpc" not in entry or "p1d_Mpc" not in entry:
                raise KeyError(
                    f"Archive entry {index} must contain 'k_Mpc' and 'p1d_Mpc'"
                )
            k_Mpc = np.asarray(entry["k_Mpc"], dtype=float)
            p1d_Mpc = np.asarray(entry["p1d_Mpc"], dtype=float)
            if k_Mpc.shape != p1d_Mpc.shape:
                raise ValueError(
                    f"Archive entry {index} has inconsistent k_Mpc and p1d_Mpc shapes"
                )
            mask = (
                np.isfinite(k_Mpc)
                & np.isfinite(p1d_Mpc)
                & (k_Mpc > k_min_Mpc)
                & (k_Mpc < k_max_Mpc)
                & (k_Mpc > 0)
            )
            if log_y:
                mask &= k_Mpc * p1d_Mpc > 0
            if not np.any(mask):
                continue
            label = None
            if label_extrema and index in extrema:
                label = f"{_label(parameter)} = {value:.4g}"
            ax.plot(
                k_Mpc[mask],
                k_Mpc[mask] * p1d_Mpc[mask],
                color=colormap(normalization(value)),
                alpha=alpha,
                label=label,
            )
            plotted += 1
        if plotted == 0:
            raise ValueError("No finite P1D samples fall inside the requested k range")
        ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel(r"$k_\parallel$ [1/Mpc]")
        ax.set_ylabel(r"$k_\parallel P_{\rm 1D}(k_\parallel)$")
        ax.set_title(r"$P_{\rm 1D}$ dependence on " + _label(parameter))
        if label_extrema:
            ax.legend()
        if add_colorbar:
            scalar_mappable = plt.cm.ScalarMappable(
                norm=normalization, cmap=colormap
            )
            figure.colorbar(
                scalar_mappable, ax=ax, label=_label(parameter)
            )
        figure.tight_layout()
        _save_figure(figure, save_path)
        return figure, ax

    @staticmethod
    def plot_igm_histories(histories: Mapping[str, Mapping[str, Any]], *, parameters: Sequence[str] | None = None, labels: Mapping[str, str] | None = None, highlighted_simulations: Sequence[str] | None = None, excluded_simulations: Sequence[str] | None = None, default_color: str = "black", highlight_color: str = "red", default_alpha: float = 0.2, highlight_alpha: float = 1.0, axes: Any = None, save_path: str | Path | None = None) -> tuple[plt.Figure, np.ndarray]:
        """Plot IGM histories, masking zero values marked unavailable in files."""
        parameters = list(parameters or ["tau_eff", "gamma", "sigT_kms", "kF_kms"])
        highlighted, excluded = set(highlighted_simulations or []), set(excluded_simulations or [])
        figure, flat_axes = _axes_grid(len(parameters), 2, axes)
        for simulation, history in histories.items():
            if simulation in excluded:
                continue
            color = highlight_color if simulation in highlighted else default_color
            alpha = highlight_alpha if simulation in highlighted else default_alpha
            redshift = np.asarray(history["z"])
            for axis, parameter in zip(flat_axes, parameters, strict=True):
                values = np.asarray(history[parameter])
                mask = np.isfinite(redshift) & np.isfinite(values) & (values != 0)
                axis.plot(redshift[mask], values[mask], color=color, alpha=alpha)
        for index, axis in enumerate(flat_axes[:len(parameters)]):
            axis.set_ylabel(_label(parameters[index], labels))
            if index >= len(parameters) - 2:
                axis.set_xlabel(_label("z", labels))
        for axis in flat_axes[len(parameters):]:
            axis.set_visible(False)
        figure.tight_layout()
        _save_figure(figure, save_path)
        return figure, flat_axes

    @staticmethod
    def compare_igm_histories(history_sets: Mapping[str, Mapping[str, Mapping[str, Any]]], *, dataset_labels: Sequence[str] | None = None, parameters: Sequence[str] | None = None, parameter_labels: Mapping[str, str] | None = None, colors: Sequence[str] | None = None, alphas: Sequence[float] | None = None, excluded_simulations: Sequence[str] | None = None, rescaling_datasets: Sequence[str] = ("Nyx",), base_rescaling_index: int = 0, axes: Any = None, save_path: str | Path | None = None) -> tuple[plt.Figure, np.ndarray]:
        """Compare IGM histories from named data sets, masking unavailable zeros.

        For data sets listed in ``rescaling_datasets``, simulation labels that
        end in an integer rescaling index are interpreted using
        ``base_rescaling_index``. The base histories and named test simulations
        are drawn as lines; other rescalings are shown as unconnected dots to
        avoid overcrowding.
        """
        parameters = list(parameters or ["tau_eff", "gamma", "sigT_kms", "kF_kms"])
        names, sets = list(history_sets), list(history_sets.values())
        if dataset_labels is not None:
            names = list(dataset_labels)
        if len(names) != len(sets):
            raise ValueError("dataset_labels and history_sets must have the same length")
        colors = list(colors or [f"C{index}" for index in range(len(sets))])
        alphas = list(alphas or [0.2] * len(sets))
        if len(colors) != len(sets) or len(alphas) != len(sets):
            raise ValueError("colors and alphas must match the number of history sets")
        excluded = set(excluded_simulations or [])
        rescaling_names = {name.casefold() for name in rescaling_datasets}
        figure, flat_axes = _axes_grid(len(parameters), 2, axes)
        for name, histories, color, alpha in zip(names, sets, colors, alphas, strict=True):
            for simulation, history in histories.items():
                if simulation in excluded:
                    continue
                match = re.search(r"_(\d+)$", simulation)
                is_rescaling = (
                    name.casefold() in rescaling_names
                    and match is not None
                    and int(match.group(1)) != base_rescaling_index
                )
                redshift = np.asarray(history["z"])
                for axis, parameter in zip(flat_axes, parameters, strict=True):
                    values = np.asarray(history[parameter])
                    mask = np.isfinite(redshift) & np.isfinite(values) & (values != 0)
                    if is_rescaling:
                        axis.plot(
                            redshift[mask],
                            values[mask],
                            linestyle="none",
                            marker=".",
                            markersize=2,
                            color=color,
                            alpha=alpha,
                        )
                    else:
                        axis.plot(
                            redshift[mask],
                            values[mask],
                            color=color,
                            alpha=alpha,
                        )
        for index, axis in enumerate(flat_axes[:len(parameters)]):
            axis.set_ylabel(_label(parameters[index], parameter_labels))
            if index >= len(parameters) - 2:
                axis.set_xlabel(_label("z", parameter_labels))
        legend_handles = [
            plt.Line2D([], [], color=color, label=name)
            for name, color in zip(names, colors, strict=True)
        ]
        flat_axes[0].legend(handles=legend_handles)
        for axis in flat_axes[len(parameters):]:
            axis.set_visible(False)
        figure.tight_layout()
        _save_figure(figure, save_path)
        return figure, flat_axes
