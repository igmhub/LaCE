"""Plotting helpers for LaCE archives and emulators."""

from lace.plotting.archive import ArchivePlotter, PARAMETER_LABELS
from lace.plotting.corner import plot_parameter_corner
from lace.plotting.covariance import (
    plot_l1o_bias,
    plot_l1o_correlation,
    plot_l1o_covariance_robustness,
    plot_l1o_errors,
)
from lace.plotting.emulator import plot_emulator_predictions, plot_p1d_vs_emulator
from lace.plotting.emulator_precision import (
    EmulatorPrecisionPlotter,
    complete_igm_parameters,
)

__all__ = [
    "ArchivePlotter",
    "PARAMETER_LABELS",
    "EmulatorPrecisionPlotter",
    "complete_igm_parameters",
    "plot_emulator_predictions",
    "plot_p1d_vs_emulator",
    "plot_parameter_corner",
    "plot_l1o_correlation",
    "plot_l1o_bias",
    "plot_l1o_covariance_robustness",
    "plot_l1o_errors",
]
