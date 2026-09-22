"""Backward-compatible imports for plotting helpers.

New code should import from :mod:`lace.plotting`.
"""

from lace.plotting.corner import plot_parameter_corner
from lace.plotting.emulator import plot_p1d_vs_emulator


def create_corner_plot(list_of_dfs, params_to_plot, **kwargs):
    """Compatibility wrapper around :func:`lace.plotting.plot_parameter_corner`."""
    samples = [dataframe.loc[:, params_to_plot].to_numpy() for dataframe in list_of_dfs]
    return plot_parameter_corner(
        samples,
        labels=kwargs.pop("labels", params_to_plot),
        dataset_labels=kwargs.pop("legend_labels", None),
        colors=kwargs.pop("colors", None),
        truth_values=kwargs.pop("truth_values", None),
        save_path=kwargs.pop("save_path", None),
    )
