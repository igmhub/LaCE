"""Regression tests for the Gadget simulation archive."""

from lace.archive.gadget_archive import GadgetArchive


EXPECTED_PEDERSEN21_TEST_SIMULATIONS = [
    "mpg_central",
    "mpg_seed",
    "mpg_growth",
    "mpg_neutrinos",
    "mpg_curved",
    "mpg_running",
    "mpg_reio",
]


def test_pedersen21_test_simulations():
    """The Pedersen21 archive must expose its expected test simulations."""

    archive = GadgetArchive(postproc="Pedersen21")

    assert archive.list_sim_test == EXPECTED_PEDERSEN21_TEST_SIMULATIONS
