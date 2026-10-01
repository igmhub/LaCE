"""Regression tests for the Gadget simulation archive."""

from pathlib import Path

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


def test_cabayol23_fixp3d_uses_corrected_training_files_and_available_tests():
    """The corrected archive keeps five training rescalings and legacy tests."""
    archive = GadgetArchive(postproc="Cabayol23_fixp3d")

    training_files, _ = archive._get_file_names("mpg_0", 0, 0, 0)
    central_files, _ = archive._get_file_names("mpg_central", 0, 0, 0)
    seed_files, _ = archive._get_file_names("mpg_seed", 0, 0, 0)

    assert [Path(name).name.split("_0_")[0] for name in training_files] == [
        "p1d_reshaped",
        "p1d_reshaped_stau",
    ]
    assert [Path(name).name.split("_0_")[0] for name in central_files] in (
        ["p1d_reshaped", "p1d_reshaped_stau"],
        ["p1d_stau", "p1d_setau"],
    )
    assert [Path(name).name.split("_0_")[0] for name in seed_files] in (
        ["p1d_reshaped", "p1d_reshaped_stau"],
        ["p1d_stau"],
    )

    training = [
        item
        for item in archive.data
        if item["sim_label"] == "mpg_0"
        and item["ind_snap"] == 0
        and item["ind_phase"] == 0
        and item["ind_axis"] == 0
    ]
    assert len(training) == 5
    assert sorted(item["ind_rescaling"] for item in training) == list(range(5))
