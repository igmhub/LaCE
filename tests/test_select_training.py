"""Regression tests for training-data selection."""

from lace.emulator.constants import TrainingSet
from lace.emulator import select_training


class _RecordingArchive:
    """Minimal archive recording construction and data-selection arguments."""

    def __init__(self, **kwargs):
        self.constructor_kwargs = kwargs
        self.training_kwargs = None

    def get_training_data(self, **kwargs):
        self.training_kwargs = kwargs
        return ["selected"]


def test_named_training_set_forwards_drop_z(monkeypatch):
    """Named archives must honor the requested held-out redshift."""

    monkeypatch.setattr(
        select_training.gadget_archive, "GadgetArchive", _RecordingArchive
    )
    archive, training_data = select_training.select_training(
        archive=None,
        training_set=TrainingSet.CABAYOL23,
        emu_params=["mF"],
        drop_sim=["mpg_0"],
        drop_z=[3.0],
        include_central=False,
        z_max=4.2,
        average="both",
    )

    assert training_data == ["selected"]
    assert archive.training_kwargs == {
        "emu_params": ["mF"],
        "drop_sim": ["mpg_0"],
        "drop_z": [3.0],
        "z_max": 4.2,
        "average": "both",
    }
