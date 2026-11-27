"""House style of the IGMHub figures (STIX fonts).

LaCE never changes Matplotlib settings at import time: every plotting entry
point of :mod:`lace.plotting` calls :func:`apply_house_style` explicitly.
"""

from contextlib import contextmanager

import matplotlib as mpl

#: Matplotlib settings of the house style.
HOUSE_STYLE = {"font.family": "STIXGeneral", "mathtext.fontset": "stix"}


def apply_house_style():
    """Set the house style (STIX text and math fonts) in ``rcParams``."""
    mpl.rcParams.update(HOUSE_STYLE)


@contextmanager
def house_style():
    """Context manager applying the house style only inside the block.

    Tick labels are created when a figure is drawn, so draw or save the figure
    inside the block.
    """
    with mpl.rc_context(HOUSE_STYLE):
        yield
