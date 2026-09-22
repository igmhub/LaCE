Simulation archives
===================

LaCE supplies archive interfaces for the MPG Gadget and Nyx simulation suites.
Archives provide training and test snapshots, their cosmological and IGM
parameters, and P1D measurements. The concrete archive classes are
``GadgetArchive`` and ``NyxArchive``.

Nyx data is external to the Python package. LaCE uses the NERSC DESI location
by default and can persist a local replacement in ``~/.config/lace/paths.toml``
with ``lace.configuration.set_nyx_path``. Gadget and emulator assets are
resolved from the repository data directory in the current release.
