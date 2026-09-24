Installation
============

LaCE requires Python 3.12 or newer. It is normally installed alongside its
scientific dependencies in a dedicated environment:

.. code-block:: console

   conda create -n lace python=3.12
   conda activate lace
   git clone https://github.com/igmhub/LaCE.git
   cd LaCE
   python -m pip install -e .

Install the documentation tools and build the site locally with:

.. code-block:: console

   python -m pip install -e ".[docs]"
   make docs

Run the local regression tests with:

.. code-block:: console

   make test

The suite covers the Pedersen21 archive, default cosmology reference values,
the packaged MP-Gadget emulator, and ``RescaledCosmology`` star parameters
compared with fresh CAMB calculations.

LaCE uses ``/global/cfs/cdirs/desi/science/lya/y1-p1d/likelihood_files/nyx_files``
as its default Nyx directory at NERSC. On another machine, configure the local
directory once from the command line:

.. code-block:: bash

   set_nyx_path /path/to/nyx_files

This stores the path in ``~/.config/lace/paths.toml``. A one-off archive can
instead receive ``nyx_path="/path/to/nyx_files"`` directly.


External assets and installed wheels
------------------------------------

Simulation suites and model bundles are intentionally external assets. This
keeps a normal wheel small and avoids distributing obsolete and L1O weights.
Set a persistent data root once (it must contain ``GPmodels/``,
``ff_mpgcen.npy``, and, if needed, ``sim_suites/``):

.. code-block:: python

   from lace.configuration import set_data_path
   set_data_path("/path/to/lace-data")

An explicit ``data_path`` takes precedence over this setting; an explicit
``model_path`` takes precedence for a single emulator. The NERSC Nyx default
and ``set_nyx_path`` interface are unchanged. ``h5py`` is installed with LaCE
for Nyx HDF5 archives.
