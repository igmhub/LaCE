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

   python -m pip install -e ".[test]"
   pytest -q

LaCE uses ``/global/cfs/cdirs/desi/science/lya/y1-p1d/likelihood_files/nyx_files``
as its default Nyx directory at NERSC. On another machine, configure the local
directory once from the command line:

.. code-block:: bash

   set_nyx_path /path/to/nyx_files

This stores the path in ``~/.config/lace/paths.toml``. A one-off archive can
instead receive ``nyx_path="/path/to/nyx_files"`` directly.
