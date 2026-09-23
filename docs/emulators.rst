Emulators
=========

The supported emulator factory is ``lace.emulator.set_emulator(emulator_label)``.
Only ``CH24_mpgcen_gpr`` and ``CH24_nyxcen_gpr`` are supported. Legacy NN
and GP implementations live in ``lace.emulator.old_nn_emulator`` and
``lace.emulator.old_gp_emulator`` for historical training workflows; they are
not part of the production API.
They predict the flux P1D from cosmological and IGM parameters.

The emulator training range, parameter conventions, and returned P1D units
are defined by the selected training archive. Requesting wavenumbers outside
the training range may require extrapolation and should be treated carefully.


Model assets, compatibility, and trust
--------------------------------------

Use external assets with an installed wheel, for example:

.. code-block:: python

   from lace.emulator import set_emulator
   emulator = set_emulator("CH24_mpgcen_gpr", model_path="/data/lace/GPmodels/CH24_mpgcen_gpr")

The associated ``ff_mpgcen.npy`` is resolved beside the external ``GPmodels``
root, from ``data_path``, or explicitly with ``normalization_path``. Loading
never creates directories; training creates its requested output directory.
L1O weights are historical validation assets and are never distributed: the
training and validation notebooks require an explicit, non-blank path.

Newly trained bundles include a JSON manifest, checked before NumPy object
arrays and pickle checkpoints are read. It records schema, identity, excluded
simulation, provenance, LaCE/Python/dependency versions, and checksums.
Checksums detect damaged trusted bundles; they do not make arbitrary pickles
safe. Load weights only from a trusted source. Existing full CH24 bundles use
the documented legacy policy: they remain usable but their historical training
provenance is unknown. LaCE pins scikit-learn because pickle compatibility is
not guaranteed across releases. Retrain or migrate a model and numerically
validate representative predictions before changing the supported pin.
