Emulators
=========

The supported emulator factory is ``lace.emulator.set_emulator(emulator_label)``.
Only ``CH24_mpgcen_gpr`` (MP-Gadget suite, central simulation included) and
``CH24_nyxcen_gpr`` (Nyx suite, central simulation included) are supported;
cup1d calls them ``lace_mpg`` and ``lace_nyx``. Both are
:class:`lace.emulator.gp_emulator_multi.GPEmulator` instances that predict the
flux P1D in Mpc units from linear-power (``Delta2_p``, ``n_p`` and, for Nyx,
``alpha_p``) and IGM (``mF``, ``sigT_Mpc``, ``gamma``, ``kF_Mpc``) parameters.
``GPEmulator`` also trains the variants without the central simulation
(``CH24_mpg_gpr``, ``CH24_nyx_gpr``); their bundles are not distributed.
The neural-network emulators of earlier releases were removed (see the Git
history); the training workflow is ``notebooks/emulator/training_emulator.py``.

Emulator errors
---------------

The leave-one-out relative covariance of each supported emulator is stored as
``data/covariance/l1O_cov_<emulator_label>.npy`` and read by cup1d to add the
emulator error to the likelihood. ``notebooks/emulator/compute_cov.py``
regenerates it.

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

Newly trained bundles include one JSON manifest per full or leave-one-out
model, checked before normalization data, NumPy object arrays, and pickle
checkpoints are read. It records schema, identity, excluded simulation,
provenance, LaCE/Python/dependency versions, and checksums.
Checksums detect damaged trusted bundles; they do not make arbitrary pickles
safe. Load weights only from a trusted source. Existing full CH24 bundles use
the documented legacy policy: they remain usable but their historical training
provenance is unknown. LaCE pins scikit-learn because pickle compatibility is
not guaranteed across releases. Retrain or migrate a model and numerically
validate representative predictions before changing the supported pin.
