Cosmology
=========

``lace.cosmo`` provides the shared cosmology interface used by LaCE and its
downstream analyses. ``Cosmology`` evaluates the requested cosmology with
CAMB. ``RescaledCosmology`` reuses a fiducial cosmology when the background
parameters are fixed and only primordial linear-power parameters change.

Use ``RescaledCosmology`` for changes in ``As``, ``ns``, or ``nrun`` with the
same background. Analyses that change the background should construct a new
``Cosmology`` object so CAMB recomputes the expansion history and transfer
function. ``notebooks/cosmology/compare_rescaled_cosmology.py`` demonstrates
the fixed-background comparison against fresh CAMB calculations, including the
``Delta2_star``, ``n_star``, and ``alpha_star`` parameters used by cup1d.

The API reference documents the available linear-power and compressed-parameter
methods.

Named cosmologies
-----------------

``Cosmology(cosmo_label=...)`` accepts every label defined in
``lace.cosmo.labeled_cosmologies``: the Planck 2018 fiducials
(``Planck18_wonu``, the default, ``Planck18_wnu`` and ``Planck18_noBAO_wnu``),
reference variations around them (``Planck18_high_omh2``, ``Planck18_mnu03``,
``DESIDR2_ACT``, ...), and the simulation cosmologies: ``mpg_<sim>`` (read from
``sim_suites/Australia20/mpg_emu_cosmo.npy`` in the configured data path),
``nyx_<sim>`` (read from the configured Nyx path), ``accel2`` and ``sherwood``.
cup1d's ``set_cosmo`` uses the same labels.
