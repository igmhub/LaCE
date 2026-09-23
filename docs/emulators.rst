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
