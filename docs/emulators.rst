Emulators
=========

The supported emulator factory is ``lace.emulator.emulator_manager.set_emulator``.
Current production labels are ``CH24_mpgcen_gpr`` and ``CH24_nyxcen_gpr``.
They predict the flux P1D from cosmological and IGM parameters.

The emulator training range, parameter conventions, and returned P1D units
are defined by the selected training archive. Requesting wavenumbers outside
the training range may require extrapolation and should be treated carefully.
