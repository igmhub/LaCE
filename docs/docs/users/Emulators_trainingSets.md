# PREDEFINED EMULATORS AND TRAINING SETS

## PREDEFINED EMULATORS
LaCE provides a set of predefined emulators that have been validated. These emulators are:

- Neural network emulators:
    - Gadget emulators: 
        - Cabayol23: Neural network emulating the optimal P1D of Gadget simulations fitting coefficients to a 5th degree polynomial. It goes to scales of 4Mpc^{-1} and z<=4.5.
        - Cabayol23+: Neural network emulating the optimal P1D of Gadget simulations fitting coefficients to a 5th degree polynomial. It goes to scales of 4Mpc^{-1} and z<=4.5. Updated version compared to Cabayol+23 paper.
        - Cabayol23_extended: Neural network emulating the optimal P1D of Gadget simulations fitting coefficients to a 7th degree polynomial. It goes to scales of 8Mpc^{-1} and z<=4.5.
        - Cabayol23+_extended: Neural network emulating the optimal P1D of Gadget simulations fitting coefficients to a 5th degree polynomial. It goes to scales of 4Mpc^{-1} and z<=4.5. Updated version compared to Cabayol+23 paper.
    - Nyx emulators:
        - Nyx_v0: Neural network emulating the optimal P1D of Nyx simulations fitting coefficients to a 6th degree polynomial. It goes to scales of 4Mpc^{-1} and z<=4.5.
        - Nyx_v0_extended: Neural network emulating the optimal P1D of Nyx simulations fitting coefficients to a 6th degree polynomial. It goes to scales of 8Mpc^{-1} and z<=4.5.
        - Nyx_alphap: Neural network emulating the optimal P1D of Nyx simulations fitting coefficients to a 6th degree polynomial. It goes to scales of 4Mpc^{-1} and z<=4.5.
        - Nyx_alphap_extended: Neural network emulating the optimal P1D of Nyx simulations fitting coefficients to a 6th degree polynomial. It goes to scales of 8Mpc^{-1} and z<=4.5.
        - Nyx_alphap_cov: Neural network under testing for the Nyx_alphap emulator.
    
- Gaussian Process emulators:
    - Gadget emulators:
        - "Pedersen21": Gaussian process emulating the optimal P1D of Gadget simulations. Pedersen+21 paper.
        - "Pedersen23": Updated version of Pedersen21 emulator. Pedersen+23 paper.
        - "Pedersen21_ext": Extended version of Pedersen21 emulator.
        - "Pedersen21_ext8": Extended version of Pedersen21 emulator up to k=8 Mpc^-1.
        - "Pedersen23_ext": Extended version of Pedersen23 emulator.
        - "Pedersen23_ext8": Extended version of Pedersen23 emulator up to k=8 Mpc^-1.

## PREDEFINED TRAINING SETS

Similarly, LaCE provides a set of predefined training sets that have been used to train the emulators. These training sets correspond to a simulations suite, a postprocessing and the addition (or not) of mean flux rescalings. The training sets are:

- "Pedersen21": Training set used in [Pedersen+21 paper](https://arxiv.org/abs/2103.05195). Gadget simulations without mean flux rescalings.
- "Cabayol23": Training set used in [Cabayol+23 paper](https://arxiv.org/abs/2303.05195). Gadget simulations with mean flux rescalings and measuring the P1D along the three principal axes of the simulation box.
- "Nyx_Oct2023": Training set using Nyx version from October 2023.
- "Nyx_Jul2024": Training set using Nyx version from July 2024.

## CONNECTION BETWEEN PREDEFINED EMULATORS AND TRAINING SETS
The following table shows the default training set for each predefined emulator.

| Emulator | Training Set | Simulation | Type | Description |
|----------|--------------|------------|------|-------------|
| Cabayol23 | Cabayol23 | Gadget | NN | Neural network emulator trained on Gadget simulations with mean flux rescaling |
| Cabayol23+ | Cabayol23 | Gadget | NN | Updated version of Cabayol23 emulator |
| Cabayol23_extended | Cabayol23 | Gadget | NN | Extended version of Cabayol23 emulator (k up to 8 Mpc^-1) |
| Cabayol23+_extended | Cabayol23 | Gadget | NN | Extended version of Cabayol23+ emulator (k up to 8 Mpc^-1) |
| Nyx_v0 | Nyx_Oct2023 | Nyx | NN | Neural network emulator trained on Nyx simulations |
| Nyx_v0_extended | Nyx_Oct2023 | Nyx | NN | Extended version of Nyx_v0 emulator (k up to 8 Mpc^-1) |
| Nyx_alphap | Nyx_Oct2023 | Nyx | NN | Neural network emulator trained on updated Nyx simulations |
| Nyx_alphap_extended | Nyx_Oct2023 | Nyx | NN | Extended version of Nyx_alphap emulator (k up to 8 Mpc^-1) |
| Nyx_alphap_cov | Nyx_Jul2024 | Nyx | NN | Testing version of Nyx_alphap emulator |
| Pedersen21 | Pedersen21 | Gadget | GP | GP emulator trained on Gadget simulations without mean flux rescaling |
| Pedersen23 | Pedersen21 | Gadget | GP | Updated version of Pedersen21 GP emulator |
| Pedersen21_ext | Pedersen21 | Gadget | GP | Extended version of Pedersen21 GP emulator |
| Pedersen21_ext8 | Pedersen21 | Gadget | GP | Extended version of Pedersen21 GP emulator (k up to 8 Mpc^-1) |
| Pedersen23_ext | Pedersen21 | Gadget | GP | Extended version of Pedersen23 GP emulator |
| Pedersen23_ext8 | Pedersen21 | Gadget | GP | Extended version of Pedersen23 GP emulator (k up to 8 Mpc^-1) |