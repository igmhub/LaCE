# Making predictions with LaCE

## Loading an emulator 
The easiest way to load an emulator is to use the `set_emulator` class. This can be done with an [archive](archive.md) or a training set. This will load a trained emulator with the specifications of the emulator label.

```python
archive = gadget_archive.GadgetArchive(postproc="Pedersen21")
emulator = set_emulator(
        emulator_label="Pedersen23_ext",
        archive=archive,
    )
```

```python
archive = gadget_archive.GadgetArchive(postproc="Pedersen21")
emulator = set_emulator(
        emulator_label="Pedersen23_ext",
        training_set="Pedersen21",
    )
```

The supported emulators are: 
    - "Pedersen21": Gaussian process emulating the optimal P1D of Gadget simulations. Pedersen+21 paper.
    - "Pedersen23": Updated version of Pedersen21 emulator. Pedersen+23 paper.
    - "Pedersen21_ext": Extended version of Pedersen21 emulator.
    - "Pedersen21_ext8": Extended version of Pedersen21 emulator up to k=8 Mpc^-1.
    - "Pedersen23_ext": Extended version of Pedersen23 emulator.
    - "Pedersen23_ext8": Extended version of Pedersen23 emulator up to k=8 Mpc^-1.
    - "CH24": Emulator from Chabanier & Haehnelt 2024 paper.
    - "Cabayol23": Neural network emulating the optimal P1D of Gadget simulations from Cabayol+23 paper.
    - "Cabayol23+": Updated version of Cabayol23 emulator.
    - "Cabayol23_extended": Extended version of Cabayol23 emulator to k=8 Mpc^-1.
    - "Cabayol23+_extended": Extended version of Cabayol23+ emulator to k=8 Mpc^-1.
    - "Nyx_v0": Neural network emulating the optimal P1D of Nyx simulations.
    - "Nyx_v0_extended": Extended version of Nyx_v0 emulator to k=8 Mpc^-1.
    - "Nyx_alphap": Nyx emulator including alpha_p parameter.
    - "Nyx_alphap_extended": Extended version of Nyx_alphap emulator to k=8 Mpc^-1.
    - "Nyx_alphap_cov": Nyx emulator under testing using the covariance matrix from DESI Y1.

Another option is to load an emulator model that does not correspond to a predifined emulator label. This can be done by, for example:

```python
emulator = NNEmulator(training_set='Cabayol23', 
            emulator_label='Cabayol23+',
            model_path='path/to/model.pt',
            drop_sim=None,
            train=False)
```
where `model_path` is the path to the `.pt` file containing the trained model and `train=False` indicates that the model is not being trained. In the model you are loading has been trained by dropping simulations, you should specify which simulations to drop using the `drop_sim` argument.

