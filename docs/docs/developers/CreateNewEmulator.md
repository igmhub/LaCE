# CREATING NEW EMULATORS

The training of the emulators is done with the code `train.py`. This code is used to train custom emulators already defined with an emulator label. However, you might need to define a new emulator label with new hyperparameters. This tutorial will guide you through the process of creating a new emulator label.

The file `lace/emulator/constants.py` contains the definitions of the emulator labels, training sets, and the emulator parameters associated with each emulator label.

To create a new emulator label, you first need to add your new emulator label to the `EmulatorLabel` class in the `constants.py` file, for example:

```python
class EmulatorLabel(StrEnum):
    ...
    NEW_EMULATOR = "New_Emulator"
```

"New emulator" is the name of the new emulator label that identifies it in the emulator calls, e.g. `NNEmulator(emulator_label="New_Emulator")`.

Then this label needs to be added to `GADGET_LABELS` or `NYX_LABELS` in the `constants.py` file, depending on the training set you used to train your emulator. For example, if this is a new Gadget emulator, you need to add it to `GADGET_LABELS`:

```python
GADGET_LABELS = {
    ...
    EmulatorLabel.NEW_EMULATOR,
}
```

The dictionary `EMULATOR_PARAMS` also needs to be updated with the new emulator parameters. Here, one needs to add all the arguments needed to initialize the emulator class. For example:

```python
    "Nyx_alphap_cov": {
        "emu_params": [
            "Delta2_p",
            "n_p",
            "alpha_p",
            "mF",
            "sigT_Mpc",
            "gamma",
            "kF_Mpc",
        ],
        "emu_type": "polyfit",
        "kmax_Mpc": 4,
        "ndeg": 6,
        "nepochs": 600,
        "step_size": 500,
        "nhidden": 6,
        "max_neurons": 400,
        "lr0": 2.5e-4,
        "weight_decay": 8e-3,
        "batch_size": 100,
        "amsgrad": True,
        "z_max": 5,
        "include_central": False,
    }
```

Finally, you need to add a description of the new emulator in the `EMULATOR_DESCRIPTIONS` dictionary:

```python
EMULATOR_DESCRIPTIONS = {
    ...
    EmulatorLabel.NEW_EMULATOR: "Description of the new emulator",
}
```
With this, you have added a new emulator label to the code! You should be able to train your new emulator with the command:

```bash
python scripts/train.py --config=path/to/config.yaml
```
or call the emulator directly with:

```python
emulator = NNEmulator(emulator_label="New_Emulator",
                      archive=archive)
```


## Loading the new emulator
Once you have defined a new emulator label, you might want to save the trained emulator models and load them without the need of retraining. This can be done either specifying the `model_path` argument when initializing the emulator. 

```python
emulator = NNEmulator(emulator_label="New_Emulator",
                      model_path="path/to/model.pt",
                      train=False,
                      archive=archive)
```
And also using the `emulator_manager` function:

```python
emulator = emulator_manager(emulator_label="New_Emulator"
                            archive=archive)
```

In the first case, since you are specifying the `model path`, there is no naming convention for the model file. However, in the second case, the saved models must be stored in the following way:
- The folder must be  `data/NNmodels/` from the root of the repository.
- For a specific emulator label, you need to create a new folder, e.g. `New_Emulator`.
- For the emulator using all training simulations, the model file is named `New_Emulator.pt`.
- For the emulator using the training set excluding a given simulation, the model file is named `New_Emulator_drop_sim_{simulation suite}_{simulation index}.pt`. For example, if you exclude the 10th simulation from the mpg training set, the model file is named `New_Emulator_drop_sim_mpg_10.pt`.   

The emulator manager will automatically find the correct model file for the given emulator label. To set this up, you need to add the new emulator label to the `folder` dictionary in the `emulator_manager.py` file.
```python
folder = {
    ...
    EmulatorLabel.NEW_EMULATOR: "NNmodels/New_Emulator/",
}
```