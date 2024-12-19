# MAKING PREDICTIONS WITH LACE

## Loading a predefined emulator 
The easiest way to load an emulator is to use the `set_emulator` class. This can be done with an [archive](archive.md) or a training set. This will load a trained emulator with the specifications of the [emulator label](Emulators_trainingSets.md).

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

The supported emulators can be found in the [Emulators_trainingSets.md](./Emulators_trainingSets.md) file.

## Loading a custom emulator

Another option is to load an emulator model that does not correspond to a predifined emulator label. This can be done by, for example:

```python
emulator = NNEmulator(training_set='Cabayol23', 
            emulator_label='Cabayol23+',
            model_path='path/to/model.pt',
            drop_sim=None,
            train=False)
```
where `model_path` is the path to the `.pt` file containing the trained model and `train=False` indicates that the model is not being trained. In the model you are loading has been trained by dropping simulations, you should specify which simulations to drop using the `drop_sim` argument.

## Making predictions 

To emulate the P1D of a simulation, you can use the `emulate_p1d_Mpc` method. This method requires a dictionary containing the simulation parameters.

```python
p1d = emulator.emulate_p1d_Mpc(sim_params, k_Mpc)
```

## Predicitng from a config file

One can also make predictions and plot them by using the `predict.py` script in the `scripts` folder. This script allows to make predictions on a test set, plot the P1D errors and save the predictions. An example of how to use this script is:

```bash python
python bin/predict.py --config config_files/config_predict.yaml
```
Similarly to the [training script](emulatorTraining.md), the config file accepts the following fields:

There are two ways of loading an emulator:
1. By providing an emulator label (see list of supported emulators [here](./Emulators_trainingSets.md))
- `emulator_type`: Choose between "NN" (neural network) or "GP" (Gaussian process) emulator
- `emulator_label`: Label of the predefined model to use (see list of supported emulators [here](./Emulators_trainingSets.md)) 
- `drop_sim`: Simulation to exclude from training (optional)
- `archive`: Configuration for loading simulation archive
  - `file`: "Nyx" or "Gadget"
  - `version`: Version of the simulation archive
  - `sim_test`: Label of the test simulation to use for predictions. See list of available simulations [here](./Simulations_list.md).
- `average_over_z`: Whether to average P1D errors over redshift (true/false)
- `save_plot_path`: Path where to save the validation plot. If None, the plot is not saved.
- `save_predictions_path`: Path where to save the predictions. If None, the predictions are not saved.


2. By providing a path to the directory containing the trained model (this directory should contain the `.pt` file.
- `emulator_type`: Choose between "NN" (neural network) or "GP" (Gaussian process) emulator
- `drop_sim`: Simulation to exclude from training (optional)
- `archive`: Configuration for loading simulation archive
  - `file`: "Nyx" or "Gadget"
  - `version`: Version of the simulation archive
  - `sim_test`: Label of the test simulation to use for predictions. See list of available simulations [here](./Simulations_list.md).
- `emulator_params`: List of parameters used by the emulator. The default is `["Delta2_p", "n_p", "alpha_p", "sigT_Mpc", "gamma", "kF_Mpc"]`.
- `average_over_z`: Whether to average P1D errors over redshift (true/false)
- `save_plot_path`: Path where to save the validation plot. If None, the plot is not saved.
- `save_predictions_path`: Path where to save the predictions. If None, the predictions are not saved.
- `hyperparameters`: Dictionary containing the hyperparameters of the emulator. This will be used **ONLY** if `emulator_label` is not provided.
- `model_path`: Path to the directory containing the trained model (this directory should contain the `.pt` file).
