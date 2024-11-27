# EMULATOR TRAINING
The training of the emulators is done with the code `train.py`. which is in the `scripts` folder. This code is used to train the emulators with the data available in the repository. 

In order to train the emulator, one needs to specify the training configuration file. This file is a .yaml file that contains the parameters for the training. An example of this file is `config.yaml` in the `config_files` folder. 

To run the training, one needs to run the following command:

```python
python scripts/train.py --config=path/to/config.yaml
```

## Configuration file instructions

The configuratoin file contains the following parameters:

1. `emulator_type`: Specifies the emulator type, either "NN" or "GP".

2. `emulator_label`: Specifies the predefined model to train. For example, "Cabayol23+". This parameter is optional, if it is not provided, the code will train a new model based on the provided hyperparameters. The options from the emulator_label are defined in the [Emulators_trainingSets.md](./Emulators_trainingSets.md) file.

3. Data source (choose one): The data source can be either an archive or a predefined training set. For archive, the following options are available:
   a. `archive`:
      - `file`: Specifies the simulation type, either "Nyx" or "Gadget".
      - `version`: Specifies the Nyx version or Gadget postprocessing version. Options are:
        - "Pedersen21": Gadget postprocessing in [Pedersen+21 paper](https://arxiv.org/abs/2103.05195).
        - "Cabayol23": Gadget postprocessing in [Cabayol+23 paper](https://arxiv.org/abs/2303.05195).
        - "Nyx23_Oct2023": Nyx version from October 2023.
        - "Nyx23_Jul2024": Nyx version from July 2024.
   b. `training_set`: Specifies a predefined training set. The options are defined in the [Emulators_trainingSets.md](./Emulators_trainingSets.md) file.
   There is the option of dropping simulations from the training set or archive. This is done providing a simulation to the `drop_sim` parameter. The simulations are listed in [Simulations_list.md](./Simulations_list.md).

4. `emulator_params`: List of parameters the emulator will use for predictions. By default, it uses ['Delta2_p', 'n_p','mF', 'sigT_Mpc', 'gamma', 'kF_Mpc']. Nyx_alphap emulators use ['Delta2_p', 'n_p','mF', 'sigT_Mpc', 'gamma', 'kF_Mpc', 'alpha_p'].

5. `drop_sim`: Simulation to exclude from training (optional).

6. `training_set`: Predefined training set to use for training. The options are defined in the Emulators_trainingSets.md](./Emulators_trainingSets.md) file.

7. `save_path`: Location to save the trained model, relative to the project root directory.



8. `hyperparameters`: Neural network and training configuration. These parameters need to be specified when the emulator_label is not provided. If the emulator_label is provided, the hyperparameters will be taken from the predefined emulator. The parameters are:

   • `kmax_Mpc`: Maximum k value in Mpc^-1 to consider.

   • `ndeg`: Degree of the polynomial fit.

   • `nepochs`: Number of training epochs.

   • `step_size`: Number of epochs between learning rate adjustments.

   • `drop_sim`: Simulation to exclude from training (optional).

   • `drop_z`: Redshift to exclude from training (optional).

   • `nhidden`: Number of hidden layers.

   • `max_neurons`: Maximum number of neurons per layer.

   • `seed`: Random seed for reproducibility.

   • `lr0`: Initial learning rate.

   • `batch_size`: Number of samples per batch.

   • `weight_decay`: L2 regularization factor.

   • `z_max`: Maximum redshift to consider.


To train the **GP emulator**, the `emulator_type` parameter is set to "GP" and the `emulator_label` to one of the predefined emulators in the [Emulators_trainingSets.md](./Emulators_trainingSets.md) file. **The GP emulator on the Nyx archive is not supported**. The hyperparameters used by the GP emulator are:

 - `kmax_Mpc`: Maximum k value in Mpc^-1 to consider.
 - `ndeg`: Degree of the polynomial fit.
 - `drop_sim`: Simulation to exclude from training (optional).
 - `z_max`: Maximum redshift to consider.
