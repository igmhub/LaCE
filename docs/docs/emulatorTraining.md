# NN - Emulator Training
The training of the emulators is done with the code `train.py`. which is in the `scripts` folder. This code is used to train the emulators with the data available in the repository. 

In order to train the emulator, one needs to specify the training configuration file. This file is a .yaml file that contains the parameters for the training. An example of this file is `config.yaml` in the `scripts` folder. 

To run the training, one needs to run the following command:

```
python scripts/train.py --config=path/to/config.yaml
```

The configuratoin file contains the following parameters:

1. `emulator_type`: Specifies the emulator type, either "NN" or "GP".

2. `emulator_label`: Specifies the predefined model to train. For example, "Cabayol23+". This parameter is optional, if it is not provided, the code will train a new model based on the provided hyperparameters. The options from the emulator_label are:
    - Cabayol23: Neural network emulating the optimal P1D of Gadget simulations fitting coefficients to a 5th degree polynomial. It goes to scales of 4Mpc^{-1} and z<=4.5.
    - Cabayol23+: Neural network emulating the optimal P1D of Gadget simulations fitting coefficients to a 5th degree polynomial. It goes to scales of 4Mpc^{-1} and z<=4.5. Updated version compared to Cabayol+23 paper.
    - Cabayol23_extended: Neural network emulating the optimal P1D of Gadget simulations fitting coefficients to a 7th degree polynomial. It goes to scales of 8Mpc^{-1} and z<=4.5.
    - Cabayol23+_extended: Neural network emulating the optimal P1D of Gadget simulations fitting coefficients to a 5th degree polynomial. It goes to scales of 4Mpc^{-1} and z<=4.5. Updated version compared to Cabayol+23 paper.
    - Nyx_v0: Neural network emulating the optimal P1D of Nyx simulations fitting coefficients to a 6th degree polynomial. It goes to scales of 4Mpc^{-1} and z<=4.5.
    - Nyx_v0_extended: Neural network emulating the optimal P1D of Nyx simulations fitting coefficients to a 6th degree polynomial. It goes to scales of 8Mpc^{-1} and z<=4.5.
    - Nyx_alphap: Neural network emulating the optimal P1D of Nyx simulations fitting coefficients to a 6th degree polynomial. It goes to scales of 4Mpc^{-1} and z<=4.5.
    - Nyx_alphap_extended: Neural network emulating the optimal P1D of Nyx simulations fitting coefficients to a 6th degree polynomial. It goes to scales of 8Mpc^{-1} and z<=4.5.
    - Nyx_alphap_cov: Neural network under testing for the Nyx_alphap emulator.

3. Data source (choose one):
   a. `archive`:
      - `file`: Specifies the simulation type, either "Nyx" or "Gadget".
      - `version`: Specifies the Nyx version or Gadget postprocessing version. Options are:
        - "Pedersen21": Gadget postprocessing in Pedersen+21 paper.
        - "Cabayol23": Gadget postprocessing in Cabayol+23 paper.
        - "Nyx23_Oct2023": Nyx version from October 2023.
        - "Nyx23_Jul2024": Nyx version from July 2024.
   b. `training_set`: Specifies a predefined training set. Set to null if using an archive. The options are:
        - "Pedersen21": Gadget postprocessing in Pedersen+21 paper.
        - "Cabayol23": Gadget postprocessing in Cabayol+23 paper.
        - "Oct2023": Nyx version from October 2023.
        - "Jul2024": Nyx version from July 2024.

4. `emulator_params`: List of parameters the emulator will use for predictions. By default, it uses ['Delta2_p', 'n_p','mF', 'sigT_Mpc', 'gamma', 'kF_Mpc']. Nyx_alphap emulators use ['Delta2_p', 'n_p','mF', 'sigT_Mpc', 'gamma', 'kF_Mpc', 'alpha_p'].

5. `hyperparameters`: Neural network and training configuration:
   `kmax_Mpc`: Maximum k value in Mpc^-1 to consider.
   `ndeg`: Degree of the polynomial fit.
   `nepochs`: Number of training epochs.
   `step_size`: Number of epochs between learning rate adjustments.
   `drop_sim`: Simulation to exclude from training (optional).
   `drop_z`: Redshift to exclude from training (optional).
   `nhidden`: Number of hidden layers.
   `max_neurons`: Maximum number of neurons per layer.
   `seed`: Random seed for reproducibility.
   `lr0`: Initial learning rate.
   `batch_size`: Number of samples per batch.
   `weight_decay`: L2 regularization factor.
   `z_max`: Maximum redshift to consider.

5. `save_path`: Location to save the trained model, relative to the project root directory.
