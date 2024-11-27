# TRAINING OPTIONS

There are several features that can be used to customize the training of the emulators. This tutorial will guide you through the process of training emulators with different options.

- [Weighting with a covariance matrix](#weighting-with-a-covariance-matrix)
- [Weighting simulations depending of the scalings (mean flux, temperature )](#weighting-simulations-depending-of-the-scalings-mean-flux-temperature)

## Weighting with a covariance matrix
The emulator supports weighting the training simulations with a covariance matrix. This covariance matrix is used to weight the training simulations during the training of the neural network.

To train an emulator with a covariance matrix, you need to provide a covariance matrix for the training simulations. Currently, the emulator only supports a diagonal covariance matrix. It is important that the covariance matrix is given in the __k__ binning of the training simulations.

The function '_load_DESIY1_err' in the `nn_emulator.py` file loads a covariance matrix. The covariance must be a json file with the relative error as a function of __z__ for each __k__ bin.

From the relative error file in 'data/DESI_cov/rel_err_DESI_Y1.npy', we can generate the json file with the following steps:

First we load the data from the relative error file:

```python
cov =  np.load(PROJ_ROOT / "data/DESI_cov/rel_error_DESIY1.npy", allow_pickle=True)
# Load the data dictionary
data = cov.item()
```

Then we extract the arrays. This has a hidden important step. In the original relative error file, the values for the relative error are set to 100 correspond to the scales not measured by DESI. The value of 100 is set at random, and can be optimized for the training. Initial investigations indicated that setting the value to 5 was working well. However, this parameter could be furtehr refined. Currently is set to 5, but other values of this dummy value could be used.

```python
# Extract the arrays
z_values = data['z']
rel_error_Mpc = data['rel_error_Mpc']
rel_error_Mpc[rel_error_Mpc == 100] = 5

k_cov = data['k_Mpc']
```

Then we extract the __k__ values for the training simulations to ensure that the covariance matrix is given in the __k__ binning of the training simulations.

```python
testing_data_central = archive.get_testing_data('nyx_central')
testing_data = archive.get_testing_data('nyx_0')
k_Mpc_LH = testing_data[0]['k_Mpc'][testing_data[0]['k_Mpc']<4]
```
And then we create the dictionary with the relative error as a function of __z__ for each __k__ bin:

```python
# Load the data dictionary
data = cov.item()
z_values = data['z']

dict_={}
for z, rel_error_row in zip(z_values, rel_error_Mpc):
    f = interp1d(k_cov, rel_error_row, fill_value="extrapolate")
    rel_error_Mpc_interp = f(k_Mpc_LH)
    rel_error_Mpc_interp[0:3] = rel_error_Mpc_interp[3]
    dict_[f"{z}"]=rel_error_Mpc_interp.tolist()
        
# Create a new dictionary with z as keys and corresponding rel_error_Mpc rows as values
#z_to_rel_error_serializable = {float(z): rel_error_row.tolist() for z, rel_error_row in z_to_rel_error.items()}
```

And finally we save the dictionary to a json file:

```python
# Save the z_to_rel_error dictionary to a JSON file
with open(PROJ_ROOT / "data/DESI_cov/rerr_DESI_Y1.json", "w") as json_file:
    json.dump(dict_, json_file, indent=4)
```

## Weighting simulations depending of the scalings (mean flux, temperature )

The `nn_emulator.py` file contains a function `_get_rescalings_weights` that allows to weight the simulations depending on the scalings. This can be used to give more importance to the snapshots with certain scalings. It is possible to weight differently based on the scaling value and the redshift. Initial investigations did not show an improvement in the emulator performance when weighting the simulations. However, might be worth to further investigate this option.

The function `_get_rescalings_weights` can be customized by changing the line:

```python
weights_rescalings[np.where([(d['val_scaling'] not in [0,1] and d['z'] in [2.8, 3,3.2,3.4]) for d in self.training_data])] = 1
```
The weight value of 1 does not have any effect on the training. To downweight certain snapshots, a value lower than 1 can be used. In this particular case, modifying it to a lower value, for example 0.5, would downweight the snapshots with a scaling value not equal to 0 or 1 (temparature scalings) and a redshift in the range [2.8, 3,3.2,3.4].

Initial investigations showed that very low values of the weights, for example 0.01 already led to a similar performance to the one of an emulator trained with equal weights.