# ESTIMATING COMPRESSED PARAMETERS WITH COSMOPOWER

To estimate the compressed parameters with cosmopower, one needs to follow the [installation instructions](../installation.md) of LaCE with cosmopower.

1. [Making predictions with cosmopower](#making-predictions-with-cosmopower)
2. [Estimating compressed parameters](#estimating-compressed-parameters)
    1. [For individual cosmologies](#for-indivudual-cosmologies) 
    2. [For a cosmology chain](#for-a-cosmology-chain)
3. [Training your own cosmopower emulator](#training-your-own-cosmopower-emulator)

## Making predictions with cosmopower
To see examples of how to make predictions with cosmopower, one can follow the tutorial notebook in the notebooks folder.

Cosmopower emulates the linear matter power spectrum. To use the emulator, one needs to provide a set of cosmological parameters that depend on the comsological parameters used to train the emulator.

To load an emulator, you need to do: 

```python
cp_nn = cp.cosmopower_NN(restore=True, 
                         restore_filename=emu_path)
```
There are two trained cosmopower emulators that you can find in the data folder:
- `Pk_cp_NN.pkl`: $\Lambda$CDM emulator.
- `Pk_cp_NN_sumnu.pkl`: $\Lambda$ CDM + $\sum m_\nu$ emulator.

When providing the path to the emulator, you need to give the path to the `Pk_cp_NN.pkl` or `Pk_cp_NN_sumnu.pkl` file __without__ the `.pkl` extension.

To know the parameters that the emulator uses, you can do:

```python
print(cp_nn.parameters())
```
And then to make predictions, you need to provide a dictionary with the parameters that the emulator uses.

```python
# Define the cosmology dictionary
cosmo = {'H0': [cosmo_params["H0"]],
         'h': [cosmo_params["H0"]/100],
         'mnu': [cosmo_params["mnu"]],
         'Omega_m': [(cosmo_params["omch2"] + cosmo_params["ombh2"]) / (cosmo_params["H0"]/100)**2],
         'Omega_Lambda': [1- (cosmo_params["omch2"] + cosmo_params["ombh2"]) / (cosmo_params["H0"]/100)**2],
         'omega_cdm': [cosmo_params["omch2"]],
         'omega_b': [cosmo_params["ombh2"]],
         'As': [cosmo_params["As"]],
         'ns': [cosmo_params["ns"]]}
```
And call the emulator with:

```python
Pk_Mpc = cp_nn.ten_to_predictions_np(cosmo)
```
To access the __k__ values, you can do:

```python
k_Mpc = cp_nn.modes
```
Then to convert to km/s, you can do:

```python
k_kms, Pk_kms = linPCosmologyCosmopower.convert_to_kms(cosmo, 
                                                       k_Mpc, 
                                                       Pk_Mpc, 
                                                       z_star = z_star)
```

## Estimating compressed parameters 

### For indivudual cosmologies
Once you have the predictions, you can estimate the compressed parameters with:
```python
# call the class.
linP_Cosmology_Cosmopower = linPCosmologyCosmopower()
```
Fit the polynomial to the power spectrum:
```python
linP_kms = linP_Cosmology_Cosmopower.fit_polynomial(
    xmin = kmin_kms / kp_kms, 
    xmax= kmax_kms / kp_kms, 
    x = k_kms / kp_kms, 
    y = Pk_kms, 
    deg=2
)
```

And then estimate the star parameters:

```python
starparams_CP = linP_Cosmology_Cosmopower.get_star_params(linP_kms = linP_kms, 
                                                          kp_kms = kp_kms)
```

where `linP_kms` are the linear matter power spectrum predictions in km/s and `kp_kms` is the pivot point in s/km. 

### For a cosmology chain
To estimate the parameters for a cosmology chain, you first need to call the class:
```python
fitter_compressed_params = linPCosmologyCosmopower()
```

Then check the expected parameters of the cosmopower model:
```python
print(fitter_compressed_params.cp_emulator.cp_emulator.parameters())
```

And create a dictionary with the naming convertion between the parameters in your chain and the parameters used by the cosmopower model. We need additional parameters to convert to km/s.

```python
param_mapping = {
    'h': 'h',
    'm_ncdm': 'm_ncdm',
    'omega_cdm': 'omega_cdm',
    'Omega_m': 'Omega_m',
    'Omega_Lambda': 'Omega_Lambda',
    'ln_A_s_1e10': 'ln_A_s_1e10',
    'n_s': 'n_s'
}
```
And then call the function to estimate the parameters:

```python
linP_cosmology_results = fitter_compressed_params.fit_linP_cosmology(chains_df = df, 
                                                                     param_mapping = param_mapping)
```

By default in loads the $\Lambda$CDM + \sum m_\nu emulator. If you want to use another emualtor, specify the name in the class argument 'cosmopower_model' and save the model in the data/cosmopower_models folder. The model needs to be a `.pkl` file and the it must be called without the `.pkl` extension.

## Training your own cosmopower emulator
To train your own cosmopower emulator, you can follow the tutorial notebook.
First, one needs to create a LH sampler with the parameters of interest. For example, to create a LH sampler with the $\Lambda$CDM + $\sum m_\nu$ emulator, you can do:
```python
dict_params_ranges = {
    'ombh2': [0.015, 0.03],
    'omch2': [0.05, 0.3],
    'H0': [60, 80],
    'ns': [0.8, 1.2],
    'As': [5e-10, 4e-9],
    'mnu': [0, 2],}
```

And create the LH sample as:

```python
create_LH_sample(dict_params_ranges = dict_params_ranges,
                     nsamples = 10_000,
                     filename = "LHS_params_sumnu.npz")
```

After that, you need to create the power spectrum to train the emulator. 

```python
generate_training_spectra(input_LH_filename = 'LHS_params_sumnu.npz',
                          output_filename = "linear_sumnu.dat")
```

Make sure that the input file is the one you created in the previous step. This calls CAMB to generate the power spectrum. It takes time to run. So far, we have not needed a long training sample to reach good accuracy.

Once the power spectrum is generated, you can train the emulator with:

```python
cosmopower_prepare_training(params = ["H0", "mnu", "omega_cdm", "omega_b", "As", "ns"],
    Pk_filename = "linear_sumnu.dat")
```

followed by:
```python
cosmopower_train_model(model_save_filename = "Pk_cp_NN_test")
```
