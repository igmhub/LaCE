# Estimating compressed parameters with cosmopower
To estimate the compressed parameters with cosmopower, one needs to follow the [installation instructions](installation.md) of LaCE with cosmopower.

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
cosmo = {'H0': [cosmo_params["H0"]],
         'mnu': [cosmo_params["mnu"]],
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

## Estimating compressed parameters 

### For indivudual cosmologies
Once you have the predictions, you can estimate the compressed parameters with:
```python
# call the class.
linP_Mpc = cp_nlinP_Cosmology_Cosmopower = linPCosmologyCosmopower()n.ten_to_predictions_np(cosmo)
```

```python
starparams_CP = linP_Cosmology_Cosmopower.get_star_params(linP = linP_Mpc, 
                                       kp = kp_Mpc)
```
where `linP_Mpc` are the linear matter power spectrum predictions and `kp_Mpc` is the pivot point in Mpc. 

To run this function, you need to first smooth the linear matter power spectrum with a polynomial defining the minimum and maximum __k__ values that you want to fit.

```python
linP_Mpc = linP_Cosmology_Cosmopower.fit_polynomial(
    xmin = kmin_Mpc / kp_Mpc, 
    xmax= kmax_Mpc / kp_Mpc, 
    x = k_Mpc / kp_Mpc, 
    y = Pk_Mpc, 
    deg=2)
```

### For a cosmology chain
To estimate the parameters for a cosmology chain, you first need to call the class:
```python
fitter_compressed_params = linPCosmologyCosmopower()
```

Then check the expected parameters of the cosmopower model:
```python
print(fitter_compressed_params.cp_emulator.cp_emulator.parameters())
```

And create a dictionary with the naming convertion between the parameters in your chain and the parameters used by the cosmopower model.

```python
param_mapping = {
    'h': 'h',
    'm_ncdm': 'm_ncdm',
    'omega_cdm': 'omega_cdm',
    'Omega_m': 'Omega_m',
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

