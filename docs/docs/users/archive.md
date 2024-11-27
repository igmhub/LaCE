# ARCHIVE

The LaCE emulators support two types of archives:
- Gadget archive: Contains the P1D of Gadget simulations described in [Pedersen+21](https://arxiv.org/abs/2011.15127).  
- Nyx archive: Contains the P1D of Nyx simulations described in (In prep.)

## Loading a Gadget Archive
The Gadget archive contains 30 training simulations and seven test simulations. Each simulation contains 11 snapshotts covering redshifts from 2 to 4.5 in steps of 0.25

To laod a Gadget archive, you can use the `GadgetArchive` class:
```python
from lace.archive.gadget_archive import GadgetArchive
```
The Gadget archive can be loaded with different post-processings: the one described in [Pedersen+21](https://arxiv.org/abs/2011.15127), and the one from [Cabayol+23](https://arxiv.org/abs/2305.19064).

The P1D from the Gadget archive with the Pedersen+21 post-processing can be accessed as follows:
```python
archive = GadgetArchive(postproc='Pedersen21')
```
This post-processing measures the P1D along one of the three box axes and contains three mean-flux rescaling per snapshot.

## Loading a Nyx Archive
To load the Nyx archive, you can use the `NyxArchive` class:
```python
from lace.archive.nyx_archive import NyxArchive
```
Since the Nyx archive is not publicly available yet, you need to set the `NYX_PATH` environment variable to the path to the Nyx files on your local computer.

There are two versions of the Nyx archive available: `Oct2023` and `Jul2024`. The first one contains 17 training simulations and 4 test simulations, and the second one contains 17 training simulations and 3 test simulations. Each simulation contains 14 snapshotts covering redshifts from 2.2 to 4.8 in steps of 0.2 plus additional snapshotts at higher redshifts for some of the simulations. In both cases, it is not recommended to use simulation number 14. 

The P1D from the Nyx archive with the Oct2023 version can be accessed as follows:
```python
archive = NyxArchive(nyx_version='Oct2023')
```
And the P1D from the Nyx archive with the Jul2024 version can be accessed as follows:
```python
archive = NyxArchive(nyx_version='Jul2024')
```

## Accessing the Training Set and the Test Set
To access all data in the archive, you can use the `archive.data`. This will load all the snapshots and mean fluxes for all the simulations in the archive. 

If you want to access only the training set, you can use 
```python
archive.get_training_data(emu_params=emu_params)
```
and you will automatically load all snapshots and mean flux rescalings available in the training simulations.  

For the test set, the equivalent function is:
```python
archive.get_testing_data(sim_label='mpg_central')
```
where you can replace `sim_label` by any of the test simulation labels available in the archive. This will only load the fiducial snapshots without mean flux rescalings. 

## Key keywords in the archive
The archive contains many keywords that can be used to access specific data. Here is a non-exhaustive list of the most important ones:

- `sim_label`: The label of the simulation. It can be any of the test simulation labels available in the archive.
- `z`: The snapshot redshift.
- `ind_axis`: Indicates the axis along which the P1D is measured. It can be 0, 1, 2 or 'average'
- `ind_rescaling`: The index of mean-flux rescaling of the P1D.
- `val_scaling`: The value of mean-flux rescaling of the P1D.
- `cosmo_params`: A dictionary containing the cosmological parameters of the simulation.
- `p1d_Mpc`: The P1D in Mpc.
- `k_Mpc`: The wavevector in Mpc.
 