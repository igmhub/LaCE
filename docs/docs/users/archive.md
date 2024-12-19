# ARCHIVES

The LaCE emulators support two types of archives:

- Gadget archive: Contains the P1D of Gadget simulations described in [Pedersen+21](https://arxiv.org/abs/2011.15127).  
- Nyx archive: Contains the P1D of Nyx simulations described in (In prep.)

## Loading a Gadget Archive
The Gadget archive contains 30 training simulations and 7 test simulations. Each simulation contains 11 snapshotts covering redshifts from 2 to 4.5 in steps of 0.25.

To laod a Gadget archive, you can use the `GadgetArchive` class:
```python
from lace.archive.gadget_archive import GadgetArchive
```
The Gadget archive can be loaded with different post-processings: the one described in [Pedersen+21](https://arxiv.org/abs/2011.15127), and the one from [Cabayol+23](https://arxiv.org/abs/2305.19064).

The P1D from the Gadget archive with the Pedersen+21 post-processing can be accessed as follows:
```python
archive = GadgetArchive(postproc='Pedersen21')
```
This post-processing measures the P1D along one of the three box axes and does not contain mean-flux rescalings.

On the other hand, the P1D from the Gadget archive with the Cabayol+23 post-processing can be accessed as follows:
```python
archive = GadgetArchive(postproc='Cabayol23')
```
This post-processing measures the P1D along the three box principal axes and contains five mean-flux rescaling per snapshot.

## Loading a Nyx Archive
To load the Nyx archive, you can use the `NyxArchive` class:

```python
from lace.archive.nyx_archive import NyxArchive
```
Since the Nyx archive is not publicly available yet, **you need to set the `NYX_PATH` environment variable to the path to the Nyx files** on your local computer (or the cluster where you are running the code).

There are two versions of the Nyx archive available: Oct2023 and Jul2024. The first one contains 17 training simulations and 4 test simulations, and the second one contains 17 training simulations and 3 test simulations (the simulations are better described [here](./Simulations_list.md)). Each simulation contains 14 snapshots covering redshifts from 2.2 to 4.8 in steps of 0.2 plus additional snapshotts at higher redshifts for some of the simulations. In both cases, it is not recommended to use simulation number 14. 

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
where you can replace `sim_label` by any of the test simulation labels available in the archive (see [here](./Simulations_list.md)). This will only load the fiducial snapshots without mean flux rescalings. ¡