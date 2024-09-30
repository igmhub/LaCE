# Import necessary modules
## General python modules
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import os
import argparse  # Used for parsing command-line arguments

## LaCE specific modules
import lace
from lace.emulator.nn_emulator import NNEmulator
from lace.archive import nyx_archive, gadget_archive
from lace.utils import poly_p1d
from lace.utils.plotting_functions import plot_p1d_vs_emulator


def test():
    """
    Function to plot emulated P1D using specified archive (Nyx or Gadget).
    
    Parameters:
    archive (str): Archive to use for data ('Nyx' or 'Gadget')
    """
    archive_name = 'Gadget'
    # Get the base directory of the lace module
    repo = os.path.dirname(lace.__path__[0]) + "/"
    
    # Define the parameters for the emulator specific to Gadget
    emu_params = ['Delta2_p', 'n_p', 'mF', 'sigT_Mpc', 'gamma', 'kF_Mpc']
    training_set='Cabayol23'
    emulator_label='Cabayol23+'
    model_path = f'{repo}data/NNmodels/Cabayol23+/Cabayol23+_drop_sim'

    # Initialize a GadgetArchive instance for postprocessing data
    archive = gadget_archive.GadgetArchive(postproc="Cabayol23")
    
    for ii, sim in enumerate(archive.list_sim):
        if sim=='mpg_central':
            model_path = f'{repo}data/NNmodels/Cabayol23+/Cabayol23+.pt'

            emulator = NNEmulator(
                training_set=training_set,
                emulator_label=emulator_label,
                emu_params=nyx_emu_params,
                model_path=model_path,
                drop_sim=None,
                train=False,
            )
        else:
            emulator = NNEmulator(
                training_set=training_set,
                emulator_label=emulator_label,
                emu_params=nyx_emu_params,
                model_path=model_path+f'_{sim}.pt',
                drop_sim=sim,
                train=False,
            )
        
        # Get testing data for the current simulation
        testing_data = archive.get_testing_data(sim_label=f'{sim}')
        if sim!='nyx_central':
            testing_data = [d for d in testing_data if d['val_scaling']==1]
            
        # Plot and save the emulated P1D
        plot_p1d_vs_emulator(
            testing_data,
            emulator,
            save_path=f'{repo}data/validation_figures/{archive_name}/{sim}.png'
        )
    return

# Call the function to execute the test
test()
