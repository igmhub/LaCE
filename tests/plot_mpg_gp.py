# Import necessary modules
## General python modules
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for rendering plots
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import os
import argparse  # Used for parsing command-line arguments

## LaCE specific modules
import lace
from lace.emulator.gp_emulator import GPEmulator
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
    training_set='Pedersen21'
    emulator_label='Pedersen23'
    

    # Initialize a GadgetArchive instance for postprocessing data
    archive = gadget_archive.GadgetArchive(postproc="Cabayol23")
    
    # Directory for saving plots
    save_dir = f'{repo}data/tmp_validation_figures/{archive_name}/'
    #save_dir = '{repo}tmp/validation_figures/'
    # Create the directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)
    
    for ii, sim in enumerate(archive.list_sim):
        if sim in archive.list_sim_test:
            emulator = GPEmulator(
                training_set=training_set,
                emulator_label=emulator_label,
                emu_params=emu_params,
            )
        else:
            emulator = GPEmulator(
                training_set=training_set,
                emulator_label=emulator_label,
                emu_params=emu_params,
                drop_sim=sim,
            )
        
        # Get testing data for the current simulation
        testing_data = archive.get_testing_data(sim_label=f'{sim}')
        if sim != 'nyx_central':
            testing_data = [d for d in testing_data if d['val_scaling'] == 1]
            
        # Plot and save the emulated P1D
        save_path = f'{save_dir}{sim}{emulator_label}.png'
        plot_p1d_vs_emulator(testing_data, emulator, save_path=save_path)
    
    return  

# Call the function to execute the test
test()
