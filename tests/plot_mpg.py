import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for rendering plots
from matplotlib import pyplot as plt
import lace
from lace.emulator.nn_emulator import NNEmulator
from lace.archive import gadget_archive
from lace.utils.plotting_functions import plot_p1d_vs_emulator

def test(output_dir):
    """
    Function to plot emulated P1D using specified archive and save plots to output directory.
    
    Parameters:
    output_dir (str): Directory to save the generated plots.
    """
    archive_name = 'Gadget'
    # Get the base directory of the lace module
    repo = os.path.dirname(lace.__path__[0]) + "/"
    
    # Define the parameters for the emulator specific to Gadget
    emu_params = ['Delta2_p', 'n_p', 'mF', 'sigT_Mpc', 'gamma', 'kF_Mpc']
    training_set = 'Cabayol23'
    emulator_label = 'Cabayol23+'
    model_path = f'{repo}data/NNmodels/Cabayol23+/Cabayol23+_drop_sim'

    # Initialize a GadgetArchive instance for postprocessing data
    archive = gadget_archive.GadgetArchive(postproc="Cabayol23")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for ii, sim in enumerate(['mpg_1', 'mpg_central']):
        if sim == 'mpg_central':
            model_path_central = f'{repo}data/NNmodels/Cabayol23+/Cabayol23+.pt'
            emulator = NNEmulator(
                training_set=training_set,
                emulator_label=emulator_label,
                emu_params=emu_params,
                model_path=model_path_central,
                train=False,
            )
        else:
            emulator = NNEmulator(
                training_set=training_set,
                emulator_label=emulator_label,
                emu_params=emu_params,
                model_path=model_path + f'_{sim}.pt',
                drop_sim=sim,
                train=False,
            )
        
        # Get testing data for the current simulation
        testing_data = archive.get_testing_data(sim_label=f'{sim}')
        if sim != 'nyx_central':
            testing_data = [d for d in testing_data if d['val_scaling'] == 1]
        
        # Plot and save the emulated P1D
        save_path = os.path.join(output_dir, f'{sim}.png')
        plot_p1d_vs_emulator(testing_data, emulator, save_path=save_path)

    return

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate and save emulator plots.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the plots.")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the test function with the specified output directory
    test(args.output_dir)
