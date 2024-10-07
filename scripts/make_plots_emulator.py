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


def plot_emulated_p1d(archive_name='Nyx'):
    """
    Function to plot emulated P1D using specified archive (Nyx or Gadget).
    
    Parameters:
    archive (str): Archive to use for data ('Nyx' or 'Gadget')
    """
    
    # Print the type of archive selected for debugging
    print(f"Selected archive: {archive_name}")

    # Get the base directory of the lace module
    repo = os.path.dirname(lace.__path__[0]) + "/"
    print(f"Repository base path: {repo}")  # Print path for clarity

    # Choose archive and emulator parameters based on the archive type
    if archive_name == 'Nyx':
        # Define the parameters for the emulator specific to Nyx
        nyx_emu_params = ['Delta2_p', 'n_p', 'alpha_p', 'mF', 'sigT_Mpc', 'gamma', 'kF_Mpc']
        emulator_label='Nyx_alphap'
        training_set = 'Nyx23_Oct2023'
        model_path = f'{repo}data/NNmodels/Nyxap_Oct2023/Nyx_alphap_drop_sim'
        
        # Initialize a NyxArchive instance for postprocessing data
        archive = nyx_archive.NyxArchive(verbose=True)
        print("Initialized NyxArchive.")
    
    elif archive_name == 'Gadget':
        # Define the parameters for the emulator specific to Gadget
        nyx_emu_params = ['Delta2_p', 'n_p', 'mF', 'sigT_Mpc', 'gamma', 'kF_Mpc']
        training_set='Cabayol23'
        emulator_label='Cabayol23+'
        model_path = f'{repo}data/NNmodels/Cabayol23+/Cabayol23+_drop_sim'
        
        # Initialize a GadgetArchive instance for postprocessing data
        archive = gadget_archive.GadgetArchive(postproc="Cabayol23")
        print("Initialized GadgetArchive.")
    
    else:
        # Raise an error if an invalid archive is provided
        raise ValueError("archive must be 'Nyx' or 'Gadget'")

    # Iterate over each simulation in the archive
    skip_sims = ['nyx_14', 'nyx_15', 'nyx_16', 'nyx_17', 'nyx_seed', 'nyx_wdm']
    for ii, sim in enumerate(archive.list_sim):
        if sim in skip_sims:
            continue

        print(f"Processing simulation: {sim}")  # Print simulation being processed

        # Initialize the emulator with the given parameters and model path
        if (sim=='nyx_central')|(sim=='mpg_central'):
            if archive_name == 'Gadget':
                model_path = f'{repo}data/NNmodels/Cabayol23+/Cabayol23+.pt'
            elif archive_name == 'Nyx':
                model_path = f'{repo}data/NNmodels/Nyxap_Oct2023/Nyx_alphap.pt'
                
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
        
        print(f"Initialized emulator for simulation {sim}")

        # Get testing data for the current simulation
        testing_data = archive.get_testing_data(sim_label=f'{sim}')
        if sim!='nyx_central':
            testing_data = [d for d in testing_data if d['val_scaling']==1]
            
        print(f"Retrieved testing data for simulation {sim}")

        # Plot and save the emulated P1D
        plot_p1d_vs_emulator(
            testing_data,
            emulator,
            save_path=f'{repo}data/validation_figures/{archive_name}/{sim}.png'
        )
        print(f"Saved plot for simulation {sim}")

    print("Plotting complete!")
    return


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Plot emulated P1D")
    
    # Add an argument for specifying the archive type ('Nyx' or 'Gadget')
    parser.add_argument(
        "--archive", 
        type=str, 
        default="Nyx", 
        choices=["Nyx", "Gadget"], 
        help="Specify the archive to use, either 'Nyx' or 'Gadget'"
    )
    
    # Parse the arguments provided via command line
    args = parser.parse_args()

    # Print the parsed arguments for clarity
    print(f"Command-line argument for archive: {args.archive}")

    # Call the function with the parsed argument
    plot_emulated_p1d(archive_name=args.archive)
