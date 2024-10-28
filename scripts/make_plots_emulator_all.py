# Import necessary modules
## General python modules
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import os
import argparse  
from loguru import logger

## LaCE specific modules
import lace
from lace.emulator.nn_emulator import NNEmulator
from lace.archive import nyx_archive, gadget_archive
from lace.utils import poly_p1d
from lace.utils.plotting_functions import plot_p1d_vs_emulator


def make_p1d_err_plot(p1ds_err, kMpc_test):
    """
    Plot the P1D errors with 16th and 84th percentiles shaded.
    
    Parameters:
    p1ds_err (np.array): Array of P1D errors
    kMpc_test (np.array): k values in Mpc^-1
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate median, 16th and 84th percentiles
    p1d_median = np.nanmedian(p1ds_err.reshape(-1, len(kMpc_test)), axis=0)
    perc_16 = np.nanpercentile(p1ds_err.reshape(-1, len(kMpc_test)), 16, axis=0)
    perc_84 = np.nanpercentile(p1ds_err.reshape(-1, len(kMpc_test)), 84, axis=0)
    
    # Plot median line
    ax.plot(kMpc_test, p1d_median,  color='crimson')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    ax.fill_between(kMpc_test, perc_16, perc_84, alpha=0.3, color='crimson')
    
    ax.set_xlabel('k (Mpc$^{-1}$)')
    ax.set_ylabel('Relative Error in P1D')
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(f'p1d_errors_all.pdf', bbox_inches='tight')
    plt.close()

def plot_emulated_p1d(archive_name='Nyx'):
    """
    Function to plot emulated P1D using specified archive (Nyx or Gadget).
    
    Parameters:
    archive (str): Archive to use for data ('Nyx' or 'Gadget')
    """
    # Print the type of archive selected for debugging
    logger.info(f"Selected archive: {archive_name}")

    # Get the base directory of the lace module
    repo = os.path.dirname(lace.__path__[0]) + "/"
    logger.info(f"Repository base path: {repo}")  # Print path for clarity

    # Choose archive and emulator parameters based on the archive type
    if archive_name == 'Nyx':
        # Define the parameters for the emulator specific to Nyx
        nyx_emu_params = ['Delta2_p', 'n_p', 'alpha_p', 'mF', 'sigT_Mpc', 'gamma', 'kF_Mpc']
        emulator_label='Nyx_alphap'
        training_set = 'Nyx23_Oct2023'
        model_path = f'{repo}data/NNmodels/Nyxap_Oct2023/Nyx_alphap_drop_sim'
        
        # Initialize a NyxArchive instance for postprocessing data
        archive = nyx_archive.NyxArchive(verbose=False,
                                         force_recompute_linP_params=False,
                                         kp_Mpc=0.7)
        logger.info("Initialized NyxArchive.")
    
    elif archive_name == 'Gadget':
        # Define the parameters for the emulator specific to Gadget
        nyx_emu_params = ['Delta2_p', 'n_p', 'mF', 'sigT_Mpc', 'gamma', 'kF_Mpc']
        training_set='Cabayol23'
        emulator_label='Cabayol23+'
        model_path = f'{repo}data/NNmodels/Cabayol23+/Cabayol23+_drop_sim'
        
        # Initialize a GadgetArchive instance for postprocessing data
        archive = gadget_archive.GadgetArchive(postproc="Cabayol23")
        logger.info("Initialized GadgetArchive.")
    
    else:
        # Raise an error if an invalid archive is provided
        raise ValueError("archive must be 'Nyx' or 'Gadget'")

    # Iterate over each simulation in the archive
    skip_sims = ['nyx_14', 'nyx_15', 'nyx_16', 'nyx_17','nyx_18', 'nyx_seed', 'nyx_wdm', "nyx_3_ic", "nyx_central"]
    sims_to_process = [sim for sim in archive.list_sim if sim not in skip_sims]
    testing_data_central = archive.get_testing_data(sim_label=f"nyx_central")

    z = [d['z'] for d in testing_data_central if d['z'] < 4.8]
    Nz = len(z)
    Nk = len(testing_data_central[0]['p1d_Mpc'][(testing_data_central[0]['k_Mpc'] > 0) & (testing_data_central[0]['k_Mpc'] < 4)])+1

    Nsim = len(sims_to_process)
    p1ds_err = np.zeros((Nsim, Nz, Nk))

    for ii, sim in enumerate(sims_to_process):
        if sim in skip_sims:
            continue

        logger.info(f"Processing simulation: {sim}")  # Print simulation being processed

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
        
        logger.info(f"Initialized emulator for simulation {sim}")

        # Get testing data for the current simulation
        testing_data = archive.get_testing_data(sim_label=f'{sim}')
        
        if sim!='nyx_central':
            testing_data = [d for d in testing_data if d['val_scaling']==1]
            
        logger.info(f"Retrieved testing data for simulation {sim}")

        z = [d['z'] for d in testing_data if d['z'] < 4.8]
        for m in range(Nz):
            kMpc_test = testing_data[m]['k_Mpc']
            p1d_true = testing_data[m]['p1d_Mpc']
            p1d_true = p1d_true[(kMpc_test > 0) & (kMpc_test < 4)]
            kMpc_test = kMpc_test[(kMpc_test > 0) & (kMpc_test < 4)]
            fit_p1d = poly_p1d.PolyP1D(kMpc_test, p1d_true, kmin_Mpc=1e-3, kmax_Mpc=4, deg=5)
            p1d_true = fit_p1d.P_Mpc(kMpc_test)   

            try:
                p1d_emu = emulator.emulate_p1d_Mpc(testing_data[m], kMpc_test)
                p1ds_err[ii,m, :] = (p1d_emu / p1d_true  -1)*100
            except Exception as e:
                logger.warning(f"Emulation failed for m={m}: {str(e)}. Skipping this iteration.")
                p1ds_err[ii,m, :] = np.nan
                continue
        
        
    np.savetxt(f'p1d_errors_all.txt', p1ds_err.reshape(-1, Nk))
    make_p1d_err_plot(p1ds_err, kMpc_test)


    logger.info("Plotting complete!")

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
