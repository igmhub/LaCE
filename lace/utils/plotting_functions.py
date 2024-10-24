import os
import numpy as np
import matplotlib.pyplot as plt
from lace.utils import poly_p1d
import matplotlib.cm as cm

def plot_p1d_vs_emulator(testing_data, emulator, save_path=None):
    """
    Plots the true P1D and the emulated P1D for each redshift in the testing data,
    along with a relative error panel below the main plot.
    
    Parameters:
    - testing_data: List of dictionaries containing the data for different redshifts.
    - emulator: An emulator object with a method emulate_p1d_Mpc() for generating P1D.
    
    Returns:
    - None (displays the plot).
    """
    # Initialize arrays to store true and emulated P1D values
    Nk = len(testing_data[0]['p1d_Mpc'][(testing_data[0]['k_Mpc'] > 0) & (testing_data[0]['k_Mpc'] < 4)])
    z = [d['z'] for d in testing_data if d['z'] < 4.8]
    Nz = len(z)
    
    p1ds_true = np.zeros((Nz, Nk))
    p1ds = np.zeros((Nz, Nk))

    # Use the 'inferno' colormap for a more intense color scheme
    cmap = cm.get_cmap('tab20', Nz)
    
    # Set up figure with two subplots: main plot and relative error plot
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8, 8))
    
    for m in range(Nz):
        if 'kF_Mpc' not in testing_data[m]:
            continue

        # Extract relevant data
        p1d_true = testing_data[m]['p1d_Mpc']
        kMpc = testing_data[m]['k_Mpc']
        
        # Filter kMpc values between 0 and 4
        kMpc_test = kMpc[(kMpc > 0) & (kMpc < 4)]
        p1d_true = p1d_true[(kMpc > 0) & (kMpc < 4)]
        
        # Fit the true P1D using a polynomial
        fit_p1d = poly_p1d.PolyP1D(kMpc_test, p1d_true, kmin_Mpc=1e-3, kmax_Mpc=4, deg=5)
        p1d_true = fit_p1d.P_Mpc(kMpc_test)
        
        # Emulate P1D using the emulator
        p1d = emulator.emulate_p1d_Mpc(testing_data[m], kMpc_test)

        # Store the results
        p1ds_true[m] = p1d_true
        p1ds[m] = p1d
        
        # Get the redshift and assign color
        redshift = testing_data[m]['z']
        color = cmap(m)
        
        # Plot on the main P1D plot
        ax1.scatter(kMpc_test, kMpc_test * p1d, label=f'$z={redshift:.1f}$', color=color, marker='^')
        ax1.plot(kMpc_test, kMpc_test * p1d_true, color=color)
        
        # Calculate and plot relative error on the second panel
        relative_error = (p1d_true - p1d) / p1d_true
        ax2.plot(kMpc_test, relative_error, label=f'$z={redshift:.1f}$', color=color)

    # Set labels and titles
    ax1.set_ylabel(r'$k \times P1D$', fontsize=14)
    ax2.set_xlabel(r'$k$ [1/Mpc]', fontsize=14)
    ax2.set_ylabel('Relative Error', fontsize=12)

    # Add grid and legend to the plots
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)
    ax2.grid(True, which='both', linestyle='--', alpha=0.7)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Set fonts to serif for the entire plot
    plt.rc('font', family='serif')

    # Save and show the plot
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()
    plt.close()
