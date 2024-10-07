import os
import numpy as np
from matplotlib import pyplot as plt
from lace.utils import poly_p1d
import matplotlib.cm as cm

def plot_p1d_vs_emulator(testing_data, emulator, save_path=None):
    """
    Plots the true P1D and the emulated P1D for each redshift in the testing data.
    
    Parameters:
    - testing_data: List of dictionaries containing the data for different redshifts.
    - emulator: An emulator object with a method emulate_p1d_Mpc() for generating P1D.
    
    Returns:
    - None (displays the plot).
    """
    # Initialize arrays to store true and emulated P1D values
    Nk = len(testing_data[0]['p1d_Mpc'][(testing_data[0]['k_Mpc']>0)&(testing_data[0]['k_Mpc']<4)])
    z = [d['z'] for d in testing_data if d['z']<4.8 ]
    Nz = len(z)
    
    p1ds_true = np.zeros(shape=(Nz, Nk))
    p1ds = np.zeros(shape=(Nz, Nk))

    # Create a colormap with 11 unique colors
    cmap = cm.get_cmap('viridis', Nz)
    
    plt.figure()

    # Loop through the entries in the testing data
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
        
        # Emulate P1D using the provided emulator
        p1d = emulator.emulate_p1d_Mpc(testing_data[m], kMpc_test)

        # Store the results
        p1ds_true[m] = p1d_true
        p1ds[m] = p1d

        # Get the redshift and assign a unique color from the colormap
        redshift = testing_data[m]['z']
        color = cmap(m)

        # Plot the true and emulated P1D
        plt.scatter(kMpc_test, kMpc_test * p1d, label=f'$z={redshift:.1f}$', color=color, marker='^')
        plt.plot(kMpc_test, kMpc_test * p1d_true, color=color)

    # Set axis labels
    plt.xlabel(r'$k$ [1/Mpc]')
    plt.ylabel(r'$k$ * P1D')

    # Add the legend outside the plot for better readability
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Save the plot
    if save_path != None:
        plt.savefig(save_path, bbox_inches='tight')

    # Show the plot
    plt.show()
    plt.close()
    return