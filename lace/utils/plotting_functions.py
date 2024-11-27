import os
import numpy as np
import matplotlib.pyplot as plt
from lace.utils import poly_p1d
import matplotlib.cm as cm
import corner

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

def create_corner_plot(
    main_data_dict, 
    labels=None, 
    figsize=(8, 8), 
    additional_data_dicts=None, 
    additional_colors=None, 
    legend_labels=None, 
    truth_values=None,  # Add truth values as a parameter
    save_path=None
):
    """
    Creates a triangular corner plot with 1-sigma contours, labels, optional overlayed contours, and truth values with a legend.

    Parameters:
        main_data_dict (dict): Main dictionary where keys are parameter names, and values are 1D arrays of parameter values.
        labels (list, optional): List of strings for axis labels (LaTeX supported). Defaults to dictionary keys.
        figsize (tuple, optional): Figure size for the plot. Defaults to (8, 8).
        additional_data_dicts (list, optional): List of additional dictionaries with parameter data for overlaying contours.
        additional_colors (list, optional): List of colors for additional datasets. Defaults to None (assigns random colors).
        legend_labels (list, optional): List of labels for the legend. Defaults to "Dataset 1", "Dataset 2", etc.
        truth_values (list, optional): List of truth values corresponding to the parameter labels. Defaults to None (no truth points).
        save_path (str, optional): File path to save the plot. Defaults to None (won't save).

    Returns:
        matplotlib.figure.Figure: The generated corner plot figure.
    """    

    # Convert the main dictionary to a 2D array
    main_data = np.array(list(main_data_dict.values())).T  # Shape (N_samples, N_parameters)
    
    # Default labels to dictionary keys if not provided
    if labels is None:
        labels = list(main_data_dict.keys())
    
    # Generate the base corner plot with the main data
    figure = corner.corner(
        main_data,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],  # Show 1-sigma intervals
        show_titles=False,
        plot_density=False,
        plot_datapoints=False,
        color="steelblue",
        fill_contours=False,
        levels=(0.68, 0.95),  # 1-sigma and 2-sigma contours
        smooth=1.0,  # Smoothing for the KDE
        scale_hist=True,
        figsize=figsize,
    )
 
    ndim = len(labels)
    axes = np.array(figure.axes).reshape((ndim, ndim))

    if truth_values:
        # Loop over the diagonal
        for i in range(ndim):
            ax = axes[i, i]
            ax.axvline(truth_values[i], color="black")

        """# Loop over the histograms
        for yi in range(ndim):
            for xi in range(yi):
                ax = axes[yi, xi]
                ax.axvline(truth_values[xi], color="black")
                ax.axvline(truth_values[xi], color="black")
                ax.axhline(truth_values[yi], color="black")
                ax.axhline(truth_values[yi], color="black")
                ax.plot(truth_values[xi], truth_values[yi], "black")
                ax.plot(truth_values[xi], truth_values[yi], "black")"""

    # Overlay contours for additional datasets if provided
    legend_handles = []
    if additional_data_dicts:
        if additional_colors is None:
            # Generate random colors if not specified
            additional_colors = [f"C{i}" for i in range(len(additional_data_dicts))]
            
        
        
        if legend_labels is None:
            # Generate default labels if not specified
            legend_labels = [f"Dataset {i+1}" for i in range(len(additional_data_dicts))]
        
        for idx, additional_data_dict in enumerate(additional_data_dicts):
            # Convert the additional dictionary to a 2D array
            additional_data = np.array(list(additional_data_dict.values())).T  # Shape (N_samples, N_parameters)
            color = additional_colors[idx % len(additional_colors)]
            
            # Overlay contours on existing axes
            figure = corner.corner(
                additional_data,
                fig=figure,  # Use the existing figure
                color=color,
                quantiles=[0.16, 0.5, 0.84],
                plot_density=False,
                plot_datapoints=False,
                fill_contours=False,  # No fill for overlays
                levels=(0.68, 0.95),  # 1-sigma and 2-sigma contours
                smooth=1.0,
            )
            
            # Add to legend handles
            legend_handles.append(plt.Line2D([], [], color=color, label=legend_labels[idx+1]))
    
    # Add legend for the main data
    legend_handles.append(plt.Line2D([], [], color="steelblue", label=legend_labels[0]))

    # Add legend to the figure
    figure.legend(handles=legend_handles, loc='upper right', fontsize=10, frameon=True)
    
    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Corner plot saved to {save_path}")

    plt.show()
