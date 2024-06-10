# Import necessary modules
from lace.archive.gadget_archive import GadgetArchive
from lace.emulator.nn_emulator import NNEmulator
from lace.utils import poly_p1d
import numpy as np

def test():
    # Define the parameters for the emulator
    emu_params = ['Delta2_p', 'n_p', 'mF', 'sigT_Mpc', 'gamma', 'kF_Mpc']
    
    # Initialize a GadgetArchive instance for postprocessing data
    archive = GadgetArchive(postproc="Cabayol23",
                           force_recompute_linP_params=True,
                           kp_Mpc=0.7)
    
    # Retrieve testing data from the archive
    testing_data = archive.get_testing_data(sim_label='mpg_central')
    
    # Extract kMpc and p1d_true data from testing data
    kMpc = testing_data[0]['k_Mpc']
    p1d_true = testing_data[4]['p1d_Mpc']
    p1d_true = p1d_true[(kMpc > 0) & (kMpc < 4)]
    kMpc = kMpc[(kMpc > 0) & (kMpc < 4)]
    
    # Fit a polynomial to the true p1d data
    fit_p1d = poly_p1d.PolyP1D(kMpc, 
                               p1d_true, 
                               kmin_Mpc=1e-3, 
                               kmax_Mpc=4, 
                               deg=5)
    
    p1d_true = fit_p1d.P_Mpc(kMpc)
    
    # Set up the emulator
    emulator_C23 = NNEmulator(training_set='Cabayol23', 
                              emulator_label='Cabayol23+',
                              model_path='NNmodels/Cabayol23+/Cabayol23+.pt',
                              train=False)
    
    # Emulate p1d_Mpc data using the emulator
    p1d = emulator_C23.emulate_p1d_Mpc(testing_data[4], kMpc)
    
    # Calculate the percentage error between the emulated and true p1d data
    percent_error = np.mean((p1d / p1d_true - 1) * 100)
    
    # Assert that the percentage error is less than 1%
    assert percent_error < 1
# Call the function to execute the test
test()