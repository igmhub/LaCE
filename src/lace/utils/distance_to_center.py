from lace.archive.gadget_archive import GadgetArchive
from lace.archive.nyx_archive import NyxArchive
import numpy as np
import os
import json

def calculate_distance_to_center(sim_test,
                           sim_suite='mpg',
                           archive=None,
                           emu_params=['Delta2_p', 'n_p', 'mF', 'sigT_Mpc', 'gamma', 'kF_Mpc']):
    """
    Calculate the normalized distance of a test simulation from the center of the parameter space defined by emulator parameters.

    Parameters:
    - sim_test (str): Identifier for the test simulation whose distance to the parameter space center is to be calculated.
    - sim_suite (str, optional): Specifies the simulation suite ('mpg' or 'nyx') to use for fetching the data. Defaults to 'mpg'.
    - archive (object, optional): An already instantiated archive object to use, avoiding reinitialization if passed. Defaults to None.
    - emu_params (list of str, optional): List of emulator parameter names to consider in the distance calculation. These parameters should be present in the simulation data.

    Returns:
    - numpy.ndarray: The normalized distances of the test simulation data from the central values of the parameter space.
    """
    # Initialize the appropriate archive if not provided
    if archive==None:
        if sim_suite=='mpg':
            archive = GadgetArchive(postproc="Cabayol23")
        elif sim_suite=='nyx':
            archive = NyxArchive(verbose=False)
        else:
            raise ValueError("Available sim suites are 'mpg' and 'nyx'") 

    if sim_suite == 'nyx':
        z_min = 2.19
    else:
        z_min=2
        
        
    central = archive.get_testing_data(sim_label=sim_suite+'_central')
        
    # Fetch all training data for the given emulator parameters            
    training_data_all=archive.get_training_data(emu_params)
    
    training_data = [training_data_all[i] for i in range(len(training_data_all)) 
                     if training_data_all[i]['ind_axis']=='average' 
                     and training_data_all[i]['ind_phase']=='average' 
                     and training_data_all[i]['val_scaling']==1 ]  
    
    training_data = np.array([[value
                    for key, value in training_data[i].items()
                    if key in emu_params] for i in range(len(training_data))])
    
    # Calculate the range (max - min) for each emulator parameter within the training data
    DeltaParams = training_data.max(0) - training_data.min(0) 
    
    
    test_sim_data = archive.get_testing_data(sim_label=sim_test,
                                            z_min=z_min)
    
    z_sim = np.round([d['z'] for d in test_sim_data],2)
    
    
    test_sim_data = np.array(
        [[test_sim_data[i].get(param, np.nan) for param in emu_params] 
         for i in range(len(test_sim_data))])
    
    
    
    central_data = np.array(
        [[central[i].get(param, np.nan) for param in emu_params] 
         for i in range(len(central)) 
         if np.round(central[i]['z'],2) in z_sim])
    
    if len(test_sim_data)==0:
        return np.nan

    
   # Calculate the normalized distance of the test simulation data from the central data    
    distance = np.nansum((np.abs(test_sim_data - central_data) / DeltaParams),1)
    
    return np.nanmean(distance)
            

def distances_to_dict(sim_suite='mpg', save_path=None):
    """- sim_suite (str, optional): Specifies the simulation suite ('mpg' or 'nyx') to use for fetching the data. Defaults to 'mpg'."""

    
    # Initialize the appropriate archive if not provided
    if sim_suite=='mpg':
        archive = GadgetArchive(postproc="Cabayol23")
    elif sim_suite=='nyx':
        archive = NyxArchive(verbose=False)
    else:
        raise ValueError("Available sim suites are 'mpg' and 'nyx'") 
    
    distances={}
    sim_list = set([archive.data[i]['sim_label'] for i in range(len(archive.data))])
    for sim_id in sim_list:
        print(sim_id)
        distance = calculate_distance_to_center(
            sim_test=sim_id,
            sim_suite=sim_suite,
            archive=archive)
        
        distances[sim_id] = distance    
     
    if save_path is not None:
        # Now, dump the distances dictionary to a JSON file
        with open(os.path.join(save_path,f'distances_{sim_suite}.json'), 'w') as f:
            json.dump(distances, f, indent=4)
    return distances
    
def get_distance_sim(sim_id, path_to_dict, sim_suite='mpg'):
    """
    Retrieve the distance for a given simulation ID from the distances dictionary.

    Parameters:
    - sim_id (str): The simulation ID for which to retrieve the distance.
    - sim_suite (str, optional): Specifies the simulation suite ('mpg' or 'nyx') to use for fetching the data. Defaults to 'mpg'.

    Returns:
    - float: The distance for the specified simulation ID.
    """    
    try:
        with open(path_to_dict, 'r') as f:
            distances = json.load(f)
        d = distances[sim_id]
    except KeyError:
        print(f"No distance found for simulation ID '{sim_id}'.")
    return d

def get_distance_simsuite(path_to_dict, sim_suite='mpg'):
    """
    Retrieve the distance for a given simulation ID from the distances dictionary.

    Parameters:
    - sim_id (str): The simulation ID for which to retrieve the distance.
    - sim_suite (str, optional): Specifies the simulation suite ('mpg' or 'nyx') to use for fetching the data. Defaults to 'mpg'.

    Returns:
    - float: The distance for the specified simulation ID.
    """    
    try:
        with open(path_to_dict, 'r') as f:
            distances = json.load(f)
    except KeyError:
        print(f"No distance found for simulation ID '{sim_id}'.")
    return distances