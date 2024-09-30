import pytest
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import lace
from lace.emulator.nn_emulator import NNEmulator
from lace.archive import gadget_archive
from lace.utils.plotting_functions import plot_p1d_vs_emulator

@pytest.fixture
def output_dir(pytestconfig):
    # Get the custom argument passed with pytest
    return pytestconfig.getoption("output_dir")

def test(output_dir):
    """
    Function to plot emulated P1D using specified archive and save plots to output directory.
    Parameters:
    output_dir (str): Directory to save the generated plots.
    """
    archive_name = 'Gadget'
    repo = os.path.dirname(lace.__path__[0]) + "/"
    
    emu_params = ['Delta2_p', 'n_p', 'mF', 'sigT_Mpc', 'gamma', 'kF_Mpc']
    training_set = 'Cabayol23'
    emulator_label = 'Cabayol23+'
    model_path = f'{repo}data/NNmodels/Cabayol23+/Cabayol23+_drop_sim'
    archive = gadget_archive.GadgetArchive(postproc="Cabayol23")
    
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
        
        testing_data = archive.get_testing_data(sim_label=f'{sim}')
        if sim != 'nyx_central':
            testing_data = [d for d in testing_data if d['val_scaling'] == 1]
        
        save_path = os.path.join(output_dir, f'{sim}.png')
        plot_p1d_vs_emulator(testing_data, emulator, save_path=save_path)
    
    return

def pytest_addoption(parser):
    # Add custom options to pytest command
    parser.addoption("--output_dir", action="store", default="tmp/validation_figures", help="Directory to save plots")
