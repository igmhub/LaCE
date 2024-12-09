from enum import StrEnum
import numpy as np

class Postprocessings(StrEnum):
    """
    Enumeration of possible archives in LaCE.
    """
    PEDERSEN21 = "Pedersen21"
    CABAYOL23 = "Cabayol23"
    POST_768 = "768_768"

# Constants for different post-processing configurations

PEDERSEN21_CONFIG = {
    "basedir": "/data/sim_suites/Australia20/",
    "n_phases": 2,
    "n_axes": 1,
    "p1d_label": "p1d",
    "sk_label": "Ns500_wM0.05",
    "basedir_params": "/data/sim_suites/Australia20/",
    "p1d_label_params": "p1d",
    "sk_label_params": "Ns500_wM0.05",
    "also_P3D": False,
    "tag_param": "parameter_redundant.json",
    "scalings_avail": [0, 1, 4],
    "training_average": "both",
    "training_val_scaling": 1,
    "training_z_min": 0,
    "training_z_max": 10,
    "testing_ind_rescaling": 0,
    "testing_z_min": 0,
    "testing_z_max": 10,
}

CABAYOL23_CONFIG = {
    "basedir": "/data/sim_suites/post_768/",
    "n_phases": 2,
    "n_axes": 3,
    "p1d_label": "p1d_stau",
    "sk_label": "Ns768_wM0.05",
    "basedir_params": "/data/sim_suites/Australia20/",
    "p1d_label_params": "p1d",
    "sk_label_params": "Ns500_wM0.05",
    "also_P3D": True,
    "tag_param": "parameter_redundant.json",
    "scalings_avail": list(range(5)),
    "training_average": "axes_phases_both",
    "training_val_scaling": "all",
    "training_z_min": 0,
    "training_z_max": 10,
    "testing_ind_rescaling": 0,
    "testing_z_min": 0,
    "testing_z_max": 10,
}

CONFIG_768_768 = {
    "basedir": "/data/sim_suites/post_768/",
    "n_phases": 2,
    "n_axes": 3,
    "p1d_label": "p1d_stau",
    "sk_label": "Ns768_wM0.05",
    "basedir_params": "/data/sim_suites/post_768/",
    "p1d_label_params": "p1d_stau",
    "sk_label_params": "Ns768_wM0.05",
    "also_P3D": True,
    "tag_param": "parameter.json",
    "scalings_avail": list(range(5)),
    "training_average": "axes_phases_both",
    "training_val_scaling": "all",
    "training_z_min": 0,
    "training_z_max": 10,
    "testing_ind_rescaling": 0,
    "testing_z_min": 0,
    "testing_z_max": 10,
}


# Mapping dictionaries for different postprocessing runs
PEDERSEN21_MAPPINGS = {
    "mpg_central": "central",
    "mpg_seed": "diffSeed_sim", 
    "mpg_growth": "h_sim",
    "mpg_neutrinos": "nu_sim",
    "mpg_curved": "curved_003",
    "mpg_running": "running_sim",
    "mpg_reio": "P18_sim"
}

CABAYOL23_MAPPINGS = {
    "mpg_central": "sim_pair_30",
    "mpg_seed": "diffSeed",
    "mpg_growth": "sim_pair_h", 
    "mpg_neutrinos": "nu_sim",
    "mpg_curved": "curved_003",
    "mpg_running": "running",
    "mpg_reio": "P18"
}

CABAYOL23_PARAM_MAPPINGS = {
    "mpg_central": "central",
    "mpg_seed": "diffSeed_sim",
    "mpg_growth": "h_sim",
    "mpg_neutrinos": "nu_sim", 
    "mpg_curved": "curved_003",
    "mpg_running": "running_sim",
    "mpg_reio": "P18_sim"
}


# Add these to existing config dictionaries
PEDERSEN21_CONFIG.update({
    "sim_mappings": PEDERSEN21_MAPPINGS,
    "param_mappings": PEDERSEN21_MAPPINGS  # Same as sim mappings
})

CABAYOL23_CONFIG.update({
    "sim_mappings": CABAYOL23_MAPPINGS,
    "param_mappings": {**CABAYOL23_MAPPINGS, **CABAYOL23_PARAM_MAPPINGS}
})

CONFIG_768_768.update({
    "sim_mappings": CABAYOL23_MAPPINGS,
    "param_mappings": CABAYOL23_MAPPINGS  # Same as sim mappings
})

# Base simulation lists
LIST_SIM_MPG_CUBE = [f"mpg_{i}" for i in range(30)]
LIST_SIM_MPG_TEST = list(CABAYOL23_MAPPINGS.keys())
LIST_ALL_SIMS_MPG = LIST_SIM_MPG_CUBE + LIST_SIM_MPG_TEST

# Simulation redshifts
LIST_SIM_REDSHIFTS_MPG = np.arange(2.0, 4.6, 0.25)

LIST_SIM_NYX_TEST = ["nyx_central", "nyx_seed", "nyx_3_ic", "nyx_wdm"]
LIST_SIM_NYX_CUBE = [f"nyx_{i}" for i in range(18)]
LIST_ALL_SIMS_NYX = LIST_SIM_NYX_TEST + LIST_SIM_NYX_CUBE
LIST_SIM_REDSHIFTS_NYX = [2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 5.0, 5.4]
