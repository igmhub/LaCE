import numpy as np


def get_cosmo_params_dict_from_label(label):

    if label == "Planck18":
        # Table 2 of https://arxiv.org/abs/1807.06209.pdf (T&E+lensing+BAO) w/ mnu=0
        cosmo_params = {
            "H0": 67.66,
            "mnu": 0,
            "omch2": 0.11933,
            "ombh2": 0.02242,
            "omk": 0,
            "As": np.exp(3.047) * 1e-10,
            "ns": 0.9665,
            "nrun": 0.0,
            "pivot_scalar": 0.05,
            "w": -1.0,
        }
    elif label == "Planck18_mnu":
        # Table 2 of https://arxiv.org/abs/1807.06209.pdf (T&E+lensing+BAO) w/ mnu=0.06
        cosmo_params = {
            "H0": 67.66,
            "mnu": 0.06,
            "omch2": 0.11933,
            "ombh2": 0.02242,
            "omk": 0,
            "As": np.exp(3.047) * 1e-10,
            "ns": 0.9665,
            "nrun": 0.0,
            "pivot_scalar": 0.05,
            "w": -1.0,
        }
    elif label == "Planck18_noBAO":
        # Table 2 of https://arxiv.org/abs/1807.06209.pdf (T&E+lensing) w/ mnu=0.06
        cosmo_params = {
            "H0": 67.36,
            "mnu": 0.06,
            "omch2": 0.1200,
            "ombh2": 0.02237,
            "omk": 0,
            "As": np.exp(3.044) * 1e-10,
            "ns": 0.9646,
            "nrun": 0.0,
            "pivot_scalar": 0.05,
            "w": -1.0,
        }
    else:
        raise ValueError("Unknown cosmology label =", label)

    return cosmo_params
