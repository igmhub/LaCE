def get_cosmo_params_dict_from_label(label):

    if label == "Planck18":
        cosmo_params = {
            "H0": 67.66,
            "mnu": 0,
            "omch2": 0.119,
            "ombh2": 0.0224,
            "omk": 0,
            "As": 2.105e-09,
            "ns": 0.9665,
            "nrun": 0.0,
            "pivot_scalar": 0.05,
            "w": -1.0,
        }
    else:
        raise ValueError("Unknown cosmology label =", label)

    return cosmo_params

