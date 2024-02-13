def check_convex_hull(archive, test_point, drop_sim=None, emu_params=["Delta2_p", "n_p", "mF", "sigT_Mpc", "gamma", "kF_Mpc"], verbose=False):
    """
    Check if the given test simulation falls within the convex hull defined by the training data.

    Args:
    archive (object): The archive object containing training and testing data.
    test_point (dict): Dictionary representing the test simulation point, with keys as parameter names and values as parameter values.
    drop_sim (str, optional): The label of the test simulation to drop from training data. Default is None.
    emu_params (list): List of parameters to consider for checking convex hull. Default is ["Delta2_p", "n_p", "mF", "sigT_Mpc", "gamma", "kF_Mpc"].

    Returns:
    bool: True if the test simulation is well covered by the training points (falls within the convex hull), False otherwise.
    """
    out = 0
    training_points = archive.get_training_data(emu_params, drop_sim=drop_sim)

    for param in emu_params:
        list_param = [d[param] for d in training_points]
        min_param, max_param = min(list_param), max(list_param)
        test_param = test_point[param]
        
        if test_param < min_param or test_param > max_param:
            print(f'{param} is outside the convex hull')
            out += 1
            
    if out == 0:
        if verbose is True:
            print('Simulation well covered by the training points')
        return True
    else:
        return False
