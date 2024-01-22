from lace.archive.gadget_archive import GadgetArchive
from lace.cosmo.thermal_broadening import thermal_broadening_kms
import numpy as np
import os


def main():
    """Save IGM history of all mpg simulations"""

    cabayol23_archive = GadgetArchive(postproc="Cabayol23")
    emu_params = [
        "Delta2_p",
        "n_p",
        "mF",
        "sigT_Mpc",
        "gamma",
        "kF_Mpc",
    ]
    training = cabayol23_archive.get_training_data(emu_params)

    # hypercube

    ind_axis = "average"
    ind_phase = "average"
    val_scaling = 1

    list_snap = np.unique(cabayol23_archive.ind_snap)
    nz = len(list_snap)
    dict_index = {}

    for sim_label in cabayol23_archive.list_sim_cube:
        cosmo_params, linP_params = cabayol23_archive._get_emu_cosmo(sim_label)
        dict_index[sim_label] = {}
        dict_index[sim_label]["z"] = np.zeros(nz)
        dict_index[sim_label]["tau_eff"] = np.zeros(nz)
        dict_index[sim_label]["gamma"] = np.zeros(nz)
        dict_index[sim_label]["sigT_kms"] = np.zeros(nz)
        dict_index[sim_label]["kF_kms"] = np.zeros(nz)
        for ind_snap in list_snap:
            for ind_book in range(len(training)):
                if (
                    (training[ind_book]["ind_snap"] == ind_snap)
                    & (training[ind_book]["ind_axis"] == ind_axis)
                    & (training[ind_book]["ind_phase"] == ind_phase)
                    & (training[ind_book]["sim_label"] == sim_label)
                    & (training[ind_book]["val_scaling"] == val_scaling)
                ):
                    dict_index[sim_label]["z"][ind_snap] = training[ind_book][
                        "z"
                    ]
                    dict_index[sim_label]["tau_eff"][ind_snap] = -np.log(
                        training[ind_book]["mF"]
                    )
                    dict_index[sim_label]["gamma"][ind_snap] = training[
                        ind_book
                    ]["gamma"]
                    _ = thermal_broadening_kms(training[ind_book]["T0"])
                    dict_index[sim_label]["sigT_kms"][ind_snap] = _

                    ind_z = np.argwhere(
                        training[ind_book]["z"] == linP_params["z"]
                    )[0, 0]
                    _ = (
                        training[ind_book]["kF_Mpc"]
                        / linP_params["dkms_dMpc"][ind_z]
                    )
                    dict_index[sim_label]["kF_kms"][ind_snap] = _

    # testing

    for sim_label in cabayol23_archive.list_sim_test:
        cosmo_params, linP_params = cabayol23_archive._get_emu_cosmo(sim_label)
        dict_index[sim_label] = {}
        dict_index[sim_label]["z"] = np.zeros(nz)
        dict_index[sim_label]["tau_eff"] = np.zeros(nz)
        dict_index[sim_label]["gamma"] = np.zeros(nz)
        dict_index[sim_label]["sigT_kms"] = np.zeros(nz)
        dict_index[sim_label]["kF_kms"] = np.zeros(nz)
        testing = cabayol23_archive.get_testing_data(sim_label)
        for ind_snap in list_snap:
            for ind_book in range(len(testing)):
                if (
                    (testing[ind_book]["ind_snap"] == ind_snap)
                    & (testing[ind_book]["ind_axis"] == ind_axis)
                    & (testing[ind_book]["ind_phase"] == ind_phase)
                    & (testing[ind_book]["sim_label"] == sim_label)
                    & (testing[ind_book]["val_scaling"] == val_scaling)
                ):
                    dict_index[sim_label]["z"][ind_snap] = testing[ind_book][
                        "z"
                    ]
                    dict_index[sim_label]["tau_eff"][ind_snap] = -np.log(
                        testing[ind_book]["mF"]
                    )
                    dict_index[sim_label]["gamma"][ind_snap] = testing[
                        ind_book
                    ]["gamma"]
                    _ = thermal_broadening_kms(testing[ind_book]["T0"])
                    dict_index[sim_label]["sigT_kms"][ind_snap] = _

                    ind_z = np.argwhere(
                        testing[ind_book]["z"] == linP_params["z"]
                    )[0, 0]
                    _ = (
                        testing[ind_book]["kF_Mpc"]
                        / linP_params["dkms_dMpc"][ind_z]
                    )
                    dict_index[sim_label]["kF_kms"][ind_snap] = _

    folder = os.environ["LACE_REPO"] + "/src/lace/data/sim_suites/Australia20/"
    np.save(folder + "IGM_histories.npy", dict_index)
    folder = os.environ["LACE_REPO"] + "/src/lace/data/sim_suites/post_768/"
    np.save(folder + "IGM_histories.npy", dict_index)


if __name__ == "__main__":
    main()
