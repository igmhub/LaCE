from lace.archive.nyx_archive import NyxArchive
from lace.cosmo.thermal_broadening import thermal_broadening_kms
import numpy as np
import os


def main():
    """Save IGM history of all mpg simulations"""

    nyx_version = "Oct2023"
    nyx_version = "models_Nyx_Mar2025_with_CGAN_val_3axes"
    nyx_archive = NyxArchive(nyx_version=nyx_version)
    emu_params = [
        "Delta2_p",
        "n_p",
        "mF",
        "sigT_Mpc",
        "gamma",
        "kF_Mpc",
    ]
    training = nyx_archive.get_training_data(emu_params)

    # hypercube

    ind_axis = "average"
    ind_phase = "average"
    ind_rescaling = 0

    list_snap = np.unique(nyx_archive.ind_snap)
    rsim_conv = {value: key for key, value in nyx_archive.sim_conv.items()}
    nz = len(list_snap)
    dict_index = {}

    for sim_label in nyx_archive.list_sim_cube:
        if sim_label == "nyx_14":
            continue

        cosmo_params, linP_params, star_params = nyx_archive._get_emu_cosmo(
            None, rsim_conv[sim_label]
        )

        for ind_rescaling in range(21):
            lab = sim_label + "_" + str(ind_rescaling)
            dict_index[lab] = {}
            dict_index[lab]["z"] = np.zeros(nz)
            dict_index[lab]["tau_eff"] = np.zeros(nz)
            dict_index[lab]["mF"] = np.zeros(nz)
            dict_index[lab]["gamma"] = np.zeros(nz)
            dict_index[lab]["sigT_kms"] = np.zeros(nz)
            dict_index[lab]["sigT_Mpc"] = np.zeros(nz)
            dict_index[lab]["kF_kms"] = np.zeros(nz)
            dict_index[lab]["kF_Mpc"] = np.zeros(nz)

            for ind_snap in list_snap:
                for ind_book in range(len(training)):
                    if (
                        (training[ind_book]["ind_snap"] == ind_snap)
                        & (training[ind_book]["ind_axis"] == ind_axis)
                        & (training[ind_book]["ind_phase"] == ind_phase)
                        & (training[ind_book]["sim_label"] == sim_label)
                        & (training[ind_book]["ind_rescaling"] == ind_rescaling)
                    ):
                        dict_index[lab]["z"][ind_snap] = training[ind_book]["z"]
                        dict_index[lab]["mF"][ind_snap] = training[ind_book][
                            "mF"
                        ]
                        dict_index[lab]["gamma"][ind_snap] = training[ind_book][
                            "gamma"
                        ]
                        dict_index[lab]["kF_Mpc"][ind_snap] = training[
                            ind_book
                        ]["kF_Mpc"]
                        dict_index[lab]["sigT_Mpc"][ind_snap] = training[
                            ind_book
                        ]["sigT_Mpc"]

                        dict_index[lab]["tau_eff"][ind_snap] = -np.log(
                            training[ind_book]["mF"]
                        )
                        _ = thermal_broadening_kms(training[ind_book]["T0"])
                        dict_index[lab]["sigT_kms"][ind_snap] = _

                        ind_z = np.argwhere(
                            np.round(training[ind_book]["z"], 2)
                            == linP_params["z"]
                        )[0, 0]
                        _ = (
                            training[ind_book]["kF_Mpc"]
                            / linP_params["dkms_dMpc"][ind_z]
                        )
                        dict_index[lab]["kF_kms"][ind_snap] = _

    # testing

    for sim_label in nyx_archive.list_sim_test:
        cosmo_params, linP_params, star_params = nyx_archive._get_emu_cosmo(
            None, rsim_conv[sim_label]
        )
        if sim_label == "nyx_central":
            ind_rescaling = 1
        else:
            ind_rescaling = None
        testing = nyx_archive.get_testing_data(
            sim_label, ind_rescaling=ind_rescaling
        )

        lab = sim_label
        dict_index[lab] = {}
        dict_index[lab]["z"] = np.zeros(nz)
        dict_index[lab]["tau_eff"] = np.zeros(nz)
        dict_index[lab]["mF"] = np.zeros(nz)
        dict_index[lab]["gamma"] = np.zeros(nz)
        dict_index[lab]["sigT_kms"] = np.zeros(nz)
        dict_index[lab]["sigT_Mpc"] = np.zeros(nz)
        dict_index[lab]["kF_kms"] = np.zeros(nz)
        dict_index[lab]["kF_Mpc"] = np.zeros(nz)

        for ind_snap in list_snap:
            for ind_book in range(len(testing)):
                if (
                    (testing[ind_book]["ind_snap"] == ind_snap)
                    & (testing[ind_book]["ind_axis"] == ind_axis)
                    & (testing[ind_book]["ind_phase"] == ind_phase)
                    & (testing[ind_book]["sim_label"] == sim_label)
                ):
                    dict_index[lab]["z"][ind_snap] = testing[ind_book]["z"]
                    dict_index[lab]["mF"][ind_snap] = testing[ind_book]["mF"]
                    if "gamma" in testing[ind_book]:
                        dict_index[lab]["gamma"][ind_snap] = testing[ind_book][
                            "gamma"
                        ]
                    else:
                        dict_index[lab]["gamma"][ind_snap] = np.nan

                    if "sigT_Mpc" in testing[ind_book]:
                        dict_index[lab]["sigT_Mpc"][ind_snap] = testing[
                            ind_book
                        ]["sigT_Mpc"]
                    else:
                        dict_index[lab]["sigT_Mpc"][ind_snap] = np.nan

                    dict_index[lab]["tau_eff"][ind_snap] = -np.log(
                        testing[ind_book]["mF"]
                    )

                    if "T0" in testing[ind_book]:
                        _ = thermal_broadening_kms(testing[ind_book]["T0"])
                        dict_index[lab]["sigT_kms"][ind_snap] = _
                    else:
                        dict_index[lab]["sigT_kms"][ind_snap] = np.nan

                    ind_z = np.argwhere(
                        np.round(testing[ind_book]["z"], 2) == linP_params["z"]
                    )[0, 0]
                    if "kF_Mpc" in testing[ind_book]:
                        _ = (
                            testing[ind_book]["kF_Mpc"]
                            / linP_params["dkms_dMpc"][ind_z]
                        )
                        dict_index[lab]["kF_kms"][ind_snap] = _
                        dict_index[lab]["kF_Mpc"][ind_snap] = testing[ind_book][
                            "kF_Mpc"
                        ]
                    else:
                        print(
                            "no kF_Mpc in ", sim_label, testing[ind_book]["z"]
                        )

    folder = os.environ["NYX_PATH"]
    np.save(folder + "/IGM_histories.npy", dict_index)


if __name__ == "__main__":
    main()
