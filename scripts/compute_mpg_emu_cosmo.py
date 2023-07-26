import numpy as np
from lace.archive.gadget_archive import GadgetArchive
from lace.setup_simulations import read_genic
from lace.cosmo import camb_cosmo, fit_linP


def main():
    labels = ["z", "dkms_dMpc", "Delta2_p", "n_p", "alpha_p", "f_p"]
    archive = GadgetArchive(
        postproc="Pedersen21", kp_Mpc=0.7, force_recompute_linP_params=True
    )

    list_snap = np.unique(archive.ind_snap)

    all_dict = []
    # iterate over simulations
    for sim_label in archive.list_sim:
        sim_dict = {}
        sim_dict["sim_label"] = sim_label
        # iterate over snapshots
        for ind_snap in list_snap:
            ind = np.argwhere(
                (archive.sim_label == sim_label)
                & (archive.ind_snap == ind_snap)
            )[0, 0]
            if archive.data[ind]["ind_snap"] == list_snap[0]:
                sim_dict["cosmo_params"] = archive.data[ind]["cosmo_params"]
                sim_dict["linP_params"] = {}
                sim_dict["linP_params"]["kp_Mpc"] = archive.data[ind]["kp_Mpc"]
            for lab in labels:
                if archive.data[ind]["ind_snap"] == list_snap[0]:
                    sim_dict["linP_params"][lab] = np.zeros(list_snap.shape[0])
                sim_dict["linP_params"][lab][ind_snap] = archive.data[ind][lab]
        all_dict.append(sim_dict)

    fout = archive.fulldir + "mpg_emu_cosmo.npy"
    print(fout)
    np.save(fout, all_dict)


if __name__ == "__main__":
    main()
