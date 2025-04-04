import numpy as np
import h5py
from lace.archive.nyx_archive import NyxArchive
from lace.cosmo import camb_cosmo, fit_linP
import os


def main():
    """Runs script to speed up the reading of Nyx data"""

    labels = ["z", "dkms_dMpc", "Delta2_p", "n_p", "alpha_p", "f_p"]
    # os.environ["NYX_PATH"] = # path to the Nyx files in your local computer

    # file containing Nyx data
    # nyx_version = "Oct2023"
    # nyx_version = "Jul2024"
    nyx_version = "models_Nyx_Mar2025_with_CGAN_val_3axes"
    nyx_fname = os.environ["NYX_PATH"] + "/" + nyx_version + ".hdf5"
    # file that will be written containing cosmo data for Nyx file
    cosmo_fname = (
        os.environ["NYX_PATH"] + "/nyx_emu_cosmo_" + nyx_version + ".npy"
    )

    nyx_archive = NyxArchive(
        nyx_version=nyx_version, force_recompute_linP_params=True, kp_Mpc=0.7
    )

    ff = h5py.File(nyx_fname, "r")

    sim_avail = np.array(list(ff.keys()))
    list_snap = np.array(list(nyx_archive.z_conv.keys()))

    all_dict = {}
    # iterate over simulations
    for sim_label in sim_avail:
        # this simulation seems to have issues
        if sim_label == "cosmo_grid_14":
            continue
        # # contained in CGAN_4096_base
        # elif sim_label == "CGAN_4096_val":
        #     continue

        isim = nyx_archive.sim_conv[sim_label]
        sim_dict = {}
        sim_dict["sim_label"] = isim
        # iterate over snapshots
        first = True
        for ind_snap in list_snap:
            isnap = nyx_archive.z_conv[ind_snap]
            ind = np.argwhere(
                (nyx_archive.sim_label == isim)
                & (nyx_archive.ind_snap == isnap)
            )[:, 0]
            # we are missing some snapshots for multiple sims
            if len(ind) != 0:
                ind = ind[0]
            else:
                continue

            if first:
                sim_dict["cosmo_params"] = nyx_archive.data[ind]["cosmo_params"]
                sim_dict["star_params"] = nyx_archive.data[ind]["star_params"]
                sim_dict["linP_params"] = {}
                sim_dict["linP_params"]["kp_Mpc"] = nyx_archive.data[ind][
                    "kp_Mpc"
                ]

            for lab in labels:
                if first:
                    sim_dict["linP_params"][lab] = np.zeros(list_snap.shape[0])

                sim_dict["linP_params"][lab][isnap] = nyx_archive.data[ind][lab]

            if first:
                first = False
        all_dict[isim] = sim_dict

    np.save(cosmo_fname, all_dict)


if __name__ == "__main__":
    main()
