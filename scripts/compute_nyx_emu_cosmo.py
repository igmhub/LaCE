import numpy as np
import h5py
from lace.archive.nyx_archive import NyxArchive, get_attrs
from lace.cosmo import camb_cosmo, fit_linP


def main():
    labels = ["z", "dkms_dMpc", "Delta2_p", "n_p", "alpha_p", "f_p"]
    nyx_fname = "/home/jchaves/Proyectos/projects/lya/data/nyx/models.hdf5"
    nyx_archive = NyxArchive(
        file_name=nyx_fname, kp_Mpc=0.7, force_recompute_linP_params=True
    )

    ff = h5py.File(nyx_fname, "r")

    sim_avail = np.array(list(ff.keys()))
    list_snap = np.array(list(nyx_archive.z_conv.keys()))

    all_dict = []
    # iterate over simulations
    for sim_label in sim_avail:
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
        all_dict.append(sim_dict)

    fout = "/home/jchaves/Proyectos/projects/lya/data/nyx/nyx_emu_cosmo.npy"
    print(fout)
    np.save(fout, all_dict)


if __name__ == "__main__":
    main()
