import numpy as np
import h5py
from lace.archive.nyx_archive import NyxArchive, get_attrs
from lace.cosmo import camb_cosmo, fit_linP


def main():
    nyx_fname = "/home/jchaves/Proyectos/projects/lya/data/nyx/models.hdf5"
    nyx_archive = NyxArchive(file_name=nyx_fname)
    ff = h5py.File(nyx_fname, "r")

    sim_avail = np.array(list(ff.keys()))

    nz = len(nyx_archive.list_sim_redshifts)

    complete_dict = []
    for isim in sim_avail:
        out_dict = {}

        out_dict["sim"] = nyx_archive.sim_conv[isim]
        # this simulation seems to have issues
        if isim == "cosmo_grid_14":
            continue

        sim_params = get_attrs(ff[isim])
        if isim == "fiducial":
            sim_params["A_s"] = 2.10e-9
            sim_params["n_s"] = 0.966
            sim_params["h"] = sim_params["H_0"] / 100
        out_dict["cosmo"] = sim_params
        sim_cosmo = camb_cosmo.get_Nyx_cosmology(sim_params)

        # compute linear power parameters at each z (will call CAMB)
        linP_zs = fit_linP.get_linP_Mpc_zs(
            sim_cosmo, nyx_archive.list_sim_redshifts, nyx_archive.kp_Mpc
        )
        # compute conversion from Mpc to km/s using cosmology
        dkms_dMpc_zs = camb_cosmo.dkms_dMpc(
            sim_cosmo, z=np.array(nyx_archive.list_sim_redshifts)
        )

        out_dict["emu_cosmo"] = {}
        out_dict["emu_cosmo"]["kp_Mpc"] = nyx_archive.kp_Mpc
        labels = ["Delta2_p", "n_p", "alpha_p", "f_p"]
        for lab in labels:
            out_dict["emu_cosmo"][lab] = np.zeros(nz)
            for ii in range(nz):
                out_dict["emu_cosmo"][lab][ii] = linP_zs[ii][lab]

        out_dict["emu_cosmo"]["dkms_dMpc"] = np.zeros(nz)
        out_dict["emu_cosmo"]["z"] = np.zeros(nz)
        for ii in range(nz):
            out_dict["emu_cosmo"]["dkms_dMpc"][ii] = dkms_dMpc_zs[ii]
            out_dict["emu_cosmo"]["z"][ii] = nyx_archive.list_sim_redshifts[ii]

        complete_dict.append(out_dict)

    fout = "/home/jchaves/Proyectos/projects/lya/data/nyx/nyx_emu_cosmo.npy"
    print(fout)
    np.save(fout, complete_dict)


if __name__ == "__main__":
    main()
