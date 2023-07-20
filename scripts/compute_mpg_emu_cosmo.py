import numpy as np
from lace.archive.gadget_archive import GadgetArchive
from lace.setup_simulations import read_genic
from lace.cosmo import camb_cosmo, fit_linP


def main():
    labels = ["z", "Delta2_p", "n_p", "alpha_p", "f_p"]
    archive = GadgetArchive(postproc="Pedersen21")

    val, ind = np.unique(archive.ind_snap, return_index=True)
    zs = np.zeros(ind.shape[0])
    for ii, iind in enumerate(ind):
        zs[ii] = archive.data[iind]["z"]

    complete_dict = []
    for jj in range(len(archive.list_sim)):
        out_dict = {}

        out_dict["sim"] = archive.list_sim[jj]
        _ = archive._sim2file_name(out_dict["sim"])
        old_lab = _[0]
        # print(out_dict["sim"], old_lab)

        genic_fname = (
            archive.fulldir_param + old_lab + "/sim_plus/paramfile.genic"
        )
        sim_cosmo_dict = read_genic.camb_from_genic(genic_fname)
        out_dict["cosmo"] = sim_cosmo_dict
        sim_cosmo = camb_cosmo.get_cosmology_from_dictionary(sim_cosmo_dict)
        linP_zs = fit_linP.get_linP_Mpc_zs(sim_cosmo, zs, archive.kp_Mpc)
        linP_zs = list(linP_zs)
        for ii in range(zs.shape[0]):
            linP_zs[ii]["z"] = zs[ii]

        out_dict["emu_cosmo"] = {}
        out_dict["emu_cosmo"]["kp_Mpc"] = archive.kp_Mpc

        for lab in labels:
            out_dict["emu_cosmo"][lab] = np.zeros(zs.shape[0])
            for ii in range(zs.shape[0]):
                out_dict["emu_cosmo"][lab][ii] = linP_zs[ii][lab]

        complete_dict.append(out_dict)

    fout = archive.fulldir + "mpg_emu_cosmo.npy"
    print(fout)
    np.save(fout, complete_dict)


if __name__ == "__main__":
    main()
