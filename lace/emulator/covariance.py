import numpy as np
from scipy.optimize import curve_fit
from lace.emulator.gp_emulator_multi import GPEmulator
from cup1d.likelihood.interface_emu import P1D_emulator


def data_for_l10_lace(archive, emulator_label, suite="nyx"):
    """
    Compute the covariance matrix of the emulator
    """

    # number of simulations
    if suite == "nyx":
        nsam = 14
    else:
        nsam = 30

    # load testing data from a random simulations of the hypercube
    # to get number of redshifts, k_Mpc, etc
    testing_data = archive.get_testing_data(suite + "_2")

    zz = []
    for iz in range(len(testing_data)):
        zz.append(testing_data[iz]["z"])
    zz = np.unique(np.array(zz))
    nz = len(zz)

    # load emulator to define kmax
    emulator = GPEmulator(emulator_label=emulator_label, train=False)

    _k_Mpc = testing_data[0]["k_Mpc"]
    ind = (_k_Mpc < emulator.kmax_Mpc) & (_k_Mpc > 0)
    k_Mpc = _k_Mpc[ind]
    k_fit = k_Mpc / emulator.kmax_Mpc

    # smooth simulation data and evaluate emulator
    p1d_Mpc_orig = np.zeros((nsam, nz, k_Mpc.shape[0]))
    p1d_Mpc_sm = np.zeros((nsam, nz, k_Mpc.shape[0]))
    p1d_Mpc_emu = np.zeros((nsam, nz, k_Mpc.shape[0]))
    mask = np.ones((nsam, nz), dtype=bool)
    for isim in range(nsam):
        # get testing data from target sim
        testing_data = archive.get_testing_data(suite + "_" + str(isim))
        if np.allclose(testing_data[0]["k_Mpc"][ind], k_Mpc) == False:
            raise ValueError(
                "k_Mpc not the same for simulation", suite + "_" + str(isim)
            )
        # Interpolate kF, only relevant for Nyx since all values are not available
        # and we would like to evaluate the emulator
        if suite == "nyx":
            kF = np.zeros(nz)
            for iz in range(len(testing_data)):
                diffz = np.abs(zz - testing_data[iz]["z"])
                iiz = np.argmin(diffz)
                if diffz[iiz] > 0.01:
                    continue
                kF[iiz] = testing_data[iz]["kF_Mpc"]
            _ = np.isfinite(kF) & (kF > 0)
            kFinter = np.interp(zz, zz[_], kF[_])

        # get emulator trained without target sim
        emulator = GPEmulator(
            emulator_label=emulator_label,
            train=False,
            drop_sim=suite + "_" + str(isim),
        )

        for iz in range(len(testing_data)):
            diffz = np.abs(zz - testing_data[iz]["z"])
            iiz = np.argmin(diffz)
            if diffz[iiz] > 0.01:
                continue

            norm = np.interp(
                k_Mpc,
                emulator.input_norm["k_Mpc"],
                emulator.norm_imF(testing_data[iz]["mF"]),
            )
            yfit = np.log(testing_data[iz]["p1d_Mpc"][ind] / norm)
            popt, _ = curve_fit(emulator.func_poly, k_fit, yfit)
            p1d_Mpc_sm[isim, iiz] = norm * np.exp(emulator.func_poly(k_fit, *popt))
            p1d_Mpc_orig[isim, iiz] = testing_data[iz]["p1d_Mpc"][ind]

            if suite == "nyx":
                int_data = testing_data[iz].copy()
                if ("kF_Mpc" not in int_data) | (
                    np.isfinite(int_data["kF_Mpc"]) == False
                ):
                    int_data["kF_Mpc"] = kFinter[iiz]
            else:
                int_data = testing_data[iz]

            use_data = True
            for par in emulator.emu_params:
                if par not in int_data:
                    use_data = False
                elif np.isfinite(int_data[par]) == False:
                    use_data = False

            if use_data:
                p1d_Mpc_emu[isim, iiz] = emulator.emulate_p1d_Mpc(int_data, k_Mpc)
            else:
                mask[isim, iiz] = False

    return zz, k_Mpc, p1d_Mpc_orig, p1d_Mpc_sm, p1d_Mpc_emu, mask


def data_for_l10_forest(archive):
    """
    Compute the covariance matrix of the emulator
    """

    # number of simulations
    nsam = len(archive.list_sim_cube)
    suite = archive.list_sim_cube[0][:3]

    # load testing data from a random simulations of the hypercube
    # to get number of redshifts, k_Mpc, etc

    testing_data = []
    for sim in archive.training_data:
        if sim["sim_label"] == suite + "_2":
            testing_data.append(sim)

    zz = []
    for iz in range(len(testing_data)):
        zz.append(testing_data[iz]["z"])
    zz = np.unique(np.array(zz))
    nz = len(zz)

    # load emulator to define kmax
    emulator = P1D_emulator()

    _k_Mpc = testing_data[0]["k_Mpc"]
    ind = (_k_Mpc < emulator.kmax_Mpc) & (_k_Mpc > 0)
    k_Mpc = _k_Mpc[ind]
    k_fit = k_Mpc / emulator.kmax_Mpc

    # smooth simulation data and evaluate emulator
    p1d_Mpc_orig = np.zeros((nsam, nz, k_Mpc.shape[0]))
    p1d_Mpc_sm = np.zeros((nsam, nz, k_Mpc.shape[0]))
    p1d_Mpc_emu = np.zeros((nsam, nz, k_Mpc.shape[0]))
    mask = np.ones((nsam, nz), dtype=bool)
    for isim in range(nsam):
        # get testing data from target sim
        sim_label = suite + "_" + str(isim)
        print(sim_label)
        testing_data = []
        for sim in archive.training_data:
            if (sim["sim_label"] == sim_label) and (sim["val_scaling"] == 1.0):
                testing_data.append(sim)

        if np.allclose(testing_data[0]["k_Mpc"][ind], k_Mpc) == False:
            raise ValueError(
                "k_Mpc not the same for simulation", suite + "_" + str(isim)
            )

        # get emulator trained without target sim
        name_emu = "l1O/forest_mpg_l1O_" + str(isim)
        emulator = P1D_emulator(name_emu=name_emu)
        emulator.set_cosmo(testing_data[0]["cosmo_params"])

        for iz in range(len(testing_data)):
            diffz = np.abs(zz - testing_data[iz]["z"])
            iiz = np.argmin(diffz)
            if diffz[iiz] > 0.01:
                continue

            ysm = emulator.model_Arinyo.P1D_Mpc(
                testing_data[iz]["z"],
                testing_data[iz]["p1d_Mpc"][ind],
                testing_data[iz]["Arinyo_min"],
            )
            p1d_Mpc_sm[isim, iiz] = ysm
            p1d_Mpc_orig[isim, iiz] = testing_data[iz]["p1d_Mpc"][ind]

            in_emu = {}
            for par in emulator.emu_params:
                in_emu[par] = testing_data[iz][par]
            yemu = emulator.emulate_p1d_Mpc(
                in_emu,
                testing_data[iz]["p1d_Mpc"][ind],
                testing_data[iz]["z"],
            )
            p1d_Mpc_emu[isim, iiz] = yemu

    return zz, k_Mpc, p1d_Mpc_orig, p1d_Mpc_sm, p1d_Mpc_emu, mask
