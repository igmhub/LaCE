import numpy as np
from scipy.optimize import curve_fit
from lace.emulator.gp_emulator_multi import GPEmulator


def data_for_l10(archive, emulator_label, suite="nyx", type_ana="cen"):
    """
    Compute the covariance matrix of the emulator

    type_ana:   cen for l10 tests evaluated at the central simulation,
                other value for classical tests
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
            p1d_Mpc_sm[isim, iiz] = norm * np.exp(
                emulator.func_poly(k_fit, *popt)
            )

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
                p1d_Mpc_emu[isim, iiz] = emulator.emulate_p1d_Mpc(
                    int_data, k_Mpc
                )
            else:
                mask[isim, iiz] = False

    return zz, k_Mpc, p1d_Mpc_sm, p1d_Mpc_emu, mask
