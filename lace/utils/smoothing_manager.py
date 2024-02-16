from lace.utils import poly_p1d


def apply_smoothing(emulator, data):
    """Apply smoothing to the p1d"""

    lenk = len(data)
    if lenk == 0:
        data = [data]

    if (data[0]["p1d_Mpc"] is None) | (data[0]["k_Mpc"] is None):
        raise ValueError("p1d_Mpc or k_Mpc not defined")

    if emulator.emu_type == "k_bin":
        # No smoothing
        pass
    elif emulator.emu_type == "polyfit":
        # Polynomial smoothing
        for ii in range(len(data)):
            fit_p1d = poly_p1d.PolyP1D(
                data[ii]["k_Mpc"],
                data[ii]["p1d_Mpc"],
                kmin_Mpc=1e-10,
                kmax_Mpc=emulator.kmax_Mpc,
                deg=emulator.ndeg,
            )
            data[ii]["p1d_Mpc_smooth"] = fit_p1d.P_Mpc(data[ii]["k_Mpc"])
    elif emulator.emu_type == "k_bin_sm":
        # Nonlinear smoothing
        for ii in range(len(data)):
            data[ii]["p1d_Mpc_smooth"] = emulator.apply_kernel_smoothing(
                data[ii]["k_Mpc"],
                data[ii]["p1d_Mpc"],
            )[0]
    else:
        raise ValueError("Smoothing technique not recognized:", technique)

    if lenk == 0:
        data = data[0]
