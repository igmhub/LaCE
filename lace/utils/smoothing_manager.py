import numpy as np
from scipy.optimize import curve_fit

from lace.utils import poly_p1d


def apply_smoothing(emulator, data, fprint=print):
    """Apply smoothing to the p1d"""

    type_data = type(data)
    if type_data is not list:
        data = [data]

    if (data[0]["p1d_Mpc"] is None) | (data[0]["k_Mpc"] is None):
        raise ValueError("p1d_Mpc or k_Mpc not defined")

    if emulator.emu_type == "k_bin":
        # No smoothing
        fprint("k-bin emulator, no smoothing is applied")
        pass
    elif emulator.emu_type == "polyfit":
        fprint("polyfit emulator, smoothing is applied")
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
        fprint("k_bin_sm emulator, smoothing is applied")
        # Nonlinear smoothing
        for ii in range(len(data)):
            _ = emulator.Kernel_Smoothing.apply_kernel_smoothing(
                data[ii]["k_Mpc"], data[ii]
            )
            data[ii]["p1d_Mpc_smooth"] = _
    elif emulator.emu_type == "gpolyfit":
        for ii in range(len(data)):
            ind_k = (data[ii]["k_Mpc"] > 0) & (
                data[ii]["k_Mpc"] < emulator.kmax_Mpc
            )
            k_Mpc = data[ii]["k_Mpc"][ind_k]
            k_fit = k_Mpc / emulator.kmax_Mpc

            norm = np.interp(
                k_Mpc,
                emulator.input_norm["k_Mpc"],
                emulator.norm_imF(data[ii]["mF"]),
            )

            y2fit = np.log(data[ii]["p1d_Mpc"][ind_k] / norm)
            par_fit, _ = curve_fit(emulator.func_poly, k_fit, y2fit)

            log_unorm_p1d = emulator.func_poly(
                data[ii]["k_Mpc"] / emulator.kmax_Mpc, *par_fit
            )

            norm = np.interp(
                data[ii]["k_Mpc"],
                emulator.input_norm["k_Mpc"],
                emulator.norm_imF(data[ii]["mF"]),
            )

            data[ii]["p1d_Mpc_smooth"] = np.exp(log_unorm_p1d) * norm
    else:
        raise ValueError(
            "Smoothing technique not recognized:", emulator.emu_type
        )

    if type_data is not list:
        data = data[0]
