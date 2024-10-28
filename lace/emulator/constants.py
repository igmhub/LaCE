from strenum import StrEnum
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]

class TrainingSet(StrEnum):
    PEDERSEN21 = "Pedersen21"
    CABAYOL23 = "Cabayol23"
    NYX23_OCT2023 = "Nyx23_Oct2023"
    NYX23_JUL2024 = "Nyx23_Jul2024"

class EmulatorLabel(StrEnum):
    CABAYOL23 = "Cabayol23"
    CABAYOL23_PLUS = "Cabayol23+"
    NYX_V0 = "Nyx_v0"
    NYX_ALPHAP = "Nyx_alphap"
    NYX_ALPHAP_COV = "Nyx_alphap_cov"
    NYX_ALPHAP_EXTENDED = "Nyx_alphap_extended"
    CABAYOL23_EXTENDED = "Cabayol23_extended"
    NYX_V0_EXTENDED = "Nyx_v0_extended"
    CABAYOL23_PLUS_EXTENDED = "Cabayol23+_extended"

GADGET_LABELS = {EmulatorLabel.CABAYOL23, EmulatorLabel.CABAYOL23_EXTENDED}
NYX_LABELS = {EmulatorLabel.NYX_V0, EmulatorLabel.NYX_V0_EXTENDED}

EMULATOR_PARAMS = {# Model parameters for different emulators
    "Cabayol23" : {
        "emu_params" : [
                "Delta2_p",
                "n_p",
                "mF",
                "sigT_Mpc",
                "gamma",
                "kF_Mpc"],
        "emu_type" : "polyfit",
        "kmax_Mpc" : 4,
        "ndeg" : 5,
        "nepochs" : 100,
        "step_size" : 75,
        "nhidden" : 5,
    "max_neurons" : 100,
    "lr0" : 1e-3,
    "weight_decay" : 1e-4,
    "batch_size" : 100,
    "amsgrad" : False,
    },
    "Cabayol23+" : {
        "emu_params" : [
                "Delta2_p",
                "n_p",
                "mF",
                "sigT_Mpc",
                "gamma",
                "kF_Mpc"],
    "emu_type" : "polyfit",
    "kmax_Mpc" : 4,
    "ndeg" : 5,
    "nepochs" : 510,
    "step_size" : 500,
    "nhidden" : 5,
    "max_neurons" : 250,
    "lr0" : 7e-4,
    "weight_decay" : 9.6e-3,
    "batch_size" : 100,
    "amsgrad" : True,
    },
    "Nyx_v0" : {
        "emu_params" : [
                "Delta2_p",
                "n_p",
                "mF",
                "sigT_Mpc",
                "gamma",
                "kF_Mpc"],
    "emu_type" : "polyfit",
    "kmax_Mpc" : 4,
    "ndeg" : 6,
    "nepochs" : 800,
    "step_size" : 700,
    "nhidden" : 5,
    "max_neurons" : 150,
    "lr0" : 5e-5,
    "weight_decay" : 1e-4,
    "batch_size" : 100,
    "amsgrad" : True,
    },
    "Nyx_alphap" : {
        "emu_params" : [
                "Delta2_p",
                "n_p",
                "alpha_p",
                "mF",
                "sigT_Mpc",
                "gamma",
                "kF_Mpc"],
        "emu_type" : "polyfit",
        "kmax_Mpc" : 4,
        "ndeg" : 6,
        "nepochs" : 600,
        "step_size" : 500,
        "nhidden" : 6,
        "max_neurons" : 400,
        "lr0" : 2.5e-4,
        "weight_decay" : 8e-3,
        "batch_size" : 100,
        "amsgrad" : True,
},
    "Nyx_alphap_extended" : {
    "emu_params" : [
                "Delta2_p",
                "n_p",
                "alpha_p",
                "mF",
                "sigT_Mpc",
                "gamma",
                "kF_Mpc"],
        "emu_type" : "polyfit",
        "kmax_Mpc" : 8,
        "ndeg" : 8,
        "nepochs" : 1000,
        "step_size" : 1000,
        "nhidden" : 6,
        "max_neurons" : 400,
        "lr0" : 2.5e-4,
        "weight_decay" : 8e-3,
        "batch_size" : 100,
        "amsgrad" : True,
},
    "Cabayol23_extended" : {
    "emu_params" : [
                "Delta2_p",
                "n_p",
                "mF",
                "sigT_Mpc",
                "gamma",
                "kF_Mpc"],
        "kmax_Mpc" : 8,
        "ndeg" : 7,
        "nepochs" : 1000,
        "step_size" : 1000,
        "nhidden" : 6,
        "max_neurons" : 400,
        "lr0" : 2.5e-4,
        "weight_decay" : 8e-3,
        "batch_size" : 100,
        "amsgrad" : True,
        "weighted_emulator" : False
},
    "Cabayol23+_extended" : {
        "emu_params" : [
                "Delta2_p",
                "n_p",
                "mF",
                "sigT_Mpc",
                "gamma",
                "kF_Mpc"],
        "emu_type" : "polyfit",
        "kmax_Mpc" : 8,
        "ndeg" : 7,
        "nepochs" : 250,
        "step_size" : 200,
        "nhidden" : 4,  
        "max_neurons" : 250,
        "lr0" : 7.1e-4,
        "weight_decay" : 4.1e-3,
        "batch_size" : 100,
        "amsgrad" : True,
        "weighted_emulator" : True,
},
"Nyx_v0_extended" : {
    "emu_params" : [
                "Delta2_p",
                "n_p",
                "mF",
                "sigT_Mpc",
                "gamma",
                "kF_Mpc"],
        "emu_type" : "polyfit",
        "kmax_Mpc" : 8,
        "ndeg" : 7,
        "nepochs" : 1000,
        "step_size" : 750,
        "nhidden" : 5,
        "weighted_emulator" : True,
},
    "Nyx_alphap_cov" : {
        "emu_params" : [
                "Delta2_p",
                "n_p",
                "alpha_p",
                "mF",
                "sigT_Mpc",
                "gamma",
                "kF_Mpc"],
        "emu_type" : "polyfit",
        "kmax_Mpc" : 4,
        "ndeg" : 6,
        "nepochs" : 1000,
        "step_size" : 800,
        "nhidden" : 6,
        "max_neurons" : 400,
        "lr0" : 2.5e-4,
        "weight_decay" : 8e-3,
        "batch_size" : 100,
        "amsgrad" : True,
        "z_max" : 4.6,
}
}

EMULATOR_DESCRIPTIONS = {
    EmulatorLabel.CABAYOL23_PLUS: (
        r"Neural network emulating the optimal P1D of Gadget simulations "
        "fitting coefficients to a 5th degree polynomial. It "
        "goes to scales of 4Mpc^{-1} and z<=4.5. The parameters "
        "passed to the emulator will be overwritten to match "
        "these ones. This option is an updated on wrt to the one in the Cabayol+23 paper."
    ),
    EmulatorLabel.CABAYOL23_PLUS_EXTENDED: (
        r"Neural network emulating the optimal P1D of Gadget simulations "
        "fitting coefficients to a 5th degree polynomial. It "
        "goes to scales of 4Mpc^{-1} and z<=4.5. The parameters "
        "passed to the emulator will be overwritten to match "
        "these ones. This option is an updated on wrt to the one in the Cabayol+23 paper."
    ),
    EmulatorLabel.NYX_V0_EXTENDED: (
        r"Neural network emulating the optimal P1D of Nyx simulations "
        "fitting coefficients to a 6th degree polynomial. It "
        "goes to scales of 8Mpc^{-1} and z<=4.5."
    ),
    EmulatorLabel.NYX_ALPHAP_EXTENDED: (
        r"Neural network emulating the optimal P1D of Nyx simulations "
        "fitting coefficients to a 6th degree polynomial. It "
        "goes to scales of 8Mpc^{-1} and z<=4.5."
    ),
    EmulatorLabel.CABAYOL23_EXTENDED: (
        r"Neural network emulating the optimal P1D of Gadget simulations "
        "fitting coefficients to a 7th degree polynomial. It "
        "goes to scales of 8Mpc^{-1} and z<=4.5."
    ),
    EmulatorLabel.NYX_V0: (
        r"Neural network emulating the optimal P1D of Nyx simulations "
        "fitting coefficients to a 6th degree polynomial. It "
        "goes to scales of 4Mpc^{-1} and z<=4.5."
    ),
    EmulatorLabel.NYX_ALPHAP: (
        r"Neural network emulating the optimal P1D of Nyx simulations "
        "fitting coefficients to a 6th degree polynomial. It "
        "goes to scales of 4Mpc^{-1} and z<=4.5."
    ),
    EmulatorLabel.NYX_ALPHAP_COV: (
        r"Neural network under testing for the Nyx_alphap emulator."
    ),
    EmulatorLabel.CABAYOL23: (
        r"Neural network emulating the optimal P1D of Gadget simulations "
        "fitting coefficients to a 5th degree polynomial. It "
        "goes to scales of 4Mpc^{-1} and z<=4.5."
    )
}



