import argparse
import yaml
import torch
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import logging
from typing import Dict, List, Tuple, Union
from tqdm import tqdm
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# our modules
from lace.archive import (gadget_archive, 
                          nyx_archive)
from lace.emulator.nn_emulator import NNEmulator
from lace.emulator.gp_emulator import GPEmulator
from lace.emulator.emulator_manager import set_emulator    
from lace.utils import poly_p1d
from lace.emulator.constants import PROJ_ROOT

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train emulators from config')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to config YAML file')
    return parser.parse_args()

def load_config(config_path: Path) -> Dict:
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def create_archive(archive_config: Dict) -> Union[nyx_archive.NyxArchive, gadget_archive.GadgetArchive]:
    logger.info(f"Creating archive: {archive_config['file']}")
    if archive_config["file"] == "Nyx":
        return nyx_archive.NyxArchive(nyx_version=archive_config["version"])
    elif archive_config["file"] == "Gadget":
        return gadget_archive.GadgetArchive(postproc=archive_config["version"])
    else:
        raise ValueError(f"Archive {archive_config['file']} not supported")
    
def predict_p1d(emulator: Union[NNEmulator, GPEmulator], test_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    zs = [d['z'] for d in test_data if d['z'] < 4.8]
    Nz = len(zs)
    Nk = len(test_data[0]['p1d_Mpc'][(test_data[0]['k_Mpc'] > 0) & (test_data[0]['k_Mpc'] < 4)])
    p1ds_err = np.zeros((Nz, Nk))
    preds = {}

    logger.info("Predicting P1D for each redshift")
    for m, z in enumerate(tqdm(zs)):
        kMpc_test = test_data[m]['k_Mpc']
        p1d_true = test_data[m]['p1d_Mpc']
        mask = (kMpc_test > 0) & (kMpc_test < 4)
        kMpc_test = kMpc_test[mask]
        p1d_true = p1d_true[mask]
        
        fit_p1d = poly_p1d.PolyP1D(kMpc_test, p1d_true, kmin_Mpc=1e-3, kmax_Mpc=4, deg=5)
        p1d_true = fit_p1d.P_Mpc(kMpc_test)   

        try:
            p1d_emu = emulator.emulate_p1d_Mpc(test_data[m], kMpc_test)
            p1ds_err[m, :] = (p1d_emu / p1d_true - 1) * 100
            preds[f"{z}"] = p1d_emu
        except Exception as e:
            logger.warning(f"Emulation failed for z={z}: {str(e)}. Skipping this redshift.")
            p1ds_err[m, :] = np.nan

    return p1ds_err, zs, kMpc_test, preds

def make_p1d_err_plot(p1ds_err: np.ndarray, kMpc_test: np.ndarray, zs_test: np.ndarray, config: dict):
    """
    Plot the P1D errors, either averaged over redshifts or individually for each redshift.

    Parameters:
    p1ds_err (np.array): Array of P1D errors
    kMpc_test (np.array): k values in Mpc^-1
    config (dict): Configuration dictionary
    """
    logger.info("Generating P1D error plot")
    fig, ax = plt.subplots(figsize=(10, 6))

    if config.get("average_over_z", False):
        # Average over redshifts
        p1d_mean = np.nanmean(p1ds_err, axis=0)
        ax.plot(kMpc_test, p1d_mean, color='crimson', label='Mean error')
    else:
        # Plot each redshift prediction separately
        num_redshifts = p1ds_err.shape[0]
        colors = plt.cm.rainbow(np.linspace(0, 1, num_redshifts))
        for i, color in enumerate(colors):
            ax.plot(kMpc_test, p1ds_err[i], color=color, label=f'z={np.round(zs_test[i], 2)}')

    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('k (Mpc$^{-1}$)')
    ax.set_ylabel('Relative Error in P1D (%)')
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.set_ylim(-5, 5)

    plt.tight_layout()
    if config["save_plot_path"] is not None:
        plt.savefig(config["save_plot_path"], bbox_inches='tight')
    plt.close()
    logger.info(f"P1D error plot saved to {config['save_plot_path']}")


def main():
    args = parse_args()
    config = load_config(args.config)

    emu_params = config["emulator_params"]
    logger.info(f"Emulator parameters: {emu_params}")
    
    archive = create_archive(config["archive"])

    logger.info("Setting up emulator")
    emulator = set_emulator(
        emulator_label=config["emulator_label"],
        archive=archive,
        drop_sim=config["drop_sim"])
    
    logger.info("Getting testing data")
    test_data = archive.get_testing_data(sim_label=config["sim_test"])

    logger.info("Predicting P1D")
    p1ds_err, zs, kMpc_test, predicted_p1d = predict_p1d(emulator, test_data)
    make_p1d_err_plot(p1ds_err, kMpc_test, zs, config)

    if config["save_predictions_path"] is not None:
        json.dump(predicted_p1d, open(config["save_predictions_path"], "w"))

    logger.info("Main function completed")

if __name__ == "__main__":
    main()