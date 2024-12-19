#! /usr/bin/env python3

import argparse
import yaml
import torch
import logging
from pathlib import Path
# our modules
from lace.archive import (gadget_archive, 
                          nyx_archive)
from lace.emulator.nn_emulator import NNEmulator
from lace.emulator.gp_emulator import GPEmulator
from lace.emulator.constants import PROJ_ROOT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description='Train emulators from config')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to config YAML file')
    return parser.parse_args()

def load_config(config_path: Path) -> dict:
    logging.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    

def main():
    args = parse_args()
    config = load_config(args.config)

    save_path = PROJ_ROOT / config["save_path"]
    if not save_path.parent.exists():
        logging.warning(f"Creating directory {save_path.parent} as it does not exist")
        save_path.parent.mkdir(parents=True)

    emu_params = config["emulator_params"]
    logging.info(f"Emulator parameters: {emu_params}")

    def create_archive(archive_config):
        logging.info(f"Creating archive with config: {archive_config}")
        if archive_config["file"] == "Nyx":
            return nyx_archive.NyxArchive(nyx_version=archive_config["version"])
        elif archive_config["file"] == "Gadget":
            return gadget_archive.GadgetArchive(postproc=archive_config["version"])
        else:
            raise ValueError(f"Archive {archive_config['file']} not supported")

    def create_nn_emulator_with_label(config):
        logging.info("Creating NN emulator with label")
        if config.get("training_set"):
            return NNEmulator(training_set=config["training_set"], 
                              emulator_label=config["emulator_label"],
                              drop_sim=config.get("drop_sim"),
                              save_path=PROJ_ROOT / config["save_path"])
        elif config.get("archive"):
            archive = create_archive(config["archive"])
            return NNEmulator(archive=archive,
                              emulator_label=config["emulator_label"],
                              drop_sim=config.get("drop_sim"),
                              save_path=PROJ_ROOT / config["save_path"])
        else:
            raise ValueError("Either training_set or archive must be provided")

    def create_gp_emulator_with_label(config):
        logging.info("Creating GP emulator with label")
        if config.get("training_set"):
            return GPEmulator(training_set=config["training_set"],
                              emulator_label=config["emulator_label"],
                              drop_sim=config.get("drop_sim"))
        elif config.get("archive"):
            archive = create_archive(config["archive"])
            return GPEmulator(archive=archive,
                              emulator_label=config["emulator_label"],
                              drop_sim=config.get("drop_sim"))
        else:
            raise ValueError("Either training_set or archive must be provided")

    def create_nn_emulator_without_label(config):
        logging.info("Creating NN emulator without label")
        hyperparameters = {
            key: (int(value) if key in ['ndeg', 'nepochs', 'step_size', 'nhidden', 'max_neurons', 'seed', 'batch_size']
                  else float(value) if key in ['kmax_Mpc', 'lr0', 'weight_decay', 'z_max']
                  else bool(value) if key == 'weighted_emulator'
                  else value)
            for key, value in config["hyperparameters"].items()
        }
        if config.get("archive"):
            archive = create_archive(config["archive"])
            return NNEmulator(archive=archive,
                              save_path=PROJ_ROOT / config["save_path"],
                              **hyperparameters)
        elif config.get("training_set"):
            return NNEmulator(training_set=config["training_set"], 
                              save_path=PROJ_ROOT / config["save_path"],
                              **hyperparameters)
        else:
            raise ValueError("Either archive or training_set must be provided")

    def create_gp_emulator_without_label(config):
        logging.info("Creating GP emulator without label")
        hyperparameters = {
            key: float(value) if key in ['kmax_Mpc', 'lr0', 'weight_decay', 'z_max']
                  else value
            for key, value in config["hyperparameters"].items()
        }
        if config.get("archive"):
            archive = create_archive(config["archive"])
            return GPEmulator(archive=archive,
                              drop_sim=config.get("drop_sim"))
        elif config.get("training_set"):
            return GPEmulator(training_set=config["training_set"],
                              drop_sim=config.get("drop_sim"))
        else:
            raise ValueError("Either archive or training_set must be provided")

    if config["emulator_type"] == "NN":
        logging.info("Creating NN emulator")
        if config.get("emulator_label"):
            emulator = create_nn_emulator_with_label(config)
        else:
            emulator = create_nn_emulator_without_label(config)
    elif config["emulator_type"] == "GP":
        logging.info("Creating GP emulator")
        if config.get("emulator_label"):
            emulator = create_gp_emulator_with_label(config)
        else:
            emulator = create_gp_emulator_without_label(config)
    else:
        raise ValueError("emulator_type must be either 'NN' or 'GP'")

    logging.info("Emulator created successfully")

if __name__ == "__main__":
    main()