import argparse
import yaml
import torch
from pathlib import Path
# our modules
from lace.archive import (gadget_archive, 
                          nyx_archive)
from lace.emulator.nn_emulator import NNEmulator
from lace.emulator.gp_emulator import GPEmulator
from lace.emulator.constants import PROJ_ROOT

def parse_args():
    parser = argparse.ArgumentParser(description='Train emulators from config')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to config YAML file')
    return parser.parse_args()

def load_config(config_path: Path) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    

def main():
    args = parse_args()
    config = load_config(args.config)

    emu_params = config["emulator_params"]

    def create_archive(archive_config):
        if archive_config["file"] == "Nyx":
            return nyx_archive.NyxArchive(nyx_version=archive_config["version"])
        elif archive_config["file"] == "Gadget":
            return gadget_archive.GadgetArchive(postproc=archive_config["version"])
        else:
            raise ValueError(f"Archive {archive_config['file']} not supported")

    def create_nn_emulator_with_label(config):
        if config.get("training_set"):
            return NNEmulator(training_set=config["training_set"], 
                              emulator_label=config["emulator_label"],
                              save_path=PROJ_ROOT / config["save_path"])
        elif config.get("archive"):
            archive = create_archive(config["archive"])
            return NNEmulator(archive=archive,
                              emulator_label=config["emulator_label"],
                              save_path=PROJ_ROOT / config["save_path"])
        else:
            raise ValueError("Either training_set or archive must be provided")

    def create_gp_emulator_with_label(config):
        if config.get("training_set"):
            return GPEmulator(training_set=config["training_set"],
                              emulator_label=config["emulator_label"])
        elif config.get("archive"):
            archive = create_archive(config["archive"])
            return GPEmulator(archive=archive,
                              emulator_label=config["emulator_label"],
                              )
        else:
            raise ValueError("Either training_set or archive must be provided")

    def create_nn_emulator_without_label(config):
        hyperparameters = config["hyperparameters"]
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
        hyperparameters = config["hyperparameters"]
        if config.get("archive"):
            archive = create_archive(config["archive"])
            return GPEmulator(archive=archive)
        elif config.get("training_set"):
            return GPEmulator(training_set=config["training_set"])
        else:
            raise ValueError("Either archive or training_set must be provided")

    if config["emulator_type"] == "NN":
        if config.get("emulator_label"):
            emulator = create_nn_emulator_with_label(config)
        else:
            emulator = create_nn_emulator_without_label(config)
    elif config["emulator_type"] == "GP":
        if config.get("emulator_label"):
            emulator = create_gp_emulator_with_label(config)
        else:
            emulator = create_gp_emulator_without_label(config)
    else:
        raise ValueError("emulator_type must be either 'NN' or 'GP'")


if __name__ == "__main__":
    main()