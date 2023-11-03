import argparse
from lace.emulator.nn_emulator import NNEmulator
from lace.archive import nyx_archive, gadget_archive
import torch

def main():
    # Create an ArgumentParser instance
    parser = argparse.ArgumentParser(description='Passing the emulator_label option')

    # Add an argument for 'nyx_version' and set its default value to "Cabayol23"

    parser.add_argument('--training_set', default=None, help="Options:'Pedersen21', 'Cabayol23', 'Nyx23'")
    parser.add_argument('--emulator_label', default=None, help='NN emulator options: Cabayol23, Cabayol23_extended. Default: Cabayol23')
    parser.add_argument('--drop_sim', default=None, help="Option to drop simulation from the training set. Options 'mpg_i' or 'nyx_i0'")
    parser.add_argument('--drop_z', default=None, help="Option to drop redshift from the training set. Provide redshift as float.")
    parser.add_argument('--save_path', default=None, help="Path where the model must be saved.")


    # Parse the command-line arguments
    args = parser.parse_args()
    
    if args.save_path is None:
        raise ValueError("'save_path' is a required arguemnt to run this script.")
        

    print(args.drop_z)
    nn_emu = NNEmulator(training_set=args.training_set,
                        emulator_label=args.emulator_label,
                        drop_sim=args.drop_sim,
                        drop_z=args.drop_z)
    
    print('It has successfully trained the NN emulator')
        
    
    
    model = nn_emu.nn.state_dict()
    
    metadata = {
        'training_set':args.training_set,
        'emulator_label': args.emulator_label,
        'drop_sim': args.drop_sim,
        'drop_z': args.drop_z
    }

    model.metadata = metadata
    
    model_data = {
        'metadata': metadata,
        'emulator': model
    }
    
    torch.save(model_data, args.save_path)

if __name__ == "__main__":
    main()
