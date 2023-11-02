import argparse
from lace.emulator.nn_emulator import NNEmulator

def main():
    # Create an ArgumentParser instance
    parser = argparse.ArgumentParser(description='Passing the emulator_label option')

    # Add an argument for 'nyx_version' and set its default value to "Cabayol23"
    parser.add_argument('--emulator_label', default="Cabayol23", help='NN emulator options: Cabayol23, Cabayol23_extended. Default: Cabayol23')

    # Parse the command-line arguments
    args = parser.parse_args()

    nn_emu_C23 = NNEmulator(training_set='Cabayol23', emulator_label=args.emulator_label)
    
    print('It has successfully trained the NN emulator')

if __name__ == "__main__":
    main()
