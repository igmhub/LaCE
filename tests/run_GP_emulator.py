import argparse
from lace.emulator.gp_emulator import GPEmulator

def main():
    # Create an ArgumentParser instance
    parser = argparse.ArgumentParser(description='Passing the emulator_label option')

    # Add an argument for 'nyx_version' and set its default value to "Cabayol23"
    parser.add_argument('--emulator_label', default="Pedersen23", help='GP emulator options: Pedersen21, Pedersen23. Default: Pedersen23')

    # Parse the command-line arguments
    args = parser.parse_args()

    gp_emu_P21 = GPEmulator(training_set='Pedersen21', emulator_label=args.emulator_label)
    
    print('It has successfully trained the GP emulator')

if __name__ == "__main__":
    main()
