import argparse
from lace.archive import gadget_archive

def main():
    # Create an ArgumentParser instance
    parser = argparse.ArgumentParser(description='Passing different archive options')

    # Add an argument for 'postproc' and set its default value to "Cabayol23"
    parser.add_argument('--postproc', default="Cabayol23", help='Post-processing options: Pedersen21 and Cabayol23. Default: Cabayol23')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Now you can access the 'postproc' argument using args.postproc
    mpg_arch = gadget_archive.GadgetArchive(postproc=args.postproc)
    
    print('It has successfully opened the Gadget archive')

if __name__ == "__main__":
    main()
