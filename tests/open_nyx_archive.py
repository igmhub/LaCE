import argparse
from lace.archive.nyx_archive import NyxArchive

def main():
    # Create an ArgumentParser instance
    parser = argparse.ArgumentParser(description='Passing the nyx file version')

    # Add an argument for 'nyx_version' and set its default value to "Cabayol23"
    parser.add_argument('--nyx_version', default="Oct2023", help='Nyx file versions: Oct2023. Default: Oct2023')

    # Parse the command-line arguments
    args = parser.parse_args()

    nyx_archive = NyxArchive(nyx_version=args.nyx_version, verbose=True)
    
    print('It has successfully opened the Nyx archive')

if __name__ == "__main__":
    main()
