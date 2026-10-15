"""Save a Nyx archive to a single file so it loads much faster.

``NyxArchive`` reads this file by default (and warns if it is missing),
instead of parsing the large HDF5 file. Rebuild it whenever the HDF5 file
changes. The cache is written next to the HDF5 file.

Example: ``python save_nyx_archive_cache.py``
"""

import argparse
import time

from lace.archive.nyx_archive import NyxArchive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nyx-version",
        default="Sept2025_include_Nyx_fid_rseed",
        help="Nyx release, i.e. models_Nyx_<version>.hdf5 (default: %(default)s)",
    )
    parser.add_argument("--nyx-path", default=None, help="override Nyx data directory")
    parser.add_argument("--nyx-file", default=None, help="explicit HDF5 file")
    parser.add_argument("--output", default=None, help="output cache file")
    args = parser.parse_args()

    start = time.time()
    archive = NyxArchive(
        nyx_version=args.nyx_version,
        nyx_file=args.nyx_file,
        nyx_path=args.nyx_path,
        use_cache=False,
    )
    fname = archive.save_cache(args.output, nyx_file=args.nyx_file)
    print(f"saved {len(archive.data)} entries to {fname} ({time.time() - start:.0f} s)")


if __name__ == "__main__":
    main()
