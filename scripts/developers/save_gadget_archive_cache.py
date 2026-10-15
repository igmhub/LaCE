"""Save a Gadget archive to a single file so it loads much faster.

``GadgetArchive`` reads this file by default (and warns if it is missing),
instead of parsing thousands of per-snapshot JSON files. Rebuild it whenever
the post-processed simulation files change, e.g. when corrected test-simulation
files are installed.

Example: ``python save_gadget_archive_cache.py --postproc Cabayol23_fixp3d``
"""

import argparse
import time

from lace.archive.gadget_archive import GadgetArchive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--postproc",
        nargs="+",
        default=["Cabayol23_fixp3d"],
        help="post-processing(s) to cache (default: Cabayol23_fixp3d)",
    )
    parser.add_argument("--data-path", default=None, help="override simulation-data root")
    parser.add_argument("--output", default=None, help="output file (single --postproc only)")
    args = parser.parse_args()
    if args.output is not None and len(args.postproc) != 1:
        parser.error("--output requires exactly one --postproc")

    for postproc in args.postproc:
        start = time.time()
        archive = GadgetArchive(
            postproc=postproc, data_path=args.data_path, use_cache=False
        )
        fname = archive.save_cache(args.output)
        print(f"{postproc}: saved {len(archive.data)} entries to {fname} "
              f"({time.time() - start:.0f} s)")


if __name__ == "__main__":
    main()
