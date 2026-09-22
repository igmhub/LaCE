"""Command-line helpers for configuring local LaCE paths."""

from __future__ import annotations

import argparse

from lace.configuration.paths import set_nyx_path


def set_nyx_path_command() -> None:
    """Save the Nyx archive location supplied on the command line."""

    parser = argparse.ArgumentParser(
        description="Save the local Nyx archive directory for LaCE."
    )
    parser.add_argument("nyx_path", help="Directory containing the Nyx archive files.")
    args = parser.parse_args()
    path = set_nyx_path(args.nyx_path)
    print(f"LaCE Nyx path saved: {path}")


if __name__ == "__main__":
    set_nyx_path_command()
