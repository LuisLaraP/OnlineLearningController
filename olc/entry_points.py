"""Definitions for all entry points."""

import argparse
import json


def olc():
    """Run an experiment using a specification file."""
    parser = argparse.ArgumentParser(
        description='Run an experiment using a specification file.'
    )
    parser.add_argument(
        'filename',
        help='path to the specifications file to be used.'
    )
    args = parser.parse_args()
    with open(args.filename, 'r') as specsFile:
        specs = json.load(specsFile)
    print(specs)
