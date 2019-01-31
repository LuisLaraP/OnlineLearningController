"""Definitions for all entry points."""

import argparse
import json

import olc.environments as envs


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
    try:
        environment = envs.make(specs['environment'])
    except:  # noqa: E722
        exit(1)
    reset = True
    for _ in range(specs['steps']):
        if reset:
            state = environment.reset()
        environment.render()
        action = environment.action_space.sample()
        state, reward, reset, info = environment.step(action)
    environment.close()
