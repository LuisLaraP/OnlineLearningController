import argparse
import json

import olc.environments as envs
from .controller import Controller


def olc():
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
	environment = envs.make(specs['environment'])
	controller = Controller(environment)
	controller.run(specs['steps'])
	environment.close()
