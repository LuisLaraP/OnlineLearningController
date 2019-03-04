import argparse
import datetime
import json
import os.path

import olc.environments as envs
from olc.controller import Controller
from olc.logger import Logger


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
	time = datetime.datetime.now().time()
	logFile = os.path.splitext(os.path.basename(args.filename))[0]
	logFile = 'logs/{}-{:%H:%M}.log'.format(logFile, time)
	logger = Logger(logFile)
	controller = Controller(environment, logger)
	controller.run(specs['steps'])
	environment.close()
