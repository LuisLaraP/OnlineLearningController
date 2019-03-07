import argparse
import datetime
import json
import os.path

import olc.environments as envs
from olc.controller import Controller
from olc.logger import Logger
from olc.simulation import Simulation


def olc():
	parser = argparse.ArgumentParser(
		description='Run an experiment.'
	)
	parser.add_argument(
		'filename',
		help='path to the specifications file to be used.'
	)
	parser.add_argument(
		'robot',
		nargs='?',
		help='path to the specifications file for the robot.'
	)
	args = parser.parse_args()

	# Read settings
	with open(args.filename, 'r') as specsFile:
		specs = json.load(specsFile)
	if args.robot is not None:
		with open(args.robot, 'r') as robotFile:
			robot = json.load(robotFile)
	else:
		robot = None

	# Connect to simulator
	if robot is not None:
		simulator = Simulation(robot)
	else:
		simulator = None

	# Create environment
	environment = envs.make(specs['environment'], simulator)

	# Create logger
	time = datetime.datetime.now().time()
	logFile = os.path.splitext(os.path.basename(args.filename))[0]
	logFile = 'logs/{}-{:%H:%M}.log'.format(logFile, time)
	logger = Logger(logFile, visualize=['reward'])

	# Create controller
	controller = Controller(environment, logger)

	# Run
	input('Press ENTER to start.\n')
	controller.run(specs['steps'])
	environment.close()
