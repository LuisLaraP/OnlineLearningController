import argparse
import datetime
import json
import os.path

import olc.environments as envs
from olc.controller import Controller
from olc.logger import Logger
from olc.settings import getDefaults, merge
from olc.simulation import Simulation


def olc():
	parser = argparse.ArgumentParser(
		description='Run an experiment.'
	)
	parser.add_argument(
		'params',
		help='path to the parameters file.'
	)
	parser.add_argument(
		'task',
		help='path to the specifications file for the task.'
	)
	parser.add_argument(
		'network',
		help='path to the specifications file for the neural network.'
	)
	parser.add_argument(
		'robot',
		help='path to the specifications file for the robot.'
	)
	args = parser.parse_args()

	# Read settings
	with open(args.params, 'r') as paramsFile:
		params = json.load(paramsFile)
	with open(args.task, 'r') as taskFile:
		task = json.load(taskFile)
	with open(args.network, 'r') as networkFile:
		network = json.load(networkFile)
	with open(args.robot, 'r') as robotFile:
		robot = json.load(robotFile)

	# Connect to simulator
	if robot is not None:
		simulator = Simulation(robot)
	else:
		simulator = None

	# Create environment
	environment = envs.make(task, simulator)

	# Create logger
	time = datetime.datetime.now().time()
	logFile = os.path.splitext(os.path.basename(args.task))[0]
	logFile = 'logs/{}-{:%H:%M}.log'.format(logFile, time)
	logger = Logger(logFile, visualize=['reward'])

	# Create controller
	defParams = getDefaults(__name__, 'params')
	mergedParams = merge(defParams, params)
	controller = Controller(mergedParams, network, environment, logger)

	# Run
	input('Press ENTER to start.\n')
	controller.run()
	environment.close()
