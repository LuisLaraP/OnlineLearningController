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
		'robot',
		help='path to the specifications file for the robot.'
	)
	args = parser.parse_args()

	# Read settings
	with open(args.params, 'r') as paramsFile:
		params = json.load(paramsFile)
	with open(args.task, 'r') as taskFile:
		task = json.load(taskFile)
	with open(args.robot, 'r') as robotFile:
		robot = json.load(robotFile)

	# Create environment
	simulator = Simulation(robot)
	environment = envs.make(task, simulator)

	# Create logger
	time = datetime.datetime.now().time()
	logDir = os.path.splitext(os.path.basename(args.task))[0]
	logDir = 'logs/{}-{:%H:%M}'.format(logDir, time)
	logger = Logger(logDir)

	# Create controller
	defParams = getDefaults(__name__, 'params')
	mergedParams = merge(defParams, params)
	controller = Controller(mergedParams, environment, logger)

	# Run
	controller.run()
	environment.close()
