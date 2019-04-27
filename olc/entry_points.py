import argparse
import datetime
import json

import olc.environments as envs
from olc.controller import Controller
from olc.logger import Logger
from olc.settings import getDefaults, merge


def olc():
	parser = argparse.ArgumentParser(
		description='Run an experiment.'
	)
	parser.add_argument(
		'settings',
		help='path to the settings file.'
	)
	args = parser.parse_args()

	# Read settings
	with open(args.settings, 'r') as settingsFile:
		settings = json.load(settingsFile)

	# Create environment
	environment = envs.make(settings['task'])

	# Create logger
	time = datetime.datetime.now().time()
	logDir = 'logs/{}-{:%H:%M}'.format(settings['task']['name'], time)
	logger = Logger(logDir)

	# Create controller
	defParams = getDefaults(__name__, 'params')
	mergedParams = merge(defParams, settings)
	controller = Controller(mergedParams, environment, logger)

	# Run
	controller.run()
	environment.close()
