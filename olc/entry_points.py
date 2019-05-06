import argparse
import datetime
import json

import olc.environments as envs
from olc.controller import Controller, Tester
from olc.logger import Logger
from olc.settings import getDefaults, merge


def olc_test():
	parser = argparse.ArgumentParser(
		description='Run an experiment.'
	)
	parser.add_argument(
		'settings',
		help='path to the settings file.'
	)
	parser.add_argument(
		'checkpoint_dir',
		help='path to the checkpoint directory.'
	)
	args = parser.parse_args()

	# Read settings
	with open(args.settings, 'r') as settingsFile:
		settings = json.load(settingsFile)

	# Create environment
	environment = envs.make(settings['task'])

	# Create controller
	defParams = getDefaults(__name__, 'params')
	mergedParams = merge(defParams, settings)
	controller = Tester(mergedParams, environment, args.checkpoint_dir)

	# Run
	controller.run()
	environment.close()


def olc_train():
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
	experimentName = '{}-{:%H:%M}'.format(settings['task']['name'], time)
	logger = Logger(experimentName)

	# Create controller
	defParams = getDefaults(__name__, 'params')
	mergedParams = merge(defParams, settings)
	controller = Controller(mergedParams, environment, logger)

	# Run
	controller.run()
	environment.close()
