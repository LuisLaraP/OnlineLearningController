from olc.environments.simulation import Simulation
from olc.settings import getDefaults, merge

_registry = {}


def make(settings):
	simulation = Simulation(settings['robot'])
	defs = getDefaults(__name__, settings['name'].lower())
	mergedSettings = merge(defs, settings)
	return _registry[settings['name']](mergedSettings, simulation)


def register(name, object):
	"""
	Add a new environment to the internal registry.

	Parameters
	----------
	name : str
		Unique identifier for the new environment.
	object : class
		Class to instantiate for this environment.
	"""
	assert name not in _registry
	_registry[name] = object
