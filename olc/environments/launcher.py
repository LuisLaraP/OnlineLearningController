from olc.settings import getDefaults, merge

_registry = {}


def make(settings, simulation):
	"""
	Create a new instance of the given environment.

	Parameters
	----------
	specs : dict
		Settings to pass to the environment. Must contain the key `name`, and its
		value determines the environment to load.

	Returns
	-------
	env
		Instance of the environment `name`.
	"""
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
