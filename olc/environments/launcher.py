"""Launcher of environment instances."""

_registry = {}


def make(specs):
	"""
	Create a new instance of the given environment.

	The internal registry is searched first. If no package with the given name is
	found, the lookup continues in the `gym` package, if it is installed.

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
	if specs['name'] in _registry:
		return _registry[specs['name']](specs)
	try:
		import gym
		env = gym.make(specs['name'])
	except ImportError:
		print("Package 'gym' is not installed.")
		raise
	except gym.error.UnregisteredEnv:
		msg = 'No environment with name {} was found.'
		print(msg.format(specs['name']))
		raise
	return env


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
