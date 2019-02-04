"""Launcher of environment instances."""

_registry = {}


def make(name):
	"""
	Create a new instance of the given environment.

	The internal registry is searched first. If no package with the given name is
	found, the lookup continues in the `gym` package, if it is installed.

	Parameters
	----------
	name : str
		Name of the environment to be instantiated.

	Returns
	-------
	env
		Instance of the environment `name`.
	"""
	if name in _registry:
		return _registry[name]()
	try:
		import gym
		env = gym.make(name)
	except ImportError:
		print("Package 'gym' is not installed.")
		raise
	except gym.error.UnregisteredEnv:
		msg = 'No environment with name {} was found.'
		print(msg.format(name))
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
