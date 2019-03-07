_registry = {}


def make(settings, simulation):
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
	if settings['name'] in _registry:
		return _registry[settings['name']](settings, simulation)
	try:
		import gym
		env = gym.make(settings['name'])
	except ImportError:
		exit("Package 'gym' is not installed.")
	except gym.error.UnregisteredEnv:
		msg = 'No environment with name {} was found.'
		exit(msg.format(settings['name']))
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
