"""Launcher of environment instances."""


def make(name):
    """
    Create a new instance of the given environment.

    If the given environment is not found in the `gym` package, an error is
    raised.

    Parameters
    ----------
    name : str
        Name of the environment to be instantiated.

    Returns
    -------
    env
        Instance of the environment `name`.
    """
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
