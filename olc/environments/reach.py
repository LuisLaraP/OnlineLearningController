"""
Reach task.

In this task, a robot manipulator must position its end effector within a
certain distance of a predefined location.

The control signal is a vector with the velocity of each of the joints of the
robot, and the reward is based on the current distance to the target position.
"""

from olc.settings import getDefaults, merge
from .spaces import Box
from .simulation import Simulation


class Reach:
	"""
	Reach task.

	Parameters
	----------
	settings : dict
		Custom settings.

	Attributes
	----------
	action_space
		Space containing all the possible actions for this environment.
	"""

	def __init__(self, settings):
		self.settings = getDefaults(__name__ + ':reach_defaults.json')
		self.settings = merge(self.settings, settings)
		self.sim = Simulation(self.settings['simulation'], self.settings['robot'])
		self.action_space = Box([0], [1])

	def close(self):
		"""Close connection to simulator."""
		self.sim.close()

	def render(self):
		"""
		Do nothing.

		Method kept for compatibility with OpenAI Gym's environment API.
		"""
		pass

	def reset(self):
		"""Reset simulation."""
		self.sim.stop()
		self.sim.start()

	def step(self, action):
		"""
		Take an action and advance the simulation.

		Parameters
		----------
		action : Numpy array
			Action to execute in this timestep.

		Returns
		-------
		state : array-like
			State of the environment after taking the given action.
		reward : float
			Reward obtained for taking the given action.
		reset : bool
			True if the episode ended in this timestep. The user must call the `reset`
			method after this happens.
		info
			This return value is always None.
		"""
		return None, 0, False, None
