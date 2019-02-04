"""
Reach task.

In this task, a robot manipulator must position its end effector within a
certain distance of a predefined location.

The control signal is a vector with the velocity of each of the joints of the
robot, and the reward is based on the current distance to the target position.
"""


class Reach:
	"""Reach task."""

	def __init__(self):
		pass

	def close(self):
		"""Close connection to simulator."""
		pass

	def render(self):
		"""
		Do nothing.

		Method kept for compatibility with OpenAI Gym's environment API.
		"""
		pass

	def reset(self):
		"""Reset simulation."""
		pass

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
		"""
		pass
