"""
Reach task.

In this task, a robot manipulator must position its end effector within a
certain distance of a predefined location.

The control signal is a vector with the velocity of each of the joints of the
robot, and the reward is based on the current distance to the target position.
"""

import numpy as np
from gym.spaces import Box


class Reach:

	def __init__(self, settings, simulation):
		self.settings = settings
		self.sim = simulation
		stateMin = np.concatenate((settings['robot']['workspace-min'], settings['robot']['joint-min']))
		stateMax = np.concatenate((settings['robot']['workspace-max'], settings['robot']['joint-max']))
		self.action_space = Box(-np.array(settings['robot']['max-velocities']), np.array(settings['robot']['max-velocities']))
		self.observation_space = Box(stateMin, stateMax)

	def close(self):
		self.sim.close()

	def reset(self):
		self.state = np.zeros(self.observation_space.low.size)
		return self.state

	def render(self):
		pass

	def step(self, action):
		return self.state, 0, True, None
