"""
Reach task.

In this task, a robot manipulator must position its end effector within a
certain distance of a predefined location.

The control signal is a vector with the velocity of each of the joints of the
robot, and the reward is based on the current distance to the target position.
"""

import time
from math import radians

import numpy as np

from .spaces import Box


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

	def __init__(self, settings, simulation):
		self.settings = settings
		self.sim = simulation
		self.action_space = Box(
			[-radians(x) for x in self.sim.robot['max-velocities']],
			[radians(x) for x in self.sim.robot['max-velocities']]
		)
		self.observation_space = Box(
			[radians(x) for x in self.sim.robot['joint-min']]
			+ [0] * len(self.sim.robot['max-velocities']),
			[radians(x) for x in self.sim.robot['joint-max']]
			+ [radians(x) for x in self.sim.robot['max-velocities']]
		)
		self.sim.registerDistanceObject(self.settings['error-object-name'])
		self.lastError = None

	def close(self):
		self.sim.close()

	def render(self):
		pass

	def reset(self):
		self.sim.stop()
		self.sim.start()
		time.sleep(0.2)
		self.lastError = self.sim.readDistance(self.settings['error-object-name'])

	def step(self, action):
		"""
		Take an action and advance the simulation.

		Parameters
		----------
		action : array-like
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
		self.sim.setJointVelocities(action)
		time.sleep(self.settings['timestep'])
		pos = self.sim.getJointPositions()
		vel = self.sim.getJointVelocities()
		error = self.sim.readDistance(self.settings['error-object-name'])
		reward = self.lastError - error
		self.lastError = error
		state = np.concatenate((pos, vel))
		if error <= self.settings['threshold-distance']:
			reset = True
		else:
			reset = False
		return state, reward, reset, None
