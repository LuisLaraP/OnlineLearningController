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

	def act(self, action):
		self.action_space.scale(action)
		self.sim.setJointVelocities(action)

	def close(self):
		self.sim.close()

	def reset(self):
		self.sim.stop()
		self.sim.start()
		time.sleep(0.2)
		self.lastError = self.sim.readDistance(self.settings['error-object-name'])

	def getState(self):
		info = {}
		pos = self.sim.getJointPositions()
		vel = self.sim.getJointVelocities()
		error = self.sim.readDistance(self.settings['error-object-name'])
		info['error'] = error
		dError = error - self.lastError
		info['error_diff'] = dError
		self.lastError = error
		state = np.concatenate((pos, vel))
		reward = self._computeReward(error, dError)
		if error <= self.settings['threshold-distance']:
			reset = True
		else:
			reset = False
		return state, reward, reset, info

	def _computeReward(self, e, de):
		propTerm = self.settings['kp'] * (self.settings['threshold-distance'] - e)
		diffTerm = -self.settings['kd'] * de
		return propTerm + diffTerm
