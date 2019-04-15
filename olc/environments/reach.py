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
			self.sim.robot['workspace-min']
			+ [radians(x) for x in self.sim.robot['joint-min']]
			+ [0] * len(self.sim.robot['max-velocities']),
			self.sim.robot['workspace-max']
			+ [radians(x) for x in self.sim.robot['joint-max']]
			+ [radians(x) for x in self.sim.robot['max-velocities']]
		)
		self.sim.registerDistanceObject(self.settings['error-object-name'])
		self.sim.registerDummyObject(self.settings['target-object-name'])
		self.lastError = None

	def act(self, action):
		self.sim.setJointVelocities(self.action_space.scale(action))

	def close(self):
		self.sim.close()

	def reset(self):
		newRef = np.random.uniform(self.sim.robot['workspace-min'],
			self.sim.robot['workspace-max'])
		newPose = np.radians(np.random.uniform(self.sim.robot['joint-min'],
			self.sim.robot['joint-max']))
		self.sim.stop()
		self.sim.setJointPositions(newPose)
		self.sim.setDummyPosition(self.settings['target-object-name'], newRef)
		self.sim.start()
		time.sleep(0.2)
		self.reference = newRef
		self.lastError = self.sim.readDistance(self.settings['error-object-name'])

	def getState(self):
		info = {}
		pos = self.sim.getJointPositions()
		vel = self.sim.getJointVelocities()
		error = self.sim.readDistance(self.settings['error-object-name'])
		info['error'] = error
		dError = error - self.lastError
		self.lastError = error
		info['error_diff'] = dError
		state = np.concatenate((self.reference, pos, vel))
		state = np.divide(state - self.observation_space.low,
			self.observation_space.high - self.observation_space.low)
		if error <= self.settings['threshold-success']:
			reward = self.settings['reward-success']
			reset = True
		elif error >= self.settings['threshold-failure']:
			reward = self.settings['reward-failure']
			reset = True
		else:
			reward = self._computeReward(error, dError)
			reset = False
		return state, reward, reset, info

	def _computeReward(self, e, de):
		propTerm = self.settings['kp'] * (self.settings['threshold-success'] - e)
		diffTerm = -self.settings['kd'] * de
		return propTerm + diffTerm
