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
			+ [radians(x) for x in self.sim.robot['joint-min']],
			self.sim.robot['workspace-max']
			+ [radians(x) for x in self.sim.robot['joint-max']]
		)
		self.sim.registerDistanceObject(self.settings['error-object-name'])
		self.sim.registerDummyObject(self.settings['target-object-name'])
		self.lastError = None

	def close(self):
		self.sim.close()

	def reset(self):
		self.lastError = self.settings['threshold-failure'] + 1
		while self.lastError > self.settings['threshold-failure']:
			newRef = np.random.uniform(self.sim.robot['workspace-min'],
			self.sim.robot['workspace-max'])
			newPose = np.radians(np.random.uniform(self.sim.robot['joint-min'],
			self.sim.robot['joint-max'])) / 2
			self.sim.stop()
			self.sim.setJointVelocities(np.zeros(self.action_space.low.shape))
			self.sim.setJointPositions(newPose)
			self.sim.setDummyPosition(self.settings['target-object-name'], newRef)
			self.sim.start()
			time.sleep(0.2)
			self.reference = newRef
			self.lastError = self.sim.readDistance(self.settings['error-object-name'])
		self.action = np.zeros(self.action_space.low.shape)
		self.state = np.concatenate((self.reference, newPose))
		self.lastPos = self.state[-self.action_space.low.size:]
		self.blockCount = 0
		return self.state

	def render(self):
		pass

	def step(self, action):
		self.action = np.clip(action, -1, 1)
		self.sim.setJointVelocities(self.action_space.scale(action))
		time.sleep(self.settings['timestep'])
		info = {'lastResult': 0}
		pos = self.sim.getJointPositions()
		error = self.sim.readDistance(self.settings['error-object-name'])
		info['error'] = error
		dError = error - self.lastError
		self.lastError = error
		info['error_diff'] = dError
		self.state = np.concatenate((self.reference, pos))
		self.state = np.divide(self.state - self.observation_space.low,
			self.observation_space.high - self.observation_space.low)
		stall = self._detectBlock()
		reward = self._computeReward(error, dError)
		if error <= self.settings['threshold-success']:
			reset = True
			info['lastResult'] = 1
		elif error >= self.settings['threshold-failure'] or stall:
			reset = True
			info['lastResult'] = -1
		else:
			reset = False
			info['lastResult'] = 0
		return self.state, reward, reset, info

	def _computeReward(self, e, de):
		return np.sign(de) - np.linalg.norm(self.action)

	def _detectBlock(self):
		curPos = self.state[-self.action_space.low.size:]
		if np.count_nonzero(np.isclose(curPos, self.lastPos, atol=1e-3)) >= 3:
			self.blockCount += 1
			if self.blockCount == 10:
				self.blockCount = 0
				self.lastPos = curPos
				return True
		else:
			self.blockCount = 0
			self.lastPos = curPos
			return False
