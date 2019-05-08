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
		stateMin = np.concatenate((settings['robot']['workspace-min'], np.radians(settings['robot']['joint-min']), -np.radians(settings['robot']['max-velocities'])))
		stateMax = np.concatenate((settings['robot']['workspace-max'], np.radians(settings['robot']['joint-max']), np.radians(settings['robot']['max-velocities'])))
		self.action_space = Box(-np.array(settings['robot']['max-torques']), np.array(settings['robot']['max-torques']))
		self.observation_space = Box(stateMin, stateMax)
		self.sim.readDistance(settings['error-object-name'])
		self.rewardVelFactor = 1 / np.linalg.norm(np.radians(settings['robot']['max-velocities']))

	def close(self):
		self.sim.close()

	def reset(self):
		self.sim.stop()
		self.state = self.observation_space.sample()
		ref = self.state[:len(self.settings['robot']['workspace-min'])]
		self.sim.setDummyPosition(self.settings['target-object-name'], ref)
		pose = self.state[len(self.settings['robot']['workspace-min']):-len(self.settings['robot']['max-velocities'])]
		self.sim.setPose(pose)
		self.sim.setVelocities(np.zeros(self.action_space.low.size))
		self.sim.start()
		self.sim.step()
		self.state[len(self.settings['robot']['workspace-min']):] = np.concatenate(self.sim.getRobotState())
		return self.state

	def render(self):
		pass

	def step(self, action):
		self.sim.setTorques(action)
		self.sim.step()
		self.state = np.concatenate((np.zeros(3),) + self.sim.getRobotState())
		error = self.sim.readDistance(self.settings['error-object-name'])
		reward = -error - np.linalg.norm(self.state[-self.action_space.low.size:]) * self.rewardVelFactor
		return self.state, reward, False, None
