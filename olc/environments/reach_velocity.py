import time

import numpy as np
from gym.spaces import Box


class ReachVelocity:

	def __init__(self, settings, simulation):
		self.settings = settings
		self.sim = simulation
		stateMin = np.concatenate((settings['robot']['workspace-min'], np.radians(settings['robot']['joint-min'])))
		stateMax = np.concatenate((settings['robot']['workspace-max'], np.radians(settings['robot']['joint-max'])))
		self.action_space = Box(-np.array(settings['robot']['max-velocities']), np.array(settings['robot']['max-velocities']))
		self.observation_space = Box(stateMin, stateMax)
		self.sim.readDistance(settings['error-object-name'])
		self.rewardVelFactor = 1 / np.linalg.norm(np.radians(settings['robot']['max-velocities']))

	def close(self):
		self.sim.close()

	def reset(self):
		self.sim.stop()
		state = self.observation_space.sample()
		self.reference = state[:len(self.settings['robot']['workspace-min'])]
		self.sim.setDummyPosition(self.settings['target-object-name'], self.reference)
		self.pose = state[len(self.settings['robot']['workspace-min']):]
		self.sim.setPose(self.pose)
		self.sim.setVelocities(np.zeros(self.action_space.low.size))
		self.sim.start()
		self.sim.step()
		self.pose = self.sim.getRobotState()[0]
		self.curStep = 0
		time.sleep(0.05)
		self.potential = self._computePotential()
		return np.concatenate((self.reference, self.pose))

	def render(self):
		pass

	def step(self, action):
		self.curStep += 1
		self.sim.setVelocities(action)
		self.sim.step()
		self.pose = self.sim.getRobotState()[0]
		reward = self._computeReward()
		reset = self.curStep >= self.settings['max-steps']
		return np.concatenate((self.reference, self.pose)), reward, reset, None

	def _computePotential(self):
		return -10 * self.sim.readDistance(self.settings['error-object-name'])

	def _computeReward(self):
		newPotential = self._computePotential()
		dPotential = newPotential - self.potential
		self.potential = newPotential
		return dPotential
