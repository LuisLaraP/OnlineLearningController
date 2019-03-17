import time

from olc.neural_network import buildNetwork
from olc.noise import OrnsteinUhlenbeck
from olc.replay_buffer import ReplayBuffer


class Controller:

	def __init__(self, settings, network, environment, logger):
		self.settings = settings
		self.env = environment
		self.logger = logger
		self.q = buildNetwork('Q', network)
		self.random = OrnsteinUhlenbeck(
			self.env.action_space.low.shape,
			self.settings['timestep'],
			self.settings['noise']['theta'],
			self.settings['noise']['sigma']
		)
		self.replayBuffer = ReplayBuffer(self.settings['replay-buffer-size'])

	def run(self):
		"""
		Run an experiment on the environment.

		The simulation will run for exactly the amount of steps specified in the
		settings. If an episode ends before reaching the target number of steps, the
		environment is reset and the experiment continues.
		"""
		episode = 0
		step = 0
		while step < self.settings['steps']:
			episode += 1
			self.random.reset()
			lastState = None
			action = None
			reward = None
			state = self.env.reset()
			reset = False
			while not reset and step < self.settings['steps']:
				startTime = time.time()
				step += 1
				state, reward, reset, error = self.env.getState()
				if lastState is not None:
					self.replayBuffer.storeTransition(lastState, action, reward, state, reset)
				lastState = state
				action = self._randomPolicy()
				self.env.act(action)
				for i in range(len(action)):
					self.logger.logScalar('Action/Axis {}'.format(i + 1), action[i], step)
				self.logger.logScalar('Reward', reward, step)
				self.logger.logScalar('Error', error, step)
				activeTime = time.time() - startTime
				if activeTime > 0:
					time.sleep(self.settings['timestep'] - activeTime)
				totalTime = (time.time() - startTime) * 1000
				self.logger.logScalar('Active time', activeTime * 1000, step)
				self.logger.logScalar('Sampling time', totalTime, step)

	def _randomPolicy(self):
		return self.random.step()
