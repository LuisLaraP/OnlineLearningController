import time

import numpy as np
import tensorflow as tf

from olc.neural_network import buildNetwork
from olc.noise import OrnsteinUhlenbeck
from olc.replay_buffer import ReplayBuffer


class Controller:

	def __init__(self, settings, environment, logger):
		self.settings = settings
		self.env = environment
		self.logger = logger
		nIn = self.env.observation_space.low.size
		self.actor = buildNetwork('actor', self.settings['actor'], nIn)
		self.critic = buildNetwork('critic', self.settings['critic'], nIn)
		self.random = OrnsteinUhlenbeck(
			self.env.action_space.low.shape,
			self.settings['timestep'],
			self.settings['noise']['theta'],
			self.settings['noise']['sigma']
		)
		self.replayBuffer = ReplayBuffer(self.settings['replay-buffer-size'])
		self._setupTraining()
		self.logger.logGraph()

	def run(self):
		"""
		Run an experiment on the environment.

		The simulation will run for exactly the amount of steps specified in the
		settings. If an episode ends before reaching the target number of steps, the
		environment is reset and the experiment continues.
		"""
		self.session = tf.Session().__enter__()
		self.session.run(tf.global_variables_initializer())
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
				state, reward, reset, info = self.env.getState()
				if lastState is not None:
					self.replayBuffer.storeTransition(lastState, action, reward, state, reset)
				lastState = state
				action = self._learnedPolicy(state)
				self.env.act(action)
				loss = self._trainCritic()
				self.logger.logScalar('Loss', loss, step)
				for i in range(len(action)):
					self.logger.logScalar('Action/Axis {}'.format(i + 1), action[i], step)
				self.logger.logScalar('Reward', reward, step)
				self.logger.logScalar('Error', info['error'], step)
				self.logger.logScalar('Error rate', info['error_diff'] / self.settings['timestep'], step)
				activeTime = time.time() - startTime
				if activeTime > 0:
					time.sleep(self.settings['timestep'] - activeTime)
				totalTime = (time.time() - startTime) * 1000
				self.logger.logScalar('Active time', activeTime * 1000, step)
				self.logger.logScalar('Sampling time', totalTime, step)

	def _learnedPolicy(self, state):
		return self.session.run(self.actor.output, {
			self.actor.input: [state]
		})[0]

	def _randomPolicy(self, _):
		return self.random.step()

	def _setupTraining(self):
		self.labels = tf.placeholder(tf.float32,
			(None, self.critic.output.shape[-1]), 'labels')
		self.loss = tf.losses.mean_squared_error(self.labels, self.critic.output)
		optName = self.settings['optimizer']['name'] + 'Optimizer'
		optSettings = self.settings['optimizer']
		optSettings.pop('name', None)
		optimizer = getattr(tf.train, optName)(**optSettings)
		self.train = optimizer.minimize(self.loss)

	def _trainCritic(self):
		s0, a, r, sf, _ = self.replayBuffer.sample(self.settings['batch-size'])
		loss = 0
		if len(s0) > 0:
			returns = self.session.run(self.critic.output, {
				self.critic.input: sf
			})
			labels = np.reshape(r, (len(r), 1)) + self.settings['gamma'] * returns
			_, loss = self.session.run([self.train, self.loss], {
				self.critic.input: s0,
				self.labels: labels
			})
		return loss
