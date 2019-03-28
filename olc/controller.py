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
		self.action = tf.placeholder(
			tf.float32,
			(None, self.env.action_space.low.size),
			name='action'
		)
		self.state = tf.placeholder(
			tf.float32,
			(None, self.env.observation_space.low.size),
			name='state'
		)
		actorInputs = {'state': self.state}
		criticInputs = {'action': self.action, 'state': self.state}
		self.actor = buildNetwork('actor', self.settings['actor'], actorInputs)
		self.critic = buildNetwork('critic', self.settings['critic'], criticInputs)
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
				self.env.action_space.clip(action)
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
			self.state: [state]
		})[0]

	def _randomPolicy(self, _):
		return self.random.step()

	def _setupTraining(self):
		# Critic
		self.labels = tf.placeholder(tf.float32,
			(None, self.critic.output.shape[-1]), 'labels')
		self.loss = tf.losses.mean_squared_error(self.labels, self.critic.output)
		optName = self.settings['critic-optimizer']['name'] + 'Optimizer'
		optSettings = self.settings['critic-optimizer']
		optSettings.pop('name', None)
		criticOptimizer = getattr(tf.train, optName)(**optSettings)
		self.trainCritic = criticOptimizer.minimize(self.loss)
		self.criticGrad = tf.gradients(self.critic.output, self.action,
			name='critic_gradients'
		)
		# Actor
		self.actorGrad = tf.gradients(self.actor.output, self.actor.parameters,
			-self.criticGrad[0],
			name='actor_gradients'
		)
		optName = self.settings['actor-optimizer']['name'] + 'Optimizer'
		optSettings = self.settings['critic-optimizer']
		optSettings.pop('name', None)
		actorOptimizer = getattr(tf.train, optName)(**optSettings)
		self.trainActor = actorOptimizer.apply_gradients(zip(self.actorGrad, self.actor.parameters))

	def _trainCritic(self):
		s0, a, r, sf, _ = self.replayBuffer.sample(self.settings['batch-size'])
		loss = 0
		if len(s0) > 0:
			returns = self.session.run(self.critic.output, {
				self.state: sf,
				self.action: a
			})
			labels = np.reshape(r, (len(r), 1)) + self.settings['gamma'] * returns
			_, loss = self.session.run([self.trainCritic, self.loss], {
				self.state: s0,
				self.action: np.zeros((len(sf), 6)),
				self.labels: labels
			})
		return loss
