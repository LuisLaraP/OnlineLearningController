import numpy as np
import tensorflow as tf

import olc.noise
from olc.neural_network import buildNetwork
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
		self.isTraining = tf.placeholder_with_default(False, None, 'is_training')
		actorInputs = {'state': self.state}
		criticInputs = {'action': self.action, 'state': self.state}
		self.actor = buildNetwork('actor', self.settings['actor'], actorInputs,
			scaleLow=self.env.action_space.low, scaleHigh=self.env.action_space.high)
		self.actorTarget = buildNetwork('actor_target', self.settings['actor'], actorInputs,
			scaleLow=self.env.action_space.low, scaleHigh=self.env.action_space.high)
		self.critic = buildNetwork('critic', self.settings['critic'], criticInputs)
		self.criticTarget = buildNetwork('critic_target', self.settings['critic'], criticInputs)
		noiseName = self.settings['noise']['name']
		noiseParams = self.settings['noise']
		noiseParams.pop('name', None)
		noiseParams['ndim'] = self.env.action_space.low.size
		self.noise = getattr(olc.noise, noiseName)(**noiseParams)
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
		self.actorTarget.setParameters(self.actor.getParameters())
		self.criticTarget.setParameters(self.critic.getParameters())
		episode = 0
		self.step = 0
		self.logger.checkpoint(self.step)
		while self.step < self.settings['steps']:
			episode += 1
			self.noise.reset()
			state = self.env.reset()
			reset = False
			while not reset and self.step < self.settings['steps']:
				self.step += 1
				self.env.render()
				action = self._learnedPolicy(state) + self._randomPolicy(state)
				newState, reward, reset, info = self.env.step(action)
				if reset:
					reward = 0
				self.replayBuffer.storeTransition(state, action, reward, newState, reset)
				state = newState
				actionValue = self.session.run(self.critic.output, {self.state: [state], self.action: [action]})
				self._train()
				self._updateTargetNetworks()
				for i in range(len(action)):
					self.logger.logScalar('Action/{}'.format(i + 1), action[i], self.step)
				self.logger.logScalar('Action value', actionValue, self.step)
				self.logger.logScalar('Reward', reward, self.step)
				if self.step % self.settings['save-interval'] == 0:
					self.logger.checkpoint(self.step)

	def _learnedPolicy(self, state):
		action, sums = self.session.run([self.actor.output, self.actor.summaries], {
			self.state: [state]
		})
		self.logger.writeSummary(sums, self.step)
		return action[0]

	def _randomPolicy(self, _):
		return self.noise.step() * (self.env.action_space.high - self.env.action_space.low) * self.settings['noise-scale']

	def _setupTraining(self):
		# Critic
		self.labels = tf.placeholder(tf.float32,
			(None, self.critic.output.shape[-1]), 'labels')
		self.loss = tf.losses.mean_squared_error(self.labels, self.critic.output)
		regularizer = tf.contrib.layers.l2_regularizer(self.settings['critic-lambda'])
		self.loss += tf.contrib.layers.apply_regularization(regularizer, self.critic.parameters)
		optName = self.settings['critic-optimizer']['name'] + 'Optimizer'
		optSettings = self.settings['critic-optimizer']
		optSettings.pop('name', None)
		criticOptimizer = getattr(tf.train, optName)(**optSettings)
		self.trainCritic = criticOptimizer.minimize(self.loss, name='train_critic')
		self.criticGrad = tf.gradients(self.critic.output, self.action,
			name='critic_gradients')
		# Actor
		self.actorGrad = tf.gradients(self.actor.output, self.actor.parameters,
			-self.criticGrad[0],
			name='actor_gradients'
		)
		self.actorGrad = [x / self.settings['batch-size'] for x in self.actorGrad]
		optName = self.settings['actor-optimizer']['name'] + 'Optimizer'
		optSettings = self.settings['actor-optimizer']
		optSettings.pop('name', None)
		actorOptimizer = getattr(tf.train, optName)(**optSettings)
		self.trainActor = actorOptimizer.apply_gradients(zip(self.actorGrad, self.actor.parameters),
			name='train_actor')

	def _train(self):
		s0, a, r, sf, term = self.replayBuffer.sample(self.settings['batch-size'])
		nt = np.logical_not(term)
		labels = np.reshape(r, (len(r), 1))
		loss = 0
		if len(s0) > 0:
			actions = self.session.run(self.actorTarget.output, {
				self.state: sf[nt]
			})
			returns = self.session.run(self.criticTarget.output, {
				self.state: sf[nt],
				self.action: actions
			})
			labels[nt] += self.settings['gamma'] * returns
			actions = self.session.run(self.actor.output, {
				self.state: s0
			})
			ret = self.session.run([self.trainActor, self.trainCritic, self.loss], {
				self.state: s0,
				self.action: actions,
				self.labels: labels,
				self.isTraining: True
			})
			loss = ret[-1]
		self.logger.logScalar('Loss', loss, self.step)

	def _updateTargetNetworks(self):
		actorParams = self.actor.getParameters()
		criticParams = self.critic.getParameters()
		self.actorTarget.setParameters(actorParams, self.settings['tau'])
		self.criticTarget.setParameters(criticParams, self.settings['tau'])
