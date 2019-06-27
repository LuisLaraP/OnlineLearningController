import sys
import time

import numpy as np
import tensorflow as tf

from olc.neural_network import Actor, Critic
from olc.noise import OrnsteinUhlenbeck
from olc.replay_buffer import ReplayBuffer


class Controller:

	def __init__(self, settings, environment, logger, checkpoint):
		self.settings = settings
		self.env = environment
		self.logger = logger
		self.actionDim = self.env.action_space.low.size
		self.stateDim = self.env.observation_space.low.size
		self._setupModel()
		self._setupMetrics()
		self.logger.logGraph()
		if checkpoint is not None:
			self.checkpoint = tf.train.latest_checkpoint(checkpoint)
		else:
			self.checkpoint = None

	def run(self):
		self.session = tf.Session()
		self.session.run(tf.global_variables_initializer())
		# Initialize actor target parameters
		actorParams = self.session.run(self.actor.parameters)
		for f, t in zip(actorParams, self.actorTarget.parameters):
			t.load(f, self.session)
		# Initialize critic target parameters
		criticParams = self.session.run(self.critic.parameters)
		for f, t in zip(criticParams, self.criticTarget.parameters):
			t.load(f, self.session)
		# Create noise process
		self.noise = OrnsteinUhlenbeck(self.actionDim,
			self.settings['noise']['dt'],
			self.settings['noise']['theta'],
			self.settings['noise']['sigma']
		)
		# Load checkpoint if provided
		if self.checkpoint is not None:
			self.logger.loadCheckpoint(self.session, self.checkpoint)
			self.buffer.restore(self.session)
		# Training
		epoch = 0
		step = 0
		done = True
		confidence = 0
		self.buffer.save(self.session)
		self.logger.checkpoint(self.session, 0)
		self.test(step)
		while step < self.settings['steps']:
			startTime = time.time()
			epoch += 1
			for _ in range(self.settings['nb-rollouts']):
				if done:
					state = self.env.reset()
					self.noise.reset()
					done = False
				action = 0.5 * (1. + confidence) * self._learnedPolicy(state) + 0.5 * (1. - confidence) * self._randomPolicy(state)
				newState, reward, done, info = self.env.step(action)
				if self.settings['controller-type'] == 'continuous':
					done = False
				self.buffer.storeTransition(state, action, reward, newState, done)
				state = newState
				step, actionValue = self.session.run([self.incrementStep, self.critic.output],
					{self.action: [action], self.state: [state], self.isTraining: False})
				_, confidence, metricSums = self.session.run([self.updateMetrics, self.confidence, self.metrics],
					{self.actionValue: actionValue.item(), self.reward: reward})
				[self.logger.logScalar('Action/' + str(i), x, step) for i, x in enumerate(action)]
				if isinstance(info, dict) and 'error' in info:
					self.logger.logScalar('Error', info['error'], step)
				self.logger.writeSummary(metricSums, step)
				if self.settings['render']:
					self.env.render()
			loss = 0
			self.buffer.setCapacity(self.settings['replay-buffer-min'] + confidence * (self.settings['replay-buffer-max'] - self.settings['replay-buffer-min']))
			for _ in range(self.settings['nb-train']):
				loss += self._train()
				self.session.run([self.actorTarget.update, self.criticTarget.update])
			loss /= self.settings['nb-train']
			self.logger.logScalar('Critic loss', loss, step)
			if step % self.settings['save-interval'] == 0:
				self.buffer.save(self.session)
				self.logger.checkpoint(self.session, step)
				self.test(step)
			elapsed = time.time() - startTime
			print("Epoch {}:\tSteps: {}\tTime: {:.3}s".format(epoch, step, elapsed))

	def test(self, step):
		cumReward = 0
		for episode in range(5):
			done = False
			state = self.env.reset()
			while not done:
				action = self._learnedPolicy(state)
				state, reward, done, _ = self.env.step(action)
				cumReward += reward
		self.logger.logScalar('Learning curve', cumReward / 5, step)

	def _learnedPolicy(self, state):
		action = self.session.run(self.actor.output, {
			self.state: [state],
			self.isTraining: False
		})
		return action[0]

	def _randomPolicy(self, _):
		return self.noise.step() * (self.env.action_space.high - self.env.action_space.low)

	def _setupMetrics(self):
		decay = self.settings['metric-decay']
		confidenceStep = self.settings['confidence-step']
		self.updateMetrics = []
		with tf.variable_scope('metrics'):
			ema = tf.train.ExponentialMovingAverage(decay=decay)
			# Action value mean
			self.actionValue = tf.placeholder(tf.float32, shape=(), name='action_value')
			self.updateMetrics.append(ema.apply([self.actionValue]))
			self.meanValue = ema.average(self.actionValue)
			tf.summary.scalar('Action value', self.actionValue, collections=['metrics'])
			tf.summary.scalar('Action value average', self.meanValue, collections=['metrics'])
			# Reward mean
			self.reward = tf.placeholder(tf.float32, shape=(), name='reward')
			self.updateMetrics.append(ema.apply([self.reward]))
			self.meanReward = ema.average(self.reward)
			tf.summary.scalar('Reward', self.reward, collections=['metrics'])
			tf.summary.scalar('Reward average', self.meanReward, collections=['metrics'])
			# Reward cusum
			rewardCusumPos = tf.get_variable('reward_cusum_pos', shape=(), dtype=tf.float32, initializer=tf.initializers.zeros)
			rewardCusumNeg = tf.get_variable('reward_cusum_neg', shape=(), dtype=tf.float32, initializer=tf.initializers.zeros)
			self.updateMetrics.append(tf.assign(rewardCusumPos, decay * tf.maximum(0., rewardCusumPos + self.reward - self.meanReward)))
			self.updateMetrics.append(tf.assign(rewardCusumNeg, decay * tf.minimum(0., rewardCusumNeg + self.reward - self.meanReward)))
			self.rewardCusum = rewardCusumPos - rewardCusumNeg
			self.updateMetrics.append(self.rewardCusum)
			tf.summary.scalar('Reward cusum', self.rewardCusum, collections=['metrics'])
			# Confidence
			rewardConfidence = tf.get_variable('reward_confidence', shape=(), dtype=tf.float32, initializer=tf.initializers.zeros)
			self.updateMetrics.append(tf.assign(rewardConfidence, tf.clip_by_value(rewardConfidence + confidenceStep * tf.sign(self.settings["cusum-threshold"] - tf.abs(self.rewardCusum)), 0., 1.)))
			self.confidence = rewardConfidence
			tf.summary.scalar('Confidence', self.confidence, collections=['metrics'])
			# Summary merging
			self.metrics = tf.summary.merge_all('metrics')

	def _setupModel(self):
		self.action = tf.placeholder(tf.float32, (None, self.actionDim), name='action')
		self.state = tf.placeholder(tf.float32, (None, self.stateDim), name='state')
		self.qLabels = tf.placeholder(tf.float32, (None, 1), name='q_labels')
		self.isTraining = tf.placeholder_with_default(True, None, 'is_training')
		self.actor = Actor('actor', self.settings['actor'], self.state, self.isTraining, self.env.action_space.high, self.env.action_space.low)
		self.critic = Critic('critic', self.settings['critic'], self.action, self.state, self.isTraining)
		self.actorTarget = Actor('actor_target', self.settings['actor'], self.state, self.isTraining, self.env.action_space.high, self.env.action_space.low)
		self.criticTarget = Critic('critic_target', self.settings['critic'], self.actorTarget.output, self.state, self.isTraining)
		self.critic.createTrainOps(self.action, self.qLabels)
		self.actor.createTrainOps(self.critic.actionGrads, self.settings['batch-size'])
		self.actorTarget.createUpdateOps(self.settings['tau'], self.actor.parameters)
		self.criticTarget.createUpdateOps(self.settings['tau'], self.critic.parameters)
		self.incrementStep = tf.assign_add(tf.train.get_or_create_global_step(), 1)
		self.buffer = ReplayBuffer(self.settings['replay-buffer-max'], self.actionDim, self.stateDim)

	def _train(self):
		siBatch, aBatch, rBatch, sfBatch, tBatch = self.buffer.sample(self.settings['batch-size'])
		loss = 0
		if len(siBatch) > 0:
			# Critic
			qValues = self.session.run(self.criticTarget.output, {
				self.state: sfBatch
			})
			labels = self.settings['gamma'] * qValues + np.reshape(rBatch, (rBatch.size, 1))
			labels[tBatch] = 0
			_, loss, actions = self.session.run([self.critic.train, self.critic.loss, self.actor.output], {
				self.action: aBatch,
				self.state: siBatch,
				self.qLabels: labels
			})
			# Actor
			self.session.run(self.actor.train, {
				self.action: actions,
				self.state: siBatch
			})
		return loss
