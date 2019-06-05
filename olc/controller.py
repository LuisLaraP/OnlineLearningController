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
		# Create replay buffer
		self.buffer = ReplayBuffer(self.settings['replay-buffer-size'], self.actionDim, self.stateDim)
		# Create noise process
		self.noise = OrnsteinUhlenbeck(self.actionDim,
			self.settings['noise']['dt'],
			self.settings['noise']['theta'],
			self.settings['noise']['sigma']
		)
		# Load checkpoint if provided
		if self.checkpoint is not None:
			self.logger.loadCheckpoint(self.session, self.checkpoint)
		# Training
		epoch = 0
		step = 0
		done = True
		confidence = 0
		self.logger.checkpoint(self.session, 0)
		while step < self.settings['steps']:
			startTime = time.time()
			epoch += 1
			for _ in range(self.settings['nb-rollouts']):
				if done:
					state = self.env.reset()
					self.noise.reset()
					done = False
				action = (0.5 + confidence) * self._learnedPolicy(state) + (0.5 - confidence) * self._randomPolicy(state)
				newState, reward, done, _ = self.env.step(action)
				if self.settings['controller-type'] == 'continuous':
					done = False
				self.buffer.storeTransition(state, action, reward, newState, done)
				state = newState
				step, actionValue = self.session.run([self.incrementStep, self.critic.output],
					{self.action: [action], self.state: [state], self.isTraining: False})
				_, confidence, metricSums = self.session.run([self.updateMetrics, self.confidence, self.metrics],
					{self.actionValue: actionValue.item(), self.reward: reward})
				[self.logger.logScalar('Action/' + str(i), x, step) for i, x in enumerate(action)]
				self.logger.writeSummary(metricSums, step)
				if self.settings['render']:
					self.env.render()
			loss = 0
			for _ in range(self.settings['nb-train']):
				loss += self._train()
				self.session.run([self.actorTarget.update, self.criticTarget.update])
			loss /= self.settings['nb-train']
			self.logger.logScalar('Critic loss', loss, step)
			if step % self.settings['save-interval'] == 0:
				self.logger.checkpoint(self.session, step)
			elapsed = time.time() - startTime
			print("Epoch {}:\tSteps: {}\tTime: {:.3}s".format(epoch, step, elapsed))

	def _learnedPolicy(self, state):
		action = self.session.run(self.actor.output, {
			self.state: [state],
			self.isTraining: False
		})
		return action[0]

	def _randomPolicy(self, _):
		return self.noise.step() * (self.env.action_space.high - self.env.action_space.low)

	def _setupMetrics(self):
		decay = 0.9999
		confidenceStep = 5e-6
		self.updateMetrics = []
		with tf.variable_scope('metrics'):
			ema = tf.train.ExponentialMovingAverage(decay=decay)
			# Action value mean
			self.actionValue = tf.placeholder(tf.float32, shape=(), name='action_value')
			self.updateMetrics.append(ema.apply([self.actionValue]))
			self.meanValue = ema.average(self.actionValue)
			tf.summary.scalar('Action value', self.meanValue, collections=['metrics'])
			# Action value variance
			self.valueVariance = tf.get_variable('value_variance', shape=(), dtype=tf.float32, initializer=tf.initializers.zeros)
			incr = (1 - decay) * tf.square(self.actionValue - self.meanValue)
			self.updateMetrics.append(tf.assign(self.valueVariance, decay * (self.valueVariance + incr)))
			tf.summary.scalar('Action value variance', self.valueVariance, collections=['metrics'])
			# Action value cusum
			valueCusumPos = tf.get_variable('value_cusum_pos', shape=(), dtype=tf.float32, initializer=tf.initializers.zeros)
			valueCusumNeg = tf.get_variable('value_cusum_neg', shape=(), dtype=tf.float32, initializer=tf.initializers.zeros)
			self.updateMetrics.append(tf.assign(valueCusumPos, tf.maximum(0., decay * valueCusumPos + self.actionValue - self.meanValue)))
			self.updateMetrics.append(tf.assign(valueCusumNeg, tf.minimum(0., decay * valueCusumNeg + self.actionValue + self.meanValue)))
			self.valueCusum = valueCusumPos - valueCusumNeg
			self.updateMetrics.append(self.valueCusum)
			tf.summary.scalar('Action value cusum pos', valueCusumPos, collections=['metrics'])
			tf.summary.scalar('Action value cusum neg', valueCusumNeg, collections=['metrics'])
			tf.summary.scalar('Action value cusum', self.valueCusum, collections=['metrics'])
			# Reward mean
			self.reward = tf.placeholder(tf.float32, shape=(), name='reward')
			self.updateMetrics.append(ema.apply([self.reward]))
			self.meanReward = ema.average(self.reward)
			tf.summary.scalar('Reward', self.meanReward, collections=['metrics'])
			# Reward variance
			self.rewardVariance = tf.get_variable('reward_variance', shape=(), dtype=tf.float32, initializer=tf.initializers.zeros)
			incr = (1 - decay) * tf.square(self.reward - self.meanReward)
			self.updateMetrics.append(tf.assign(self.rewardVariance, decay * (self.rewardVariance + incr)))
			tf.summary.scalar('Reward variance', self.rewardVariance, collections=['metrics'])
			# Confidence
			valueConfidence = tf.get_variable('value_confidence', shape=(), dtype = tf.float32, initializer=tf.initializers.zeros)
			maxValueCusum = tf.get_variable('max_value_cusum', shape=(), dtype = tf.float32, initializer=tf.initializers.zeros)
			self.updateMetrics.append(tf.assign(maxValueCusum, tf.maximum(maxValueCusum, self.valueCusum)))
			self.updateMetrics.append(tf.assign(valueConfidence, tf.clip_by_value(valueConfidence + confidenceStep * tf.sign(self.settings["cusum-threshold"] - self.valueCusum), 0, 0.5)))
			self.confidence = valueConfidence
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


class Tester:

	def __init__(self, settings, environment, checkpointDir, logger):
		self.settings = settings
		self.env = environment
		self.logger = logger
		self.actionDim = self.env.action_space.low.size
		self.stateDim = self.env.observation_space.low.size
		self.action = tf.placeholder(tf.float32, (None, self.actionDim), name='action')
		self.state = tf.placeholder(tf.float32, (None, self.stateDim), name='state')
		self.isTraining = tf.placeholder_with_default(False, None, 'is_training')
		self.actor = Actor('actor', self.settings['actor'], self.state, self.isTraining, self.env.action_space.high, self.env.action_space.low)
		self.paths = tf.train.get_checkpoint_state(checkpointDir).all_model_checkpoint_paths
		self.globalStep = tf.train.get_or_create_global_step()

	def run(self):
		self.saver = tf.train.Saver(var_list=self.actor.parameters + [self.globalStep])
		self.session = tf.Session()
		self.session.run(tf.global_variables_initializer())
		for i, path in enumerate(self.paths):
			self.saver.restore(self.session, path)
			step = self.session.run(self.globalStep)
			cumReward = 0
			sys.stdout.flush()
			for episode in range(5):
				done = False
				state = self.env.reset()
				while not done:
					self.env.render()
					action = self._learnedPolicy(state)
					state, reward, done, _ = self.env.step(action)
					cumReward += reward
			self.logger.logScalar('Learning Curve', cumReward, step)
			print('Step {}\tAvg reward:{}'.format(step, cumReward / 5))

	def _learnedPolicy(self, state):
		action = self.session.run(self.actor.output, {
			self.state: [state],
			self.isTraining: False
		})
		return action[0]
