import tensorflow as tf

import olc.noise
from olc.neural_network import Actor
from olc.replay_buffer import ReplayBuffer


class Controller:

	def __init__(self, settings, environment, logger):
		self.settings = settings
		self.env = environment
		self.logger = logger
		self.actionDim = self.env.action_space.low.size
		self.stateDim = self.env.observation_space.low.size
		self.action = tf.placeholder(tf.float32, (None, self.actionDim), name='action')
		self.state = tf.placeholder(tf.float32, (None, self.stateDim), name='state')
		self.qLabels = tf.placeholder(tf.float32, (None, 1), name='q_labels')
		self.actionGrads = tf.placeholder(tf.float32, (None, self.actionDim), name='action_gradients')
		self.isTraining = tf.placeholder_with_default(False, None, 'is_training')
		self.actor = Actor('actor', self.state, self.isTraining, self.env.action_space.high, self.env.action_space.low)
		self.logger.logGraph()

	def run(self):
		pass

	def _learnedPolicy(self, state):
		action, sums = self.session.run([self.actor.output, self.actor.summaries], {
			self.state: [state]
		})
		self.logger.writeSummary(sums, self.step)
		return action[0]

	def _randomPolicy(self, _):
		return self.noise.step() * (self.env.action_space.high - self.env.action_space.low) * self.settings['noise-scale']
