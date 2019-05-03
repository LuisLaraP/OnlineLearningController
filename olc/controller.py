import tensorflow as tf

import olc.noise
from olc.neural_network import Actor, Critic
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
		self.critic = Critic('critic', self.action, self.state, self.isTraining)
		self.actorTarget = Actor('actor_target', self.state, self.isTraining, self.env.action_space.high, self.env.action_space.low)
		self.criticTarget = Critic('critic_target', self.action, self.state, self.isTraining)
		self.actor.createTrainOps(self.actionGrads, self.settings['batch-size'])
		self.critic.createTrainOps(self.qLabels)
		self.actorTarget.createUpdateOps(self.settings['tau'], self.actor.parameters)
		self.criticTarget.createUpdateOps(self.settings['tau'], self.critic.parameters)
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
