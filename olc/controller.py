import numpy as np
import tensorflow as tf

from olc.neural_network import Actor, Critic
from olc.noise import OrnsteinUhlenbeck
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
		self.isTraining = tf.placeholder_with_default(True, None, 'is_training')
		self.actor = Actor('actor', self.settings['actor'], self.state, self.isTraining, self.env.action_space.high, self.env.action_space.low)
		self.critic = Critic('critic', self.settings['critic'], self.action, self.state, self.isTraining)
		self.actorTarget = Actor('actor_target', self.settings['actor'], self.state, self.isTraining, self.env.action_space.high, self.env.action_space.low)
		self.criticTarget = Critic('critic_target', self.settings['critic'], self.action, self.state, self.isTraining)
		self.actor.createTrainOps(self.actionGrads, self.settings['batch-size'])
		self.critic.createTrainOps(self.action, self.qLabels)
		self.actorTarget.createUpdateOps(self.settings['tau'], self.actor.parameters)
		self.criticTarget.createUpdateOps(self.settings['tau'], self.critic.parameters)
		self.logger.logGraph()

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
		self.buffer = ReplayBuffer(self.settings['replay-buffer-size'])
		# Create noise process
		self.noise = OrnsteinUhlenbeck(self.actionDim,
			self.settings['noise']['dt'],
			self.settings['noise']['theta'],
			self.settings['noise']['sigma']
		)
		# Initialize replay buffer
		state = self.env.reset()
		for i in range(self.settings['replay-buffer-init']):
			action = self.env.action_space.sample()
			newState, reward, done, _ = self.env.step(action)
			self.buffer.storeTransition(state, action, reward, newState, done)
			state = newState
			if done:
				state = self.env.reset()
			if i % 1000 == 0:
				print(i)
		# Training
		step = 0
		for episode in range(1, self.settings['episodes'] + 1):
			done = False
			episodeReward = 0
			self.noise.reset()
			state = self.env.reset()
			while not done:
				step += 1
				self.env.render()
				action = self._learnedPolicy(state) + self._randomPolicy(state)
				newState, reward, done, _ = self.env.step(action)
				self.buffer.storeTransition(state, action, reward, newState, done)
				state = newState
				self._train(step)
				self.session.run([self.actorTarget.update, self.criticTarget.update])
				episodeReward += reward
				[self.logger.logScalar('Action/' + str(i), x, step) for i, x in enumerate(action)]
				self.logger.logScalar('Reward', reward, step)
			self.logger.logScalar('Learning curve', episodeReward, episode)
			print('Episode {}:\tReward:{}'.format(episode, episodeReward))

	def _learnedPolicy(self, state):
		action = self.session.run(self.actor.output, {
			self.state: [state],
			self.isTraining: False
		})
		return action[0]

	def _randomPolicy(self, _):
		return self.noise.step() * (self.env.action_space.high - self.env.action_space.low)

	def _train(self, step):
		siBatch, aBatch, rBatch, sfBatch, tBatch = self.buffer.sample(self.settings['batch-size'])
		loss = 0
		if len(siBatch) > 0:
			# Critic
			actions = self.session.run(self.actorTarget.output, {
				self.state: sfBatch
			})
			qValues = self.session.run(self.criticTarget.output, {
				self.action: actions,
				self.state: sfBatch
			})
			labels = np.reshape(rBatch, (rBatch.size, 1)) + self.settings['gamma'] * qValues
			labels[tBatch] = 0
			_, loss = self.session.run([self.critic.train, self.critic.loss], {
				self.action: aBatch,
				self.state: siBatch,
				self.qLabels: labels
			})
			# Actor
			actions = self.session.run(self.actor.output, {
				self.state: siBatch
			})
			actionGrads = self.session.run(self.critic.actionGrads, {
				self.action: actions,
				self.state: siBatch,
			})[0]
			self.session.run(self.actor.train, {
				self.state: siBatch,
				self.actionGrads: actionGrads
			})
		self.logger.logScalar('Critic loss', loss, step)
