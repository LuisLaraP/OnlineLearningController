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
		session = tf.Session()
		session.run(tf.global_variables_initializer())
		# Initialize actor target parameters
		actorParams = session.run(self.actor.parameters)
		for f, t in zip(actorParams, self.actorTarget.parameters):
			t.load(f, session)
		# Initialize critic target parameters
		criticParams = session.run(self.critic.parameters)
		for f, t in zip(criticParams, self.criticTarget.parameters):
			t.load(f, session)
		# Training
		step = 0
		for episode in range(1, self.settings['episodes'] + 1):
			done = False
			episodeReward = 0
			state = self.env.reset()
			while not done:
				step += 1
				self.env.render()
				action = [0]
				state, reward, done, _ = self.env.step(action)
				episodeReward += reward
				[self.logger.logScalar('Action/' + str(x), x, step) for x in action]
				self.logger.logScalar('Reward', reward, step)
			self.logger.logScalar('Learning curve', episodeReward, episode)
			print('Episode {}:\tReward:{}'.format(episode, episodeReward))

	def _learnedPolicy(self, state):
		action, sums = self.session.run([self.actor.output, self.actor.summaries], {
			self.state: [state]
		})
		self.logger.writeSummary(sums, self.step)
		return action[0]

	def _randomPolicy(self, _):
		return self.noise.step() * (self.env.action_space.high - self.env.action_space.low) * self.settings['noise-scale']
