import numpy as np
import tensorflow as tf


class Actor:

	def __init__(self, name, state, isTraining, boundHigh, boundLow):
		with tf.variable_scope(name):
			self.output = state
			with tf.variable_scope('layer_1'):
				fanIn = self.output.shape[-1].value
				self.output = tf.keras.layers.Dense(400,
					bias_initializer=tf.initializers.random_uniform(-1 / np.sqrt(fanIn), 1 / np.sqrt(fanIn)),
					kernel_initializer=tf.initializers.random_uniform(-1 / np.sqrt(fanIn), 1 / np.sqrt(fanIn))
				)(self.output)
				self.output = tf.keras.layers.Activation('relu')(self.output)
			with tf.variable_scope('layer_2'):
				fanIn = self.output.shape[-1].value
				self.output = tf.keras.layers.Dense(300,
					bias_initializer=tf.initializers.random_uniform(-1 / np.sqrt(fanIn), 1 / np.sqrt(fanIn)),
					kernel_initializer=tf.initializers.random_uniform(-1 / np.sqrt(fanIn), 1 / np.sqrt(fanIn))
				)(self.output)
				self.output = tf.keras.layers.Activation('relu')(self.output)
			with tf.variable_scope('layer_3'):
				self.output = tf.keras.layers.Dense(boundHigh.size,
					bias_initializer=tf.initializers.random_uniform(-3e-3, 3e-3),
					kernel_initializer=tf.initializers.random_uniform(-3e-3, 3e-3)
				)(self.output)
				self.output = tf.keras.layers.Activation('tanh')(self.output)
			self.output = tf.multiply(self.output, (boundHigh - boundLow) / 2.0)
			self.output = tf.add(self.output, (boundHigh + boundLow) / 2.0)
		self.parameters = tf.trainable_variables(scope=name)

	def createTrainOps(self, actionGrad, batchSize):
		with tf.variable_scope('train_actor'):
			self.gradient = tf.gradients(self.output, self.parameters, -actionGrad)
			self.gradient = [x / batchSize for x in self.gradient]
			optimizer = tf.train.AdamOptimizer(1e-4)
			self.train = optimizer.apply_gradients(zip(self.gradient, self.parameters))

	def createUpdateOps(self, tau, actorParams):
		with tf.variable_scope('update_actor_target'):
			self.update = []
			for old, new in zip(self.parameters, actorParams):
				self.update.append(tf.assign(new, new * tau + old * (1 - tau)))


class Critic:

	def __init__(self, name, action, state, isTraining):
		with tf.variable_scope(name):
			self.output = state
			with tf.variable_scope('layer_1'):
				fanIn = self.output.shape[-1].value
				self.output = tf.keras.layers.Dense(400,
					bias_initializer=tf.initializers.random_uniform(-1 / np.sqrt(fanIn), 1 / np.sqrt(fanIn)),
					kernel_initializer=tf.initializers.random_uniform(-1 / np.sqrt(fanIn), 1 / np.sqrt(fanIn))
				)(self.output)
				self.output = tf.keras.layers.Activation('relu')(self.output)
			with tf.variable_scope('layer_2'):
				fanIn = self.output.shape[-1].value
				a = tf.keras.layers.Dense(300,
					bias_initializer=tf.initializers.random_uniform(-1 / np.sqrt(fanIn), 1 / np.sqrt(fanIn)),
					kernel_initializer=tf.initializers.random_uniform(-1 / np.sqrt(fanIn), 1 / np.sqrt(fanIn))
				)(self.output)
				b = tf.keras.layers.Dense(300,
					bias_initializer=tf.initializers.random_uniform(-1 / np.sqrt(fanIn), 1 / np.sqrt(fanIn)),
					kernel_initializer=tf.initializers.random_uniform(-1 / np.sqrt(fanIn), 1 / np.sqrt(fanIn))
				)(action)
				self.output = tf.keras.layers.Activation('relu')(a + b)
			with tf.variable_scope('layer_3'):
				self.output = tf.keras.layers.Dense(1,
					bias_initializer=tf.initializers.random_uniform(-3e-3, 3e-3),
					kernel_initializer=tf.initializers.random_uniform(-3e-3, 3e-3)
				)(self.output)
		self.parameters = tf.trainable_variables(scope=name)

	def createTrainOps(self, labels):
		with tf.variable_scope('train_critic'):
			self.loss = tf.losses.mean_squared_error(labels, self.output)
			self.loss += sum([tf.nn.l2_loss(x) for x in self.parameters])
			optimizer = tf.train.AdamOptimizer(1e-3)
			self.train = optimizer.minimize(self.loss)

	def createUpdateOps(self, tau, criticParams):
		with tf.variable_scope('update_critic_target'):
			self.update = []
			for old, new in zip(self.parameters, criticParams):
				self.update.append(tf.assign(new, new * tau + old * (1 - tau)))
