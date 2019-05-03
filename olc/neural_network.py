import numpy as np
import tensorflow as tf


class Actor:

	def __init__(self, name, state, isTraining, boundHigh, boundLow):
		with tf.variable_scope(name):
			self.input = state
			self.output = self.input
			with tf.variable_scope('layer_1'):
				fanIn = state.shape[-1].value
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
