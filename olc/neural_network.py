import tensorflow as tf


class NeuralNetwork:

	def __init__(self):
		self.input = None
		self.output = None


def buildNetwork(name, specs):
	model = NeuralNetwork()
	with tf.variable_scope(name):
		model.input = tf.placeholder(tf.float32, name='input')
		model.output = model.input
	return model
