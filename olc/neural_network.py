import tensorflow as tf


class NeuralNetwork:

	def __init__(self):
		self.input = None
		self.output = None

	def predict(self, x):
		session = tf.get_default_session()
		return session.run(self.output, {self.input: x})


def buildNetwork(name, specs, nIn):
	model = NeuralNetwork()
	with tf.variable_scope(name):
		model.input = tf.placeholder(tf.float32, shape=(None, nIn), name='input')
		model.output = model.input
	return model
