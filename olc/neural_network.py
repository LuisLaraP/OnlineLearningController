import tensorflow as tf


class NeuralNetwork:

	def __init__(self):
		self.input = None
		self.output = None
		self.parameters = []

	def predict(self, x):
		session = tf.get_default_session()
		return session.run(self.output, {self.input: x})


def buildNetwork(name, specs, nIn):
	model = NeuralNetwork()
	with tf.variable_scope(name):
		model.input = tf.placeholder(tf.float32, shape=(None, nIn), name='input')
		model.output = model.input
		for i, layer in enumerate(specs):
			funcName = '_{}Layer'.format(layer['type'])
			globals()[funcName](i, layer, model)
	return model


def _denseLayer(i, specs, model):
	nIn = model.output.shape[-1]
	model.parameters.append(tf.get_variable(
		'w' + str(i),
		shape=(nIn, specs['units']),
		initializer=tf.initializers.zeros()
	))
	model.output = tf.matmul(model.output, model.parameters[-1])
	model.parameters.append(tf.get_variable(
		'b' + str(i),
		shape=(specs['units']),
		initializer=tf.initializers.zeros()
	))
	model.output = model.output + model.parameters[-1]
