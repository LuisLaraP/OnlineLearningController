import tensorflow as tf


class NeuralNetwork:

	def __init__(self):
		self.inputs = None
		self.output = None
		self.parameters = []

	def predict(self, x):
		session = tf.get_default_session()
		return session.run(self.output, {self.input: x})


def buildNetwork(name, specs, inputs):
	model = NeuralNetwork()
	model.inputs = inputs
	with tf.variable_scope(name):
		for i, layer in enumerate(specs):
			funcName = '_{}Layer'.format(layer['type'])
			globals()[funcName](i, layer, model)
	return model


def _denseLayer(i, specs, model):
	nIn = model.output.shape[-1]
	model.parameters.append(tf.get_variable(
		'w' + str(i),
		shape=(nIn, specs['units']),
		initializer=getattr(tf.initializers, specs['initializer'])
	))
	model.output = tf.matmul(model.output, model.parameters[-1])
	model.parameters.append(tf.get_variable(
		'b' + str(i),
		shape=(specs['units']),
		initializer=getattr(tf.initializers, specs['initializer'])
	))
	model.output = model.output + model.parameters[-1]
	if specs['activation'] != 'linear':
		model.output = getattr(tf.nn, specs['activation'])(model.output)


def _inputLayer(i, specs, model):
	if model.output is None:
		model.output = model.inputs[specs['name']]
	else:
		model.output = tf.concat([model.output, model.inputs[specs['name']]], 1)
