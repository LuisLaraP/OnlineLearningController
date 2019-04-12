import tensorflow as tf


class NeuralNetwork:

	def __init__(self):
		self.inputs = None
		self.output = None
		self.parameters = []
		self.summaries = None

	def getParameters(self):
		session = tf.get_default_session()
		return session.run(self.parameters)

	def predict(self, x):
		session = tf.get_default_session()
		return session.run(self.output, {self.input: x})

	def setParameters(self, values, tau=1.0):
		session = tf.get_default_session()
		oldValues = session.run(self.parameters)
		for i in range(len(values)):
			self.parameters[i].load(tau * values[i] + (1 - tau) * oldValues[i])


def buildNetwork(name, specs, inputs):
	model = NeuralNetwork()
	model.summaries = []
	model.inputs = inputs
	with tf.variable_scope(name):
		i = 1
		for layer in specs:
			funcName = '_{}Layer'.format(layer['type'])
			globals()[funcName](i, layer, model)
			if layer['type'] != 'input':
				i += 1
	model.summaries = tf.summary.merge(model.summaries)
	return model


def _denseLayer(i, specs, model):
	nIn = model.output.shape[-1]
	with tf.variable_scope('layer' + str(i)):
		model.parameters.append(tf.get_variable(
			'w',
			shape=(nIn, specs['units']),
			initializer=getattr(tf.initializers, specs['initializer'])()
		))
		model.output = tf.matmul(model.output, model.parameters[-1])
		model.parameters.append(tf.get_variable(
			'b',
			shape=(specs['units']),
			initializer=getattr(tf.initializers, specs['initializer'])()
		))
		model.output = model.output + model.parameters[-1]
		model.summaries.append(tf.summary.histogram('z', model.output))
		if specs['activation'] != 'linear':
			model.output = getattr(tf.nn, specs['activation'])(model.output)
		model.summaries.append(tf.summary.histogram('a', model.output))


def _inputLayer(i, specs, model):
	if model.output is None:
		model.output = model.inputs[specs['name']]
	else:
		model.output = tf.concat([model.output, model.inputs[specs['name']]], 1)
