import tensorflow as tf
from tensorflow.keras import layers


def buildNetwork(name, specs):
	with tf.variable_scope(name):
		model = tf.keras.Sequential()
		for layer in specs:
			if layer['type'] == 'dense':
				model.add(layers.Dense(layer['units']))
	return model
