import tensorflow as tf


def buildNetwork(name, specs):
	with tf.variable_scope(name):
		model = tf.keras.Sequential()
	return model
