import tensorflow as tf


class Logger:

	def __init__(self, directory):
		self.writer = tf.summary.FileWriter(directory)

	def log(self, values):
		pass
