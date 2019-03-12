import tensorflow as tf


class Logger:

	def __init__(self, directory):
		self.writer = tf.summary.FileWriter(directory)

	def logScalar(self, name, value, step):
		summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])
		self.writer.add_summary(summary, step)
