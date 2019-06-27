import tensorflow as tf


class Logger:

	def __init__(self, name):
		self.writer = tf.summary.FileWriter('logs/' + name)
		self.saver = None
		self.savePath = 'checkpoints/' + name + '/step'

	def checkpoint(self, session, step):
		if self.saver is None:
			self.saver = tf.train.Saver(max_to_keep=1)
		self.saver.save(session, self.savePath, global_step=step)

	def loadCheckpoint(self, session, path):
		if self.saver is None:
			self.saver = tf.train.Saver(max_to_keep=1)
		self.saver.restore(session, path)

	def logGraph(self):
		self.writer.add_graph(tf.get_default_graph())
		self.writer.flush()

	def logScalar(self, name, value, step):
		summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])
		self.writer.add_summary(summary, step)

	def writeSummary(self, summary, step):
		self.writer.add_summary(summary, step)
