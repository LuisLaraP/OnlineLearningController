import math

import matplotlib.pyplot as plt


def _factorize(n):
	n = int(n)
	for f in range(math.floor(math.sqrt(n)), 0, -1):
		if n % f == 0:
			return f, int(n / f)


class Logger:

	def __init__(self, filename, visualize=None):
		self.file = open(filename, 'w')
		self.graph = {}
		if visualize is not None:
			shape = _factorize(len(visualize))
			plt.ion()
			self.fig, axes = plt.subplots(max(shape), min(shape), squeeze=False)
			for i in range(len(visualize)):
				self.graph[visualize[i]] = axes[i // min(shape)][i % min(shape)]
				plt.pause(0.0001)

	def close(self):
		self.file.close()

	def log(self, values):
		strs = [str(x) for x in values]
		self.file.write('\t'.join(strs) + '\n')

	def setNames(self, names):
		self.file.write('\t'.join(names) + '\n')
