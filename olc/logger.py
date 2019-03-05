import collections
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
		self.names = None
		self.buffers = {}
		self.axes = {}
		self.limits = {}
		self.lines = {}
		if visualize is not None:
			shape = _factorize(len(visualize))
			plt.ion()
			self.fig, axes = plt.subplots(max(shape), min(shape), squeeze=False)
			for i in range(len(visualize)):
				self.buffers[i] = collections.deque([0] * 100, maxlen=100)
				self.axes[i] = axes[i // min(shape)][i % min(shape)]
				self.lines[i] = self.axes[i].plot(range(100), self.buffers[i])[0]
				self.limits[i] = [0, 1]
				plt.pause(0.0001)

	def close(self):
		self.file.close()

	def log(self, values):
		strs = [str(x) for x in values]
		self.file.write('\t'.join(strs) + '\n')
		for i in range(len(values)):
			if i not in self.lines:
				continue
			self.buffers[i].append(values[i])
			self.lines[i].set_ydata(self.buffers[i])
			if min(self.buffers[i]) < self.limits[i][0]:
				self.limits[i][0] = min(self.buffers[i])
				self.axes[i].set_ylim(self.limits[i])
			if max(self.buffers[i]) < self.limits[i][1]:
				self.limits[i][1] = max(self.buffers[i])
				self.axes[i].set_ylim(self.limits[i])
			plt.pause(0.0001)

	def setNames(self, names):
		self.names = names
		self.file.write('\t'.join(names) + '\n')
