"""Spaces for actions and states."""

import numpy as np


class Box:
	"""
	A continuous space where each coordinate is bounded.

	Parameters
	----------
	low : array-like
		Lower bounds for each coordinate.
	high : array-like
		Upper bounds for each coordinate.
	"""

	def __init__(self, low, high):
		self.low = np.array(low)
		self.high = np.array(high)
		assert self.low.shape == self.high.shape

	def clip(self, x):
		iLow = np.greater(self.low, x)
		x[iLow] = self.low[iLow]
		iHigh = np.greater(x, self.high)
		x[iHigh] = self.high[iHigh]

	def sample(self):
		"""
		Draw a random element in this space.

		Returns
		-------
		sample : array-like
			Element sampled.
		"""
		return np.random.uniform(self.low, np.nextafter(self.high, np.inf))

	def scale(self, x):
		ret = np.zeros(x.shape)
		ret[x > 0] = np.multiply(x[x > 0], self.high[x > 0])
		ret[x < 0] = -np.multiply(x[x < 0], self.low[x < 0])
		return ret
