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

	def sample(self):
		"""
		Draw a random element in this space.

		Returns
		-------
		sample : array-like
			Element sampled.
		"""
		return np.random.uniform(self.low, np.nextafter(self.high, np.inf))