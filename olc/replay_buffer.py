"""Storage for previously seen transitions."""

import collections
import random

import numpy as np


class ReplayBuffer:
	"""Storage for previously seen transitions.

	Parameters
	----------
	size : int
		Maximum number of elements to be stored.
	"""

	def __init__(self, size, actionDim, stateDim):
		self.size = size
		self.nElem = 0
		self.nextIdx = 0
		self.iState = np.zeros((size, stateDim))
		self.action = np.zeros((size, actionDim))
		self.reward = np.zeros(size)
		self.fState = np.zeros((size, stateDim))
		self.terminal = np.zeros(size, bool)

	def storeTransition(self, si, a, r, sf, t):
		"""Add the given transition to the buffer.

		If the buffer is full, the oldest entry will be deleted.

		Parameters
		----------
		s1 : Numpy array
			Initial state of the transition.
		a : int
			Action taken in the transition.
		r : float
			Reward received after taking action `a`.
		sf : Numpy array
			Final state of the transition.
		t : bool
			True if the given transition was the last in its episode.
		"""
		if self.nElem < self.size:
			self.nElem += 1
		self.iState[self.nextIdx, :] = si
		self.action[self.nextIdx, :] = a
		self.reward[self.nextIdx] = r
		self.fState[self.nextIdx, :] = sf
		self.terminal[self.nextIdx] = t
		self.nextIdx = (self.nextIdx + 1) % self.size

	def sample(self, n):
		"""Return a random set of transitions from the buffer.

		If `n` is greater than the number of transitions stored, this method returns
		empty lists.

		Parameters
		----------
		n : int
			Number of transitions to return.

		Returns
		-------
		iState : List of numpy arrays
			Initial state of the transition.
		action : List of int
			Action taken in the transition.
		reward : float
			Reward received after taking action `a`.
		fState : List of numpy arrays
			Final state of the transition.
		terminal : List of bool
			True if the given transition was the last in its episode.
		"""
		if n > self.nElem:
			return [], [], [], [], []
		idx = random.sample(range(self.nElem), n)
		return self.iState[idx, :], self.action[idx, :], self.reward[idx], self.fState[idx, :], self.terminal[idx]
