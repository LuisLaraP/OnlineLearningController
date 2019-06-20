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

	def __init__(self, max_capacity, actionDim, stateDim):
		self.max_capacity = max_capacity
		self.capacity = max_capacity
		self.head = 0
		self.size = 0
		self.iState = np.zeros((max_capacity, stateDim))
		self.action = np.zeros((max_capacity, actionDim))
		self.reward = np.zeros(max_capacity)
		self.fState = np.zeros((max_capacity, stateDim))
		self.terminal = np.zeros(max_capacity, bool)

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
		self.iState[self.head, :] = si
		self.action[self.head, :] = a
		self.reward[self.head] = r
		self.fState[self.head, :] = sf
		self.terminal[self.head] = t
		if self.size < self.capacity:
			self.size += 1
		self.head = (self.head + 1) % self.max_capacity

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
		if n > self.size:
			return [], [], [], [], []
		idx = np.random.choice(self.size, n, replace=False)
		idx = self.head - idx - 1
		return self.iState[idx, :], self.action[idx, :], self.reward[idx], self.fState[idx, :], self.terminal[idx]
