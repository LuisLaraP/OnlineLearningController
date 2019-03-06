"""Storage for previously seen transitions."""

import random


class ReplayBuffer:
	"""Storage for previously seen transitions.

	Parameters
	----------
	size : int
		Maximum number of elements to be stored.
	"""

	def __init__(self, size):
		self.size = size
		self.iState = []
		self.action = []
		self.reward = []
		self.fState = []
		self.terminal = []

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
		if len(self.action) == self.size:
			self.iState.pop(0)
			self.action.pop(0)
			self.reward.pop(0)
			self.fState.pop(0)
			self.terminal.pop(0)
		self.iState.append(si)
		self.action.append(a)
		self.reward.append(r)
		self.fState.append(sf)
		self.terminal.append(t)

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
		if n > len(self.action):
			return [], [], [], [], []
		idx = random.sample(range(len(self.action)), n)
		iStateList = [self.iState[x] for x in idx]
		actionList = [self.action[x] for x in idx]
		rewardList = [self.reward[x] for x in idx]
		fStateList = [self.fState[x] for x in idx]
		terminalList = [self.terminal[x] for x in idx]
		return iStateList, actionList, rewardList, fStateList, terminalList
