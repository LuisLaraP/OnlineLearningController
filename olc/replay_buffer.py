"""Storage for previously seen transitions."""

import collections
import random

import numpy as np


class ReplayBuffer:

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

	def setCapacity(self, capacity):
		self.capacity = capacity
		if self.size > self.capacity:
			self.size = capacity

	def storeTransition(self, si, a, r, sf, t):
		self.iState[self.head, :] = si
		self.action[self.head, :] = a
		self.reward[self.head] = r
		self.fState[self.head, :] = sf
		self.terminal[self.head] = t
		if self.size < self.capacity:
			self.size += 1
		self.head = (self.head + 1) % self.max_capacity

	def sample(self, n):
		if n > self.size:
			return [], [], [], [], []
		idx = np.random.choice(self.size, n, replace=False)
		idx = (self.head - idx - 1 + self.max_capacity) % self.max_capacity
		return self.iState[idx, :], self.action[idx, :], self.reward[idx], self.fState[idx, :], self.terminal[idx]
