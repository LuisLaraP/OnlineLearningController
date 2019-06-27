"""Storage for previously seen transitions."""

import numpy as np
import tensorflow as tf


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
		with tf.variable_scope('replay_buffer', initializer=tf.initializers.zeros):
			self.si = tf.get_variable('s_i', (max_capacity, stateDim), dtype=tf.float32, trainable=False)
			self.a = tf.get_variable('a', (max_capacity, actionDim), dtype=tf.float32, trainable=False)
			self.r = tf.get_variable('r', (max_capacity,), dtype=tf.float32, trainable=False)
			self.sf = tf.get_variable('s_f', (max_capacity, stateDim), dtype=tf.float32, trainable=False)
			self.t = tf.get_variable('t', (max_capacity,), dtype=tf.bool, trainable=False)

	def restore(self, session):
		self.iState, self.action, self.reward, self.fState, self.terminal = session.run(
			[self.si, self.a, self.r, self.sf, self.t]
		)

	def save(self, session):
		self.si.load(self.iState, session)
		self.a.load(self.action, session)
		self.r.load(self.reward, session)
		self.sf.load(self.fState, session)
		self.t.load(self.terminal, session)

	def setCapacity(self, capacity):
		self.capacity = int(round(capacity))
		if self.size > self.capacity:
			self.size = self.capacity

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
