import numpy as np


class OrnsteinUhlenbeck:

	def __init__(self, ndim, dt, theta, sigma, mu=0):
		self.dt = dt
		self.theta = theta
		self.sigma = sigma
		self.mu = mu
		self.x = np.zeros(ndim)

	def reset(self):
		self.x = np.zeros(self.x.shape)

	def step(self):
		dx = self.theta * (self.mu - self.x) * self.dt + self.sigma * np.random.normal(self.mu, self.dt)
		self.x += dx
		return self.x
