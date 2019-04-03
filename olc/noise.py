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
		wiener = np.random.normal(self.mu, self.dt, self.x.shape)
		dx = self.theta * (self.mu - self.x) * self.dt + self.sigma * wiener
		self.x += dx
		return self.x


class Sinusoid:

	def __init__(self, ndim, dt, basePeriod):
		self.dt = dt
		t = np.linspace(basePeriod, basePeriod / 2.0, num=ndim)
		self.f = 2 * np.pi / t
		self.x = 0.0

	def reset(self):
		self.x = 0.0

	def step(self):
		self.x += self.dt
		return np.sin(self.x * self.f)
