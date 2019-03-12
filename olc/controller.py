import time

from olc.neural_network import buildNetwork


class Controller:

	def __init__(self, settings, network, environment, logger):
		self.settings = settings
		self.env = environment
		self.logger = logger
		self.q = buildNetwork('Q', network)

	def run(self):
		"""
		Run an experiment on the environment.

		The simulation will run for exactly the amount of steps specified in the
		settings. If an episode ends before reaching the target number of steps, the
		environment is reset and the experiment continues.
		"""
		episode = 0
		step = 0
		while step <= self.settings['steps']:
			episode += 1
			state = self.env.reset()
			reset = False
			lastTime = time.time()
			while not reset and step <= self.settings['steps']:
				step += 1
				state, reward, reset = self.env.getState()
				action = self.env.action_space.sample()
				self.env.act(action)
				time.sleep(self.settings['timestep'])
				newTime = time.time()
				elapsed = (newTime - lastTime) * 1000
				lastTime = newTime
				self.logger.logScalar('Reward', reward, step)
				self.logger.logScalar('Sampling time', elapsed, step)
