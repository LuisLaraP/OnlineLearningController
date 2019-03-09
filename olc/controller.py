from olc.neural_network import buildNetwork


class Controller:

	def __init__(self, settings, environment, logger):
		self.settings = settings
		self.env = environment
		self.logger = logger
		self.logger.setNames(['Reward'])
		self.q = buildNetwork('Q', None)

	def run(self):
		"""
		Run an experiment on the environment.

		The simulation will run for exactly the amount of steps specified in the
		settings. If an episode ends before reaching the target number of steps, the
		environment is reset and the experiment continues.
		"""
		reset = True
		for _ in range(self.settings['steps']):
			if reset:
				state = self.env.reset()
			self.env.render()
			action = self.env.action_space.sample()
			state, reward, reset, info = self.env.step(action)
			self.logger.log([reward])
