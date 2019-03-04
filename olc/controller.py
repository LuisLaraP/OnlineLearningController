class Controller:
	"""
	Abstraction of the controller.

	Parameters
	----------
	environment
		Class compatible with OpenAI Gym's API to run the controller on.

	Attributes
	----------
	env
		Current environment class.
	"""

	def __init__(self, environment, logger):
		self.env = environment
		self.logger = logger
		self.logger.setNames(['Error'])

	def run(self, steps):
		"""
		Run an experiment on the environment.

		The simulation will run for exactly the amount of steps specified. If
		an episode ends before reaching the target number of steps, the
		environment is reset and the experiment continues.

		Parameters
		----------
		steps : int
			Amount of steps to perform.
		"""
		reset = True
		for _ in range(steps):
			if reset:
				state = self.env.reset()
			self.env.render()
			action = self.env.action_space.sample()
			state, reward, reset, info = self.env.step(action)
