import vrep


class Simulation:

	def __init__(self, robot):
		self.id = vrep.simxStart('127.0.0.1', 19997, True, True, 1000, 5)
		if self.id == -1:
			exit('Connection to V-REP failed.')
		self.running = False

	def close(self):
		self.stop()
		vrep.simxFinish(self.id)

	def start(self):
		if not self.running:
			vrep.simxSynchronous(self.id, True)
			vrep.simxStartSimulation(self.id, vrep.simx_opmode_blocking)
			self.running = True

	def step(self):
		vrep.simxSynchronousTrigger(self.id)

	def stop(self):
		if self.running:
			vrep.simxStopSimulation(self.id, vrep.simx_opmode_blocking)
			self.running = False
