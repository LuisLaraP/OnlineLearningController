import vrep


class Simulation:

	def __init__(self, robot):
		self.id = vrep.simxStart('127.0.0.1', 19997, True, True, 1000, 5)
		if self.id == -1:
			exit('Connection to V-REP failed.')

	def close(self):
		vrep.simxFinish(self.id)
