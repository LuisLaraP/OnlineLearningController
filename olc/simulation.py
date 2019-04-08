import vrep


class Simulation:
	"""
	Abstraction of a connection to V-REP.

	Parameters
	----------
	robot : dict
		Robot specs.
	"""

	def __init__(self, robot):
		self.id = vrep.simxStart("127.0.0.1", 19997, True, True, 5000, 5)
		if self.id == -1:
			exit('Connection to V-REP failed.')
		self.robot = robot
		self.running = False
		self.joints = []
		for j in robot['joints']:
			_, handle = vrep.simxGetObjectHandle(self.id, j, vrep.simx_opmode_blocking)
			self.joints.append(handle)
			vrep.simxGetJointPosition(self.id, handle, vrep.simx_opmode_streaming)
			vrep.simxGetObjectFloatParameter(self.id, handle,
				vrep.sim_jointfloatparam_velocity, vrep.simx_opmode_streaming)
		self.distances = {}
		self.dummies = {}

	def close(self):
		self.stop()
		vrep.simxFinish(self.id)

	def getJointPositions(self):
		"""
		Get the current position of all joints.

		Returns
		-------
		positions : array-like
			Vector containing the joint positions.
		"""
		positions = []
		for j in self.joints:
			_, p = vrep.simxGetJointPosition(self.id, j, vrep.simx_opmode_buffer)
			positions.append(p)
		return positions

	def getJointVelocities(self):
		"""
		Get the current velocity of all joints.

		Returns
		-------
		velocities : array-like
			Vector containing the joint velocities.
		"""
		velocities = []
		for j in self.joints:
			_, v = vrep.simxGetObjectFloatParameter(self.id, j,
				vrep.sim_jointfloatparam_velocity, vrep.simx_opmode_buffer)
			velocities.append(v)
		return velocities

	def readDistance(self, name):
		"""
		Get the value of a previously registered distance.

		Parameters
		----------
		name : str
			Name of the distance object.

		Returns
		-------
		distance : float
			Current distance value.

		See Also
		--------
		registerDistanceObject : Register a new distance object.
		"""
		return vrep.simxReadDistance(self.id, self.distances[name],
			vrep.simx_opmode_buffer)[1]

	def registerDistanceObject(self, name):
		_, self.distances[name] = vrep.simxGetDistanceHandle(self.id, name,
			vrep.simx_opmode_blocking)
		vrep.simxReadDistance(self.id, self.distances[name],
			vrep.simx_opmode_streaming)

	def registerDummyObject(self, name):
		_, self.dummies[name] = vrep.simxGetObjectHandle(self.id, name,
			vrep.simx_opmode_blocking)

	def setJointVelocities(self, vels):
		vels *= self.robot['speed-override']
		for i in range(len(self.joints)):
			vrep.simxSetJointTargetVelocity(self.id, self.joints[i], vels[i],
				vrep.simx_opmode_oneshot)

	def start(self):
		if not self.running:
			vrep.simxStartSimulation(self.id, vrep.simx_opmode_blocking)
		self.running = True

	def stop(self):
		if self.running:
			vrep.simxStopSimulation(self.id, vrep.simx_opmode_blocking)
		self.running = False
