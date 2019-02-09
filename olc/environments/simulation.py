"""An object for managing a connection to V-REP."""

import vrep

from olc.settings import getDefaults, merge


class Simulation:
	"""
	Abstraction of a connection to V-REP.

	Parameters
	----------
	settings : dict
		Connection settings.
	robot : dict
		Robot specs.
	"""

	def __init__(self, settings, robot):
		self.settings = getDefaults(__name__ + ':simulation_defaults.json')
		self.settings = merge(self.settings, settings)
		self.id = vrep.simxStart(
			self.settings['connection_address'],
			self.settings['connection_port'],
			self.settings['wait_until_connected'],
			self.settings['reconnect'],
			self.settings['timeout'],
			self.settings['comm_cycle']
		)
		if self.id == -1:
			exit('Connection to V-REP failed.')
		self.running = False
		self.joints = []
		for j in robot['joints']:
			_, handle = vrep.simxGetObjectHandle(self.id, j, vrep.simx_opmode_blocking)
			self.joints.append(handle)
			vrep.simxGetJointPosition(self.id, handle, vrep.simx_opmode_streaming)
			vrep.simxGetObjectFloatParameter(self.id, handle,
				vrep.sim_jointfloatparam_velocity, vrep.simx_opmode_streaming)

	def close(self):
		"""Stop any running simulation and close connection to V-REP."""
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

	def setJointVelocities(self, vels):
		"""
		Set the target velocity for each joint.

		Parameters
		----------
		vls : array-like
			Vector containing the desired valocities.
		"""
		for i in range(len(self.joints)):
			vrep.simxSetJointTargetVelocity(self.id, self.joints[i], vels[i],
				vrep.simx_opmode_oneshot)

	def start(self):
		"""
		Start the simulation.

		If there is already a simulation running, this method does nothing.
		"""
		if not self.running:
			vrep.simxStartSimulation(self.id, vrep.simx_opmode_blocking)
		self.running = True

	def stop(self):
		"""
		Stop the simulation.

		This method does nothing if there is no simulation running.
		"""
		if self.running:
			vrep.simxStopSimulation(self.id, vrep.simx_opmode_blocking)
		self.running = False
