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
	"""

	def __init__(self, settings):
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

	def close(self):
		"""Stop any running simulation and close connection to V-REP."""
		self.stop()
		vrep.simxFinish(self.id)

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
