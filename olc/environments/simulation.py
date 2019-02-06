"""An object for managing a connection to V-REP."""

import vrep

from olc.settings import getDefaults


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
		self.id = vrep.simxStart(
			self.settings['connection_address'],
			self.settings['connection_port'],
			self.settings['wait_until_connected'],
			self.settings['reconnect'],
			self.settings['timeout'],
			self.settings['comm_cycle']
		)
		if self.id == -1:
			print('Connection to V-REP failed.')
			exit(1)

	def close(self):
		"""Stop any running simulation and close connection to V-REP."""
		vrep.simxFinish(self.id)
