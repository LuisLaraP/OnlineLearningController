import numpy as np
import vrep


class Simulation:

	def __init__(self, robot):
		self.id = vrep.simxStart('127.0.0.1', 19997, True, True, 1000, 5)
		if self.id == -1:
			exit('Connection to V-REP failed.')
		self.running = False
		self.joints = []
		for joint in robot['joints']:
			self.joints.append(vrep.simxGetObjectHandle(self.id, joint, vrep.simx_opmode_blocking)[1])
			vrep.simxGetJointPosition(self.id, self.joints[-1], vrep.simx_opmode_streaming)
			vrep.simxGetObjectFloatParameter(self.id, self.joints[-1], vrep.sim_jointfloatparam_velocity, vrep.simx_opmode_streaming)

	def close(self):
		self.stop()
		vrep.simxFinish(self.id)

	def getRobotState(self):
		pos = np.zeros(len(self.joints))
		vel = np.zeros(len(self.joints))
		vrep.simxPauseCommunication(self.id, True)
		for i, joint in enumerate(self.joints):
			pos[i] = vrep.simxGetJointPosition(self.id, joint, vrep.simx_opmode_buffer)[1]
			vel[i] = vrep.simxGetObjectFloatParameter(self.id, joint, vrep.sim_jointfloatparam_velocity, vrep.simx_opmode_buffer)[1]
		vrep.simxPauseCommunication(self.id, False)
		return np.concatenate((pos, vel))

	def setTorques(self, torques):
		vrep.simxPauseCommunication(self.id, True)
		for j, t in zip(self.joints, torques):
			vrep.simxSetJointTargetVelocity(self.id, j, np.sign(t) * 1e10, vrep.simx_opmode_oneshot)
			vrep.simxSetJointForce(self.id, j, np.abs(t), vrep.simx_opmode_oneshot)
		vrep.simxPauseCommunication(self.id, False)

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
