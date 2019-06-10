import numpy as np
from roboschool.scene_abstract import SingleRobotEmptyScene
from roboschool.gym_mujoco_xml_env import RoboschoolMujocoXmlEnv


class Reacher7(RoboschoolMujocoXmlEnv):

	definitionFile = 'reacher7.xml'

	def __init__(self):
		self.action_dim = 2
		self.obs_dim = 7
		RoboschoolMujocoXmlEnv.__init__(self, self.definitionFile, 'body0', action_dim=self.action_dim, obs_dim=self.obs_dim)
		self.theta = np.zeros(self.action_dim)
		self.theta_dot = np.zeros(self.action_dim)

	def create_single_player_scene(self):
		return SingleRobotEmptyScene(gravity=0.0, timestep=0.0165, frame_skip=1)

	def robot_specific_reset(self):
		self.jdict["target_x"].reset_current_position(self.np_random.uniform(low=-0.47, high=0.47), 0)
		self.jdict["target_y"].reset_current_position(self.np_random.uniform(low=-0.47, high=0.47), 0)
		self.jdict["target_z"].reset_current_position(self.np_random.uniform(low=0, high=0.47), 0)
		self.fingertip = self.parts["fingertip"]
		self.target = self.parts["target"]
		for i in range(self.action_dim):
			key = 'joint' + str(i)
			limits = self.jdict[key].limits()[:2]
			if limits[0] < limits[1]:
				self.jdict[key].reset_current_position(np.random.uniform(low=limits[0], high=limits[1]), 0)
			else:
				self.jdict[key].reset_current_position(np.random.uniform(low=3.14, high=3.14), 0)

	def apply_action(self, a):
		assert(np.isfinite(a).all())
		action = 0.05 * np.clip(a, -1, 1)
		for i in range(self.action_dim):
			key = 'joint' + str(i)
			self.jdict[key].set_motor_torque(action[i])

	def calc_state(self):
		for i in range(self.action_dim):
			key = 'joint' + str(i)
			self.theta[i], self.theta_dot[i] = self.jdict[key].current_relative_position()
		target_x, _ = self.jdict["target_x"].current_position()
		target_y, _ = self.jdict["target_y"].current_position()
		target_z, _ = self.jdict["target_z"].current_position()
		self.to_target_vec = np.array(self.fingertip.pose().xyz()) - np.array(self.target.pose().xyz())
		return np.append([target_x, target_y, target_z], [self.theta, self.theta_dot])

	def calc_potential(self):
		return -100 * np.linalg.norm(self.to_target_vec)

	def step(self, a):
		assert(not self.scene.multiplayer)
		self.apply_action(a)
		self.scene.global_step()
		state = self.calc_state()
		potential_old = self.potential
		self.potential = self.calc_potential()
		electricity_cost = -0.10 * np.dot(np.abs(a), np.abs(self.theta_dot)) - 0.01 * np.abs(a).sum()
		self.rewards = [float(self.potential - potential_old), float(electricity_cost), 0]
		self.frame += 1
		self.done += 0
		self.reward += sum(self.rewards)
		self.HUD(state, a, False)
		return state, sum(self.rewards), False, {}

	def camera_adjust(self):
		x, y, z = self.fingertip.pose().xyz()
		x *= 0.5
		y *= 0.5
		self.camera.move_and_look_at(0.3, 0.3, 0.3, x, y, z)
