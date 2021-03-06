from roboschool.scene_abstract import SingleRobotEmptyScene
from roboschool.gym_mujoco_xml_env import RoboschoolMujocoXmlEnv
import numpy as np


class Reacher3(RoboschoolMujocoXmlEnv):
    '''
    Get the end of two-link robotic arm to a given spot.
    Similar to MuJoCo reacher.
    '''
    def __init__(self):
        RoboschoolMujocoXmlEnv.__init__(self, self.definitionFile, 'body0', action_dim=3, obs_dim=11)

    def create_single_player_scene(self):
        return SingleRobotEmptyScene(gravity=0.0, timestep=0.0165, frame_skip=1)

    TARG_LIMIT = 0.27
    def robot_specific_reset(self):
        self.jdict["target_x"].reset_current_position(self.np_random.uniform( low=-self.TARG_LIMIT, high=self.TARG_LIMIT ), 0)
        self.jdict["target_y"].reset_current_position(self.np_random.uniform( low=-self.TARG_LIMIT, high=self.TARG_LIMIT ), 0)
        self.fingertip = self.parts["fingertip"]
        self.target = self.parts["target"]
        self.central_joint = self.jdict["joint0"]
        self.shoulder_joint = self.jdict["joint1"]
        self.elbow_joint = self.jdict["joint2"]
        self.central_joint.reset_current_position(self.np_random.uniform( low=-3.14, high=3.14 ), 0)
        self.shoulder_joint.reset_current_position(self.np_random.uniform( low=-3.14, high=3.14 ), 0)
        self.elbow_joint.reset_current_position(self.np_random.uniform( low=-3.14, high=3.14 ), 0)

    def apply_action(self, a):
        assert( np.isfinite(a).all() )
        self.central_joint.set_motor_torque( 0.05*float(np.clip(a[0], -1, +1)) )
        self.shoulder_joint.set_motor_torque( 0.05*float(np.clip(a[1], -1, +1)) )
        self.elbow_joint.set_motor_torque( 0.05*float(np.clip(a[2], -1, +1)) )

    def calc_state(self):
        theta,      self.theta_dot = self.central_joint.current_relative_position()
        self.gamma, self.gamma_dot = self.elbow_joint.current_relative_position()
        self.alpha, self.alpha_dot = self.shoulder_joint.current_relative_position()
        target_x, _ = self.jdict["target_x"].current_position()
        target_y, _ = self.jdict["target_y"].current_position()
        self.to_target_vec = np.array(self.fingertip.pose().xyz()) - np.array(self.target.pose().xyz())
        return np.array([
            target_x,
            target_y,
            self.to_target_vec[0],
            self.to_target_vec[1],
            np.cos(theta),
            np.sin(theta),
            self.theta_dot,
            self.gamma,
            self.gamma_dot,
            self.alpha,
            self.alpha_dot
            ])

    def calc_potential(self):
        return -100 * np.linalg.norm(self.to_target_vec)

    def step(self, a):
        assert(not self.scene.multiplayer)
        self.apply_action(a)
        self.scene.global_step()

        state = self.calc_state()  # sets self.to_target_vec

        potential_old = self.potential
        self.potential = self.calc_potential()

        electricity_cost = (
            -0.10*(np.abs(a[0]*self.theta_dot) + np.abs(a[1]*self.gamma_dot) + np.abs(a[2]*self.alpha_dot))  # work torque*angular_velocity
            -0.01*(np.abs(a[0]) + np.abs(a[1]) + np.abs(a[2])) # stall torque require some energy
            )
        stuck_joint_cost = -1 if np.abs(np.abs(self.gamma)-1) < 0.05 else 0.0
        stuck_joint_cost += -1 if np.abs(np.abs(self.alpha)-1) < 0.05 else 0.0
        self.rewards = [float(self.potential - potential_old), float(electricity_cost), float(stuck_joint_cost)]
        self.frame  += 1
        self.done   += 0
        self.reward += sum(self.rewards)
        self.HUD(state, a, False)
        error = np.sqrt(state[2] ** 2 + state[3] ** 2)
        return state, sum(self.rewards), False, {'error': error}

    def camera_adjust(self):
        x, y, z = self.fingertip.pose().xyz()
        x *= 0.5
        y *= 0.5
        self.camera.move_and_look_at(0.3, 0.3, 0.3, x, y, z)


class Reacher3Base(Reacher3):

    definitionFile = 'reacher3.xml'


class Reacher3Length(Reacher3):

    definitionFile = 'reacher3length.xml'


class Reacher3Joint(Reacher3):

    definitionFile = 'reacher3.xml'

    def apply_action(self, a):
        assert(np.isfinite(a).all())
        self.central_joint.set_motor_torque(0.05*float(np.clip(a[0], -1, +1)))
        self.shoulder_joint.set_motor_torque(0)
        self.elbow_joint.set_motor_torque(0.05*float(np.clip(a[2], -1, +1)))


class Reacher3Motor(Reacher3):

    definitionFile = 'reacher3motor.xml'

    def robot_specific_reset(self):
        self.jdict["target_x"].reset_current_position(self.np_random.uniform( low=-self.TARG_LIMIT, high=self.TARG_LIMIT ), 0)
        self.jdict["target_y"].reset_current_position(self.np_random.uniform( low=-self.TARG_LIMIT, high=self.TARG_LIMIT ), 0)
        self.fingertip = self.parts["fingertip"]
        self.target = self.parts["target"]
        self.central_joint = self.jdict["joint0"]
        self.elbow_joint = self.jdict["joint2"]
        self.central_joint.reset_current_position(self.np_random.uniform( low=-3.14, high=3.14 ), 0)
        self.elbow_joint.reset_current_position(self.np_random.uniform( low=-3.14, high=3.14 ), 0)

    def apply_action(self, a):
        assert( np.isfinite(a).all() )
        self.central_joint.set_motor_torque( 0.05*float(np.clip(a[0], -1, +1)) )
        self.elbow_joint.set_motor_torque( 0.05*float(np.clip(a[2], -1, +1)) )

    def calc_state(self):
        theta,      self.theta_dot = self.central_joint.current_relative_position()
        self.gamma, self.gamma_dot = self.elbow_joint.current_relative_position()
        self.alpha, self.alpha_dot = (0., 0.)
        target_x, _ = self.jdict["target_x"].current_position()
        target_y, _ = self.jdict["target_y"].current_position()
        self.to_target_vec = np.array(self.fingertip.pose().xyz()) - np.array(self.target.pose().xyz())
        return np.array([
            target_x,
            target_y,
            self.to_target_vec[0],
            self.to_target_vec[1],
            np.cos(theta),
            np.sin(theta),
            self.theta_dot,
            self.gamma,
            self.gamma_dot,
            self.alpha,
            self.alpha_dot
            ])
