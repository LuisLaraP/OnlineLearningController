"""
Custom environments for this project.

All the environments provided in this package conform with the OpenAI Gym's
API. This is so they can be interchanged with ease.

This package also includes utilities for creating and managing instances of
environments.

Routines
--------
make
	Create a new instance of the given environment.
"""

import gym.envs

from .launcher import make, register
from .reach_torque import ReachTorque
from .reach_velocity import ReachVelocity

# Mujoco ---------------------------------------------------------------------

gym.envs.register(
	'Reacher-v3',
	entry_point='gym.envs.mujoco:ReacherEnv',
	max_episode_steps=100
)

# Roboschool -----------------------------------------------------------------

gym.envs.register(
	'Reacher2-v0',
	entry_point='olc.environments.reacher2:Reacher2Base',
	max_episode_steps=100
)
gym.envs.register(
	'Reacher2length-v0',
	entry_point='olc.environments.reacher2:Reacher2Length',
	max_episode_steps=100
)
gym.envs.register(
	'Reacher2joint-v0',
	entry_point='olc.environments.reacher2:Reacher2Joint',
	max_episode_steps=100
)

gym.envs.register(
	'Reacher3-v0',
	entry_point='olc.environments.reacher3:Reacher3',
	max_episode_steps=100
)

# Custom ---------------------------------------------------------------------

register('ReachTorque', ReachTorque)
register('ReachVelocity', ReachVelocity)
