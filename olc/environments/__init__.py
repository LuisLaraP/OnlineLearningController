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
from .reach import Reach

# Mujoco ---------------------------------------------------------------------

gym.envs.register(
	'Reacher-v3',
	entry_point='gym.envs.mujoco:ReacherEnv',
	max_episode_steps=100
)

# Custom ---------------------------------------------------------------------

register('Reach', Reach)
