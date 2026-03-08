"""MDP functions for simple Go2 task.

This sub-module contains the functions that align with genesis-forge's simple example.
It exports observation, action, reward, termination, event, and command functions.
"""

from .events import position
from .rewards import (
    base_height,
    command_tracking_lin_vel,
    command_tracking_ang_vel,
    lin_vel_z_l2,
    action_rate_l2,
    dof_similar_to_default,
)
from .terminations import timeout, bad_orientation
from .observations import (
    angle_velocity,
    linear_velocity,
    projected_gravity,
    dof_position,
    dof_velocity,
    actions,
)
# Import command configuration from locomotion velocity task
from genesis_tasks.locomotion.velocity.mdp.commands import UniformVelocityCommandCfg

__all__ = [
    # Events
    "position",
    # Commands
    "UniformVelocityCommandCfg",
    # Rewards
    "base_height",
    "command_tracking_lin_vel",
    "command_tracking_ang_vel",
    "lin_vel_z_l2",
    "action_rate_l2",
    "dof_similar_to_default",
    # Terminations
    "timeout",
    "bad_orientation",
    # Observations
    "angle_velocity",
    "linear_velocity",
    "projected_gravity",
    "dof_position",
    "dof_velocity",
    "actions",
]
