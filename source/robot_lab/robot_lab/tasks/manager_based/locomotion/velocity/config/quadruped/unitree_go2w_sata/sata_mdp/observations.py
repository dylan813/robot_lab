"""SATA-specific observation functions for motor fatigue and applied torques."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def sata_applied_torques(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Returns the applied torques from the SATA torque action term.

    Shape: (num_envs, num_joints)
    """
    action_term = env.action_manager.get_term("sata_torque")
    return action_term.processed_actions


def sata_scaled_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Returns velocity commands scaled by [2.0, 2.0, 0.25].

    Matches the original SATA observation scaling:
    - lin_vel_x * 2.0 (obs_scales.lin_vel)
    - lin_vel_y * 2.0
    - ang_vel_z * 0.25 (obs_scales.ang_vel)

    Shape: (num_envs, 3)
    """
    cmd = env.command_manager.get_command(command_name)
    scaled = cmd[:, :3].clone()
    scaled[:, 0] *= 2.0
    scaled[:, 1] *= 2.0
    scaled[:, 2] *= 0.25
    return scaled


def sata_motor_fatigue(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Returns per-joint motor state: fatigue for leg joints, normalized power for wheel joints.

    Leg joints: leaky-integral of |torque| * dt (fatigue proxy).
    Wheel joints: |T * omega| / (T_max * omega_max) — instantaneous normalized power,
    replacing the previously dead zero slots caused by wheel fatigue masking.

    Shape: (num_envs, num_joints)
    """
    action_term = env.action_manager.get_term("sata_torque")
    state = action_term.motor_fatigue.detach().clone()

    # Fill wheel slots with normalized instantaneous power: |T * omega| / (T_max * omega_max)
    torques = action_term.processed_actions
    joint_vel = action_term._asset.data.joint_vel[:, action_term._joint_ids]
    wheel_mask = action_term._wheel_joint_mask
    power_norm = (action_term.base_torque_limits[wheel_mask] * action_term.vel_limits[wheel_mask])
    state[:, wheel_mask] = (
        torch.abs(torques[:, wheel_mask] * joint_vel[:, wheel_mask]) / power_norm
    )

    return state
