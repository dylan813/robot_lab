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


def sata_motor_fatigue(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Returns the motor fatigue state from the SATA torque action term.

    Motor fatigue accumulates based on applied torque magnitude and decays
    each step. Shape: (num_envs, num_joints)
    """
    action_term = env.action_manager.get_term("sata_torque")
    return action_term.motor_fatigue.detach()
