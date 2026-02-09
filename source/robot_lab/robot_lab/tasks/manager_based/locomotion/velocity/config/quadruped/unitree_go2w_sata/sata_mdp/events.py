"""SATA-specific event functions for growth-scaled pushes and fatigue reset."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def sata_push_growth_scaled(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    max_push_vel_xy: float,
    max_push_vel_ang: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Push robots with velocity scaled by the Gompertz growth scale.

    At early training (growth~0), pushes are very weak.
    At late training (growth~1), pushes reach max velocity.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    growth = getattr(env, "_sata_growth_scale", 0.0)

    max_vel = max_push_vel_xy * growth
    max_ang = max_push_vel_ang * growth

    # Generate random push velocities (only for env_ids)
    velocities = asset.data.root_vel_w[env_ids].clone()
    # Linear velocity push (x, y, z)
    velocities[:, 0:3] = torch.empty(len(env_ids), 3, device=env.device).uniform_(-max_vel, max_vel)
    # Angular velocity push (roll, pitch, yaw)
    velocities[:, 3:6] = torch.empty(len(env_ids), 3, device=env.device).uniform_(-max_ang, max_ang)

    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def sata_reset_fatigue(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
):
    """Reset motor fatigue and activation state on episode reset.

    Initializes fatigue with small random values scaled by growth.
    """
    action_term = env.action_manager.get_term("sata_torque")
    action_term.reset(env_ids)


def sata_dof_pos_termination(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    margin: float = 0.05,
) -> torch.Tensor:
    """Terminate if any joint position exceeds URDF limits plus margin.

    This is stricter than soft limits — uses the hard URDF limits with a small buffer.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]

    # Hard limits from the URDF (not soft limits)
    pos_limits = asset.data.joint_limits[:, asset_cfg.joint_ids]
    lower = pos_limits[:, :, 0] - margin
    upper = pos_limits[:, :, 1] + margin

    exceeded = torch.any(joint_pos > upper, dim=1) | torch.any(joint_pos < lower, dim=1)
    return exceeded


def sata_flipped_termination(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate if robot is flipped (gravity z-component in body frame > 0)."""
    asset = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b[:, 2] > 0
