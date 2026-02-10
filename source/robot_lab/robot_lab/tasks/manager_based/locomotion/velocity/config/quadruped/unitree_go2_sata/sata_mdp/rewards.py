"""SATA-specific reward functions with Gompertz growth curriculum awareness."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def sata_forward(
    env: ManagerBasedRLEnv,
    command_name: str,
    tracking_sigma: float,
    vel_x_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Forward velocity tracking reward."""
    asset: RigidObject = env.scene[asset_cfg.name]
    growth = getattr(env, "_sata_growth_scale", 0.0)
    cmd = env.command_manager.get_command(command_name)

    mid_vel = (vel_x_range[1] + vel_x_range[0]) / 2.0
    target = mid_vel * max(1.0 - growth * 2.0, 0.0) + cmd[:, 0] * min(growth * 2.0, 1.0)

    return torch.exp(-torch.abs(asset.data.root_lin_vel_b[:, 0] - target) / tracking_sigma)


def sata_head_height(
    env: ManagerBasedRLEnv,
    base_height_target: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Head height reward: base height (terrain-relative) plus upright penalty (encourages the robot to stand tall and keep its head upright)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    growth = getattr(env, "_sata_growth_scale", 0.0)

    # Terrain-relative base height via height scanner
    sensor: RayCaster = env.scene[sensor_cfg.name]
    ray_hits = sensor.data.ray_hits_w[..., 2]

    if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
        base_height = torch.zeros(env.num_envs, device=env.device)
    else:
        base_height = torch.mean(
            asset.data.root_pos_w[:, 2].unsqueeze(1) - ray_hits,
            dim=1,
        ).clamp(max=base_height_target)

    # Head-up penalty based on forward component of projected gravity
    gravity_x = asset.data.projected_gravity_b[:, 0]
    clip_min = min(0.0, -0.2 * (1.5 - growth * 2.0))
    head_up = -gravity_x.clamp(min=clip_min)

    return base_height * (1.0 + growth) + head_up


def sata_moving_y(
    env: ManagerBasedRLEnv,
    command_name: str,
    tracking_sigma: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Lateral velocity tracking reward."""
    asset: RigidObject = env.scene[asset_cfg.name]
    growth = getattr(env, "_sata_growth_scale", 0.0)
    cmd = env.command_manager.get_command(command_name)

    return torch.exp(-torch.abs(asset.data.root_lin_vel_b[:, 1] - cmd[:, 1]) / tracking_sigma) * growth


def sata_moving_yaw(
    env: ManagerBasedRLEnv,
    command_name: str,
    tracking_sigma: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Yaw angular velocity tracking reward."""
    asset: RigidObject = env.scene[asset_cfg.name]
    growth = getattr(env, "_sata_growth_scale", 0.0)
    cmd = env.command_manager.get_command(command_name)

    return torch.exp(-torch.abs(asset.data.root_ang_vel_b[:, 2] - cmd[:, 2]) / tracking_sigma) * growth


def sata_soft_dof_pos_limits(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint positions approaching soft limits (computes the total excess beyond the soft joint position limits)."""
    asset = env.scene[asset_cfg.name]
    # soft limits are stored on the asset
    soft_lower = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    soft_upper = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]

    out_of_limits = -(joint_pos - soft_lower).clamp(max=0.0)
    out_of_limits += (joint_pos - soft_upper).clamp(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def sata_motor_fatigue_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize high torque output on fatigued motors (returns sum(fatigue * |torque_action|) per environment)."""
    action_term = env.action_manager.get_term("sata_torque")
    return torch.sum(action_term.motor_fatigue * torch.abs(action_term.torques_action), dim=1)


def sata_roll(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize body roll via the y-component of projected gravity."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.abs(asset.data.projected_gravity_b[:, 1])
