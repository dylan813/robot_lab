# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

# reuse base terrain curriculum
from isaaclab_tasks.manager_based.locomotion.velocity.mdp.curriculums import terrain_levels_vel

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def command_levels_lin_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_lin_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        env._original_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
        env._initial_vel_x = env._original_vel_x * range_multiplier[0]
        env._final_vel_x = env._original_vel_x * range_multiplier[1]
        env._initial_vel_y = env._original_vel_y * range_multiplier[0]
        env._final_vel_y = env._original_vel_y * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()
        base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device) + delta_command
            new_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
            new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])

            # Update ranges
            base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
            base_velocity_ranges.lin_vel_y = new_vel_y.tolist()

    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)


def command_levels_ang_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_ang_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original angular velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device)
        env._initial_ang_vel_z = env._original_ang_vel_z * range_multiplier[0]
        env._final_ang_vel_z = env._original_ang_vel_z * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.ang_vel_z = env._initial_ang_vel_z.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_ang_vel_z = torch.clamp(new_ang_vel_z, min=env._final_ang_vel_z[0], max=env._final_ang_vel_z[1])

            # Update ranges
            base_velocity_ranges.ang_vel_z = new_ang_vel_z.tolist()

    return torch.tensor(base_velocity_ranges.ang_vel_z[1], device=env.device)


def terrain_friction_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy_exp",
    friction_range: Sequence[float] = (0.1, 1.0),
    step: float = 0.05,
) -> torch.Tensor:
    """decrease friction as tracking reward stays high."""
    if env.common_step_counter == 0:
        env._terrain_friction_min = torch.tensor(friction_range[0], device=env.device)
        env._terrain_friction_max = torch.tensor(friction_range[1], device=env.device)
        env._terrain_friction_curr = env._terrain_friction_max.clone()
        _set_terrain_friction(env, env._terrain_friction_curr)

    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        avg_rew = torch.mean(episode_sums[env_ids]) / env.max_episode_length_s

        if avg_rew > 0.8 * reward_term_cfg.weight:
            env._terrain_friction_curr = torch.clamp(
                env._terrain_friction_curr - step, min=env._terrain_friction_min
            )
            _set_terrain_friction(env, env._terrain_friction_curr)

    return env._terrain_friction_curr


def _set_terrain_friction(env: ManagerBasedRLEnv, friction: torch.Tensor | float) -> None:
    friction_val = float(friction)

    if getattr(env.scene, "terrain", None) is not None and getattr(env.scene.terrain, "physics_material", None):
        env.scene.terrain.physics_material.static_friction = friction_val
        env.scene.terrain.physics_material.dynamic_friction = friction_val

    if getattr(env, "sim", None) is not None and getattr(env.sim, "physics_material", None):
        env.sim.physics_material.static_friction = friction_val
        env.sim.physics_material.dynamic_friction = friction_val

    terrain_obj = getattr(env.scene.terrain, "object", None)
    if terrain_obj is not None and hasattr(terrain_obj, "root_physx_view"):
        root_view = terrain_obj.root_physx_view
        if hasattr(root_view, "set_dynamics_friction"):
            root_view.set_dynamics_friction(friction_val)
        if hasattr(root_view, "set_statics_friction"):
            root_view.set_statics_friction(friction_val)


def terrain_levels_and_friction_two_stage(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    friction_range: Sequence[float] = (0.1, 1.0),
    friction_step: float = 0.05,
    convergence_delta: float = 0.1,
    convergence_patience: int = 5,
) -> torch.Tensor:
    """Two-stage curriculum:
        Stage 1 (terrain): climb terrain levels until convergence
        Stage 2 (friction): keep climbing terrain while friction is lowered by `friction_step` at each convergence event until reaching `friction_range[0]`
    """
    terrain = env.scene.terrain
    device = env.device
    max_level_idx = getattr(terrain, "max_terrain_level", 1) - 1

    if env.common_step_counter == 0:
        env._tf_phase = "terrain"
        env._tf_level_streak = 0
        env._tf_fric_min = torch.tensor(friction_range[0], device=device)
        env._tf_fric_max = torch.tensor(friction_range[1], device=device)
        env._tf_fric_curr = env._tf_fric_max.clone()
        _set_terrain_friction(env, env._tf_fric_curr)

    #clamped, no random reset at max
    _terrain_levels_vel_clamped(env, env_ids)

    if getattr(terrain, "terrain_levels", None) is not None:
        mean_level = float(torch.mean(terrain.terrain_levels.float()))
        log = env.extras.get("log", {})
        log["Curriculum/terrain_mean_level"] = mean_level
        log["Curriculum/terrain_max_level"] = float(max_level_idx)
        log["Curriculum/convergence_delta"] = float(convergence_delta)
        log["Curriculum/convergence_patience"] = float(convergence_patience)
        log["Curriculum/friction_step_param"] = float(friction_step)
        log["Curriculum/friction_min_param"] = float(friction_range[0])
        log["Curriculum/friction_max_param"] = float(friction_range[1])
        env.extras["log"] = log

    if env.common_step_counter % env.max_episode_length != 0:
        return env._tf_fric_curr

    mean_level = float(torch.mean(terrain.terrain_levels.float()))

    #convergence check
    prev_mean = getattr(env, "_tf_prev_mean_level", mean_level)
    env._tf_prev_mean_level = mean_level
    if not hasattr(env, "_tf_converge_streak"):
        env._tf_converge_streak = 0
    env._tf_converge_streak = env._tf_converge_streak + 1 if (mean_level - prev_mean) < convergence_delta else 0
    convergence_hit = env._tf_converge_streak >= convergence_patience

    if env._tf_phase == "terrain":
        if convergence_hit:
            env._tf_phase = "friction"
            env._tf_converge_streak = 0
            env._tf_fric_curr = env._tf_fric_max.clone()
            _set_terrain_friction(env, env._tf_fric_curr)

    elif env._tf_phase == "friction":
        if convergence_hit and env._tf_fric_curr > env._tf_fric_min:
            env._tf_converge_streak = 0
            env._tf_fric_curr = torch.clamp(env._tf_fric_curr - friction_step, min=env._tf_fric_min)
            _set_terrain_friction(env, env._tf_fric_curr)

    log = env.extras.get("log", {})
    log["Curriculum/friction"] = float(env._tf_fric_curr)
    env.extras["log"] = log
    return env._tf_fric_curr


def _terrain_levels_vel_clamped(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    """no random reset when max level is reached"""
    terrain = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    max_level_idx = getattr(terrain, "max_terrain_level", 1) - 1

    if "max_level_cap" in env.extras:
        max_level_idx = min(max_level_idx, int(env.extras["max_level_cap"]))

    asset = env.scene["robot"]
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down &= ~move_up

    terrain.terrain_levels[env_ids] = torch.clamp(
        terrain.terrain_levels[env_ids] + 1 * move_up - 1 * move_down, min=0, max=max_level_idx
    )
    terrain.env_origins[env_ids] = terrain.terrain_origins[terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]]
    return torch.mean(terrain.terrain_levels.float())
