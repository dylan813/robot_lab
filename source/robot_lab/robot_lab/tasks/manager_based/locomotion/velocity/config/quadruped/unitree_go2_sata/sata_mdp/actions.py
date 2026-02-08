"""Custom SATA torque action term with muscle activation dynamics, Hill model, and motor fatigue."""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class SATATorqueAction(ActionTerm):
    """SATA torque action term implementing biologically-inspired motor control.

    This action term implements the full SATA torque pipeline:
    1. Raw action scaling
    2. Gompertz growth-based torque limit scheduling
    3. Muscle activation dynamics (tanh saturation + EMA smoothing)
    4. Activation dropout (simulating sensory/motor loss)
    5. Hill-type muscle model (velocity-dependent force generation)
    6. Motor fatigue tracking and decay
    """

    cfg: SATATorqueActionCfg
    _asset: Articulation

    def __init__(self, cfg: SATATorqueActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        # Resolve joints
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        omni.log.info(
            f"Resolved joint names for {self.__class__.__name__}: {self._joint_names} [{self._joint_ids}]"
        )

        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints and not self.cfg.preserve_order:
            self._joint_ids = slice(None)

        # Create action buffers
        self._raw_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)

        # SATA-specific buffers
        self.torques_action = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self.activation_sign = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self.motor_fatigue = torch.zeros(self.num_envs, self._num_joints, device=self.device)

        # Torque and velocity limits from actuator config
        self.base_torque_limits = torch.ones(self._num_joints, device=self.device) * 23.5
        self.vel_limits = torch.ones(self._num_joints, device=self.device) * 30.0

        # Try to read limits from actuator model
        for actuator in self._asset.actuators.values():
            if hasattr(actuator, 'effort_limit'):
                effort = actuator.effort_limit
                if isinstance(effort, torch.Tensor):
                    # effort_limit may be 2D [num_envs, num_joints] — take first row
                    if effort.dim() >= 2:
                        self.base_torque_limits = effort[0, :self._num_joints].to(self.device)
                    else:
                        self.base_torque_limits = effort[:self._num_joints].to(self.device)
                else:
                    self.base_torque_limits[:] = float(effort)
            if hasattr(actuator, 'velocity_limit'):
                vel = actuator.velocity_limit
                if isinstance(vel, torch.Tensor):
                    if vel.dim() >= 2:
                        self.vel_limits = vel[0, :self._num_joints].to(self.device)
                    else:
                        self.vel_limits = vel[:self._num_joints].to(self.device)
                else:
                    self.vel_limits[:] = float(vel)
            break  # Use the first actuator group

        # Growth state
        self._growth_scale: float = 0.0
        self._physics_step_counter: int = 0
        self.current_torque_scale: float = self.cfg.initial_torque_scale
        self.rear_torque_scale: float = self.cfg.initial_rear_torque_scale

        # Frequency growth: track substep index within each env step
        self._substep_idx: int = 0

        # Observation dropout (lazy-installed after obs manager exists)
        self._prev_policy_obs: torch.Tensor | None = None
        self._obs_dropout_installed: bool = False

        # Identify rear leg joint indices by name
        self._rear_joint_mask = torch.zeros(self._num_joints, dtype=torch.bool, device=self.device)
        for i, name in enumerate(self._joint_names):
            if name.startswith("RL_") or name.startswith("RR_"):
                self._rear_joint_mask[i] = True

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def growth_scale(self) -> float:
        return self._growth_scale

    def _install_obs_dropout(self):
        """Wrap observation_manager.compute() to apply per-env observation dropout.

        Matches original SATA (go2_torque.py:300-302) where entire observation
        vectors are randomly replaced with the previous step's observation.
        """
        orig_compute = self._env.observation_manager.compute
        action_term = self

        def _compute_with_dropout(*args, **kwargs):
            obs_dict = orig_compute(*args, **kwargs)
            if action_term.cfg.action_loss_rate > 0 and "policy" in obs_dict:
                if action_term._prev_policy_obs is not None:
                    keep_mask = (
                        torch.rand(action_term.num_envs, device=action_term.device).unsqueeze(1)
                        > action_term.cfg.action_loss_rate
                    )
                    obs_dict["policy"] = torch.where(
                        keep_mask, obs_dict["policy"], action_term._prev_policy_obs
                    )
                action_term._prev_policy_obs = obs_dict["policy"].clone()
            return obs_dict

        self._env.observation_manager.compute = _compute_with_dropout

    def process_actions(self, actions: torch.Tensor):
        """Process raw actions.

        Called once per environment step before the physics simulation loop.
        Growth computation is done in apply_actions() to match the original SATA
        where growth updates every sim substep.
        """
        # Lazy-install observation dropout (obs manager doesn't exist during __init__)
        if not self._obs_dropout_installed:
            self._install_obs_dropout()
            self._obs_dropout_installed = True

        self._raw_actions[:] = actions
        self._substep_idx = 0

    def apply_actions(self):
        """Apply the SATA torque pipeline and set joint effort targets.

        Called every simulation substep. Growth is computed here to match
        the original SATA where the step counter increments per sim substep.
        """
        # Update physics step counter and growth (always, matching original)
        self._physics_step_counter += 1
        self._substep_idx += 1
        self._growth_scale = math.exp(
            -math.exp(-self.cfg.growth_k * (self._physics_step_counter - self.cfg.growth_x0))
        )

        # Store on env for access by reward/observation/event terms
        self._env._sata_growth_scale = self._growth_scale

        # Frequency growth: compute how many substeps should run this env step
        current_freq = (
            self._growth_scale * (self.cfg.max_freq - self.cfg.start_freq)
            + self.cfg.start_freq
        )
        sim_dt = self._env.sim.cfg.dt
        substeps_needed = max(1, round(1.0 / (sim_dt * current_freq)))

        # If past the needed substeps, just re-apply previous torques and return
        if self._substep_idx > substeps_needed:
            self._asset.set_joint_effort_target(self._processed_actions, joint_ids=self._joint_ids)
            return

        # Update torque scales based on growth
        self.current_torque_scale = (
            self._growth_scale * (self.cfg.max_torque_scale - self.cfg.initial_torque_scale)
            + self.cfg.initial_torque_scale
        )
        self.rear_torque_scale = (
            self._growth_scale * (self.cfg.max_rear_torque_scale - self.cfg.initial_rear_torque_scale)
            + self.cfg.initial_rear_torque_scale
        )

        # Update command ranges based on growth
        try:
            cmd_term = self._env.command_manager.get_term("base_velocity")
            ranges = cmd_term.cfg.ranges

            x_sum = self.cfg.vel_x_range[1] + self.cfg.vel_x_range[0]
            x_diff = self.cfg.vel_x_range[1] - self.cfg.vel_x_range[0]
            ranges.lin_vel_x = (
                max(x_sum * 0.5 - x_diff * self._growth_scale, self.cfg.vel_x_range[0]),
                min(x_sum * 0.5 + x_diff * self._growth_scale, self.cfg.vel_x_range[1]),
            )
            ranges.lin_vel_y = (
                self.cfg.vel_y_range[0] * self._growth_scale,
                self.cfg.vel_y_range[1] * self._growth_scale,
            )
            ranges.ang_vel_z = (
                self.cfg.ang_vel_z_range[0] * self._growth_scale,
                self.cfg.ang_vel_z_range[1] * self._growth_scale,
            )
        except Exception:
            pass  # Command manager may not be ready during init

        # Step 1: Scale raw actions
        self.torques_action = self._raw_actions * self.cfg.action_scale

        # Step 2: Compute growth-scaled torque limits
        torque_limits_scaled = self.base_torque_limits * self.current_torque_scale
        torque_limits_scaled = torque_limits_scaled.unsqueeze(0).expand(self.num_envs, -1).clone()
        # Scale rear legs separately
        torque_limits_scaled[:, self._rear_joint_mask] *= self.rear_torque_scale

        # Step 3: Muscle activation dynamics
        if self.cfg.activation_process:
            current_activation = torch.tanh(self.torques_action / torque_limits_scaled)
            new_activation = (
                (current_activation - self.activation_sign) * self.cfg.activation_ema_alpha
                + self.activation_sign
            )
        else:
            new_activation = self.torques_action / torque_limits_scaled

        # Step 4: Activation dropout (action loss)
        if self.cfg.action_loss_rate > 0:
            keep_mask = torch.rand(self.num_envs, device=self.device).unsqueeze(1) > self.cfg.action_loss_rate
            self.activation_sign = torch.where(keep_mask, new_activation, self.activation_sign)
        else:
            self.activation_sign = new_activation

        # Step 5: Hill muscle model (velocity-dependent force)
        if self.cfg.hill_model:
            dof_vel = self._asset.data.joint_vel[:, self._joint_ids]
            vel_limits_expanded = self.vel_limits.unsqueeze(0).expand(self.num_envs, -1)
            torques = self.activation_sign * torque_limits_scaled * (
                1 - torch.sign(self.activation_sign) * dof_vel / vel_limits_expanded
            )
        else:
            torques = self.activation_sign * torque_limits_scaled

        # Step 6: Motor fatigue tracking
        if self.cfg.motor_fatigue_enabled:
            sim_dt = self._env.sim.cfg.dt
            self.motor_fatigue = self.motor_fatigue + torch.abs(torques) * sim_dt
            self.motor_fatigue = self.motor_fatigue * self.cfg.fatigue_decay
        else:
            self.motor_fatigue.zero_()

        # Store processed actions (final torques)
        self._processed_actions = torques

        # Apply torques to the asset
        self._asset.set_joint_effort_target(torques, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset action state for specified environments."""
        if env_ids is None:
            env_ids = slice(None)

        self._raw_actions[env_ids] = 0.0
        self.activation_sign[env_ids] = 0.0

        # Reset fatigue with small random initial values scaled by growth
        if self.cfg.motor_fatigue_enabled and self._growth_scale > 0:
            self.motor_fatigue[env_ids] = torch.rand_like(self.motor_fatigue[env_ids]) * 0.2 * self._growth_scale
        else:
            self.motor_fatigue[env_ids] = 0.0


@configclass
class SATATorqueActionCfg(ActionTermCfg):
    """Configuration for the SATA torque action term."""

    class_type: type[ActionTerm] = SATATorqueAction

    # Joint selection
    joint_names: list[str] = MISSING
    preserve_order: bool = True

    # Action scaling
    action_scale: float = 5.0

    # Muscle model parameters
    activation_process: bool = True
    activation_ema_alpha: float = 0.6
    hill_model: bool = True

    # Motor fatigue
    motor_fatigue_enabled: bool = True
    fatigue_decay: float = 0.9

    # Growth parameters (Gompertz curve)
    growth_k: float = 0.00003
    growth_x0: float = 24000.0
    initial_torque_scale: float = 0.3
    max_torque_scale: float = 1.0
    initial_rear_torque_scale: float = 1.0
    max_rear_torque_scale: float = 1.0

    # Frequency growth
    start_freq: float = 100.0
    max_freq: float = 200.0

    # Domain randomization
    action_loss_rate: float = 0.1

    # Command range parameters (for growth-based scaling)
    vel_x_range: tuple[float, float] = (-0.5, 1.5)
    vel_y_range: tuple[float, float] = (-0.5, 0.5)
    ang_vel_z_range: tuple[float, float] = (-1.5, 1.5)
