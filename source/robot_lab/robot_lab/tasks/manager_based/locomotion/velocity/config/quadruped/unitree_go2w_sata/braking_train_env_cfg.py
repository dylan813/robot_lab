"""Flat-terrain fine-tuning config for braking skill acquisition."""

import math
from collections.abc import Sequence

from isaaclab.envs import mdp as isaaclab_mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from .rough_env_cfg import SATAActionsCfg, UnitreeGo2WSATARoughEnvCfg
from .sata_mdp.actions import SATATorqueActionCfg
from .sata_mdp.curriculums import BrakingVelocityCurriculum


class SettlingVelocityCommand(isaaclab_mdp.UniformVelocityCommand):
    """UniformVelocityCommand that forces vx=0 at every episode reset.

    After the reset, normal resampling takes over. This gives the robot
    time to land and stabilize before receiving any movement command.
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        metrics = super().reset(env_ids)
        if env_ids is not None and len(env_ids) > 0:
            self.vel_command_b[env_ids] = 0.0
        return metrics


@configclass
class SettlingVelocityCommandCfg(isaaclab_mdp.UniformVelocityCommandCfg):
    class_type: type = SettlingVelocityCommand


@configclass
class BrakingActionsCfg(SATAActionsCfg):
    """Actions cfg with growth pre-seeded to end-of-base-training scale."""

    sata_torque: SATATorqueActionCfg = SATATorqueActionCfg(
        asset_name="robot",
        joint_names=[
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FL_foot_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FR_foot_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "RL_foot_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", "RR_foot_joint",
        ],
        preserve_order=True,
        action_scale=5.0,
        activation_process=True,
        activation_ema_alpha=0.6,
        hill_model=True,
        motor_fatigue_enabled=True,
        fatigue_decay=0.9,
        growth_k=0.00003,
        growth_x0=24000.0,
        # Pre-seed to end of base training: 3000 iters * 24 steps = 72000
        # Growth at step 72000: exp(-exp(-0.00003*(72000-24000))) ~= 0.79
        growth_initial_steps=72000,
        freeze_growth=True,
        initial_torque_scale=0.3,
        max_torque_scale=1.0,
        scale_command_ranges=False,  # curriculum handles command ranges
    )


@configclass
class BrakingCurriculumCfg:
    """Curriculum that progressively exposes the robot to faster speeds then braking."""

    velocity_stage = CurrTerm(func=BrakingVelocityCurriculum)


@configclass
class UnitreeGo2WSATABrakingTrainEnvCfg(UnitreeGo2WSATARoughEnvCfg):
    """Fine-tuning env for braking skill: flat terrain, 3-stage velocity curriculum."""

    actions: BrakingActionsCfg = BrakingActionsCfg()
    curriculum: BrakingCurriculumCfg = BrakingCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # Flat terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # More envs for fine-tuning sample efficiency
        self.scene.num_envs = 64

        # SettlingVelocityCommand forces vx=0 on every episode reset, then
        # resamples normally after resampling_time_range seconds. This gives
        # the robot time to land and stabilize before any movement command.
        self.commands.base_velocity = SettlingVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(5.0, 8.0),
            rel_standing_envs=0.1,
            rel_heading_envs=0.0,
            heading_command=False,
            debug_vis=False,
            ranges=isaaclab_mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(0.3, 1.0),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
                heading=(-math.pi, math.pi),
            ),
        )

        self.episode_length_s = 20.0
