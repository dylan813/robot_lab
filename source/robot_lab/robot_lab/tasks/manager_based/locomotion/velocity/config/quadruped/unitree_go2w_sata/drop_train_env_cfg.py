"""Flat-terrain fine-tuning config for drop recovery."""

import torch
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from isaaclab.envs import mdp as isaaclab_mdp
from isaaclab.managers import SceneEntityCfg

from .rough_env_cfg import SATAActionsCfg, SATARewardsCfg, UnitreeGo2WSATARoughEnvCfg
from .sata_mdp import rewards as sata_rew
from .sata_mdp.actions import SATATorqueActionCfg
from .sata_mdp.curriculums import DropHeightCurriculum


def _zero_command(env):
    """Zero velocity command — robot must survive the fall and stand still."""
    return torch.zeros(env.num_envs, 3, device=env.device)


@configclass
class DropActionsCfg(SATAActionsCfg):
    """Actions cfg with growth pre-seeded to end-of-base-training scale and frozen."""

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
        scale_command_ranges=False,
    )


@configclass
class DropCurriculumCfg:
    """Curriculum that progressively increases drop height."""

    drop_height_stage = CurrTerm(func=DropHeightCurriculum)


@configclass
class DropRewardsCfg(SATARewardsCfg):
    """Rewards emphasizing stability and impact absorption over locomotion."""

    # Stronger roll penalty — upright recovery is the primary goal
    roll: RewTerm = RewTerm(
        func=sata_rew.sata_roll,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Stronger vertical velocity penalty — damp the bounce on landing
    lin_vel_z: RewTerm = RewTerm(
        func=isaaclab_mdp.lin_vel_z_l2,
        weight=-10.0,
    )


@configclass
class UnitreeGo2WSATADropTrainEnvCfg(UnitreeGo2WSATARoughEnvCfg):
    """Fine-tuning env for drop recovery: flat terrain, progressive drop height curriculum."""

    actions: DropActionsCfg = DropActionsCfg()
    curriculum: DropCurriculumCfg = DropCurriculumCfg()
    rewards: DropRewardsCfg = DropRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Flat terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        self.scene.num_envs = 64

        # Zero velocity command — policy only needs to survive and stabilize
        self.observations.policy.velocity_commands = ObsTerm(func=_zero_command)

        # Initial drop height: 0.5 m (Stage 0). Curriculum advances this.
        # z_offset = 0.5 - 0.45 (default spawn z) = 0.05
        self.events.randomize_reset_base.params["pose_range"]["z"] = (0.05, 0.05)

        # No push disturbances during drop recovery training
        if hasattr(self.events, "push_robot"):
            self.events.push_robot = None

        # Short episodes: drop + stabilization window (~8 s)
        self.episode_length_s = 8.0
