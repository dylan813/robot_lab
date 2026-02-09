"""SATA (Safe and Adaptive Torque-based Locomotion) environment configuration for Unitree Go2W.

This environment implements the SATA framework from Isaac Gym in Isaac Lab's
manager-based architecture. Key features:
- Gompertz growth curriculum for progressive difficulty
- Biologically-inspired muscle activation dynamics
- Hill-type force-velocity muscle model
- Motor fatigue tracking
- Growth-scaled rewards, commands, and domain randomization
"""

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab.envs import mdp as isaaclab_mdp

from .sata_mdp.actions import SATATorqueActionCfg
from .sata_mdp import observations as sata_obs
from .sata_mdp import rewards as sata_rew
from .sata_mdp import events as sata_ev

from robot_lab.assets.unitree import UNITREE_GO2W_SATA_CFG

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

# Ordered list of all actuated joints on Go2W (hips, thighs, calves, wheels).
GO2W_JOINT_NAMES = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FL_foot_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FR_foot_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RL_foot_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RR_foot_joint",
]


##
# Scene definition
##


@configclass
class SATASceneCfg(InteractiveSceneCfg):
    """Scene configuration for SATA with terrain and sensors."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = MISSING

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class SATACommandsCfg:
    """Command configuration for SATA.

    Uses direct angular velocity commands (no heading control).
    Ranges are dynamically updated by the growth curriculum via the action term.
    """

    base_velocity = isaaclab_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=isaaclab_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.5),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.5, 1.5),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class SATAActionsCfg:
    """Action configuration for SATA: direct torque control with muscle model."""

    sata_torque = SATATorqueActionCfg(
        asset_name="robot",
        joint_names=GO2W_JOINT_NAMES,
        preserve_order=True,
        action_scale=5.0,
        activation_process=True,
        activation_ema_alpha=0.6,
        hill_model=True,
        motor_fatigue_enabled=True,
        fatigue_decay=0.9,
        growth_k=0.00003,
        growth_x0=24000.0,
        initial_torque_scale=0.3,
        max_torque_scale=1.0,
        initial_rear_torque_scale=1.0,
        max_rear_torque_scale=1.0,
        start_freq=100.0,
        max_freq=200.0,
        action_loss_rate=0.1,
        vel_x_range=(-0.5, 1.5),
        vel_y_range=(-0.5, 0.5),
        ang_vel_z_range=(-1.5, 1.5),
    )


@configclass
class SATAObservationsCfg:
    """Observation configuration for SATA (60 observations total).

    Order matches the original SATA implementation:
    base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) +
    joint_pos_rel(12) + joint_vel(12) + commands(3) +
    torques(12) + motor_fatigue(12) = 60
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations with noise."""

        base_lin_vel = ObsTerm(
            func=isaaclab_mdp.base_lin_vel,
            noise=Unoise(n_min=-0.15, n_max=0.15),
            clip=(-100.0, 100.0),
            scale=2.0,
        )
        base_ang_vel = ObsTerm(
            func=isaaclab_mdp.base_ang_vel,
            noise=Unoise(n_min=-0.3, n_max=0.3),
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=isaaclab_mdp.projected_gravity,
            noise=Unoise(n_min=-0.3, n_max=0.3),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=isaaclab_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2W_JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-0.015, n_max=0.015),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=isaaclab_mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2W_JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-2.25, n_max=2.25),
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        velocity_commands = ObsTerm(
            func=sata_obs.sata_scaled_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        torques = ObsTerm(
            func=sata_obs.sata_applied_torques,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        motor_fatigue = ObsTerm(
            func=sata_obs.sata_motor_fatigue,
            noise=Unoise(n_min=-0.075, n_max=0.075),
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations without noise."""

        base_lin_vel = ObsTerm(
            func=isaaclab_mdp.base_lin_vel,
            clip=(-100.0, 100.0),
            scale=2.0,
        )
        base_ang_vel = ObsTerm(
            func=isaaclab_mdp.base_ang_vel,
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=isaaclab_mdp.projected_gravity,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=isaaclab_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2W_JOINT_NAMES, preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=isaaclab_mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2W_JOINT_NAMES, preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        velocity_commands = ObsTerm(
            func=sata_obs.sata_scaled_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        torques = ObsTerm(
            func=sata_obs.sata_applied_torques,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        motor_fatigue = ObsTerm(
            func=sata_obs.sata_motor_fatigue,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class SATAEventCfg:
    """Event configuration for SATA domain randomization."""

    # Startup events
    randomize_rigid_body_material = EventTerm(
        func=isaaclab_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.25),
            "dynamic_friction_range": (0.5, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    randomize_rigid_body_mass_base = EventTerm(
        func=isaaclab_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 5.0),
            "operation": "add",
            "recompute_inertia": True,
        },
    )

    randomize_rigid_body_mass_all = EventTerm(
        func=isaaclab_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (-0.0625, 0.3125),
            "operation": "add",
            "recompute_inertia": True,
        },
    )

    randomize_com_positions = EventTerm(
        func=isaaclab_mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.2, 0.2), "y": (-0.1, 0.1), "z": (-0.1, 0.1)},
        },
    )

    # Reset events
    randomize_reset_joints = EventTerm(
        func=isaaclab_mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.95, 1.05),
            "velocity_range": (0.0, 0.0),
        },
    )

    randomize_reset_base = EventTerm(
        func=isaaclab_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_fatigue = EventTerm(
        func=sata_ev.sata_reset_fatigue,
        mode="reset",
        params={},
    )

    # Interval events
    push_robot = EventTerm(
        func=sata_ev.sata_push_growth_scaled,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "max_push_vel_xy": 1.5,
            "max_push_vel_ang": 1.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class SATARewardsCfg:
    """Reward configuration for SATA.

    Reward weights match the original SATA implementation.
    All rewards are multiplied by step_dt internally by Isaac Lab.
    """

    # Positive rewards (tracking)
    forward = RewTerm(
        func=sata_rew.sata_forward,
        weight=10.0,
        params={
            "command_name": "base_velocity",
            "tracking_sigma": 0.25,
            "vel_x_range": (-0.5, 1.5),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    head_height = RewTerm(
        func=sata_rew.sata_head_height,
        weight=5.0,
        params={
            "base_height_target": 0.3,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    moving_y = RewTerm(
        func=sata_rew.sata_moving_y,
        weight=5.0,
        params={
            "command_name": "base_velocity",
            "tracking_sigma": 0.25,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    moving_yaw = RewTerm(
        func=sata_rew.sata_moving_yaw,
        weight=5.0,
        params={
            "command_name": "base_velocity",
            "tracking_sigma": 0.25,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Penalty rewards
    soft_dof_pos_limits = RewTerm(
        func=sata_rew.sata_soft_dof_pos_limits,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )

    motor_fatigue = RewTerm(
        func=sata_rew.sata_motor_fatigue_penalty,
        weight=-0.05,
    )

    dof_acc = RewTerm(
        func=isaaclab_mdp.joint_acc_l2,
        weight=-1e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )

    roll = RewTerm(
        func=sata_rew.sata_roll,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    lin_vel_z = RewTerm(
        func=isaaclab_mdp.lin_vel_z_l2,
        weight=-5.0,
    )


@configclass
class SATATerminationsCfg:
    """Termination configuration for SATA."""

    time_out = DoneTerm(func=isaaclab_mdp.time_out, time_out=True)

    dof_pos_limits = DoneTerm(
        func=sata_ev.sata_dof_pos_termination,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "margin": 0.05},
    )

    robot_flipped = DoneTerm(
        func=sata_ev.sata_flipped_termination,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


##
# Environment configuration
##


@configclass
class UnitreeGo2WSATARoughEnvCfg(ManagerBasedRLEnvCfg):
    """SATA environment configuration for Unitree Go2W on rough terrain."""

    # Scene
    scene: SATASceneCfg = SATASceneCfg(num_envs=4096, env_spacing=2.5)

    # MDP
    observations: SATAObservationsCfg = SATAObservationsCfg()
    actions: SATAActionsCfg = SATAActionsCfg()
    commands: SATACommandsCfg = SATACommandsCfg()
    rewards: SATARewardsCfg = SATARewardsCfg()
    terminations: SATATerminationsCfg = SATATerminationsCfg()
    events: SATAEventCfg = SATAEventCfg()
    curriculum = None  # SATA growth handled via action term

    def __post_init__(self):
        """Post initialization."""
        # Simulation settings
        self.decimation = 2
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # Set robot asset
        self.scene.robot = UNITREE_GO2W_SATA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        # Sensor update periods
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # Disable terrain curriculum (SATA uses its own growth)
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False
