"""Flat-terrain fine-tuning config for step-drop (drive-off-ledge) skill."""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from .drop_train_env_cfg import DropActionsCfg
from .rough_env_cfg import SATARewardsCfg, UnitreeGo2WSATARoughEnvCfg
from .sata_mdp.curriculums import StepDropCurriculum


# Block geometry — must match StepDropCurriculum constants in curriculums.py
_BLOCK_LENGTH = 3.0   # m, robot has 2.5 m run-up to the front edge
_BLOCK_WIDTH  = 2.0   # m
_BLOCK_HEIGHT = 1.0   # m tall (partially underground at lower stages)
_BLOCK_HALF_H = 0.5

# Stage 0 initial values
_STAGE0_TOP    = 0.3                          # block top surface height (m)
_STAGE0_VX     = (0.5, 1.0)                   # forward speed range (m/s)
_SPAWN_X       = -2.5                         # 2.5 m behind front edge
_SPAWN_Z_CLEAR = 0.05                         # clearance above block surface


@configclass
class StepDropCurriculumCfg:
    """Curriculum that raises the block and increases speed together."""

    step_drop_stage = CurrTerm(func=StepDropCurriculum)


@configclass
class StepDropRewardsCfg(SATARewardsCfg):
    """Rewards for step-drop training — inherits base SATARewardsCfg unchanged."""
    pass


@configclass
class UnitreeGo2WSATAStepDropTrainEnvCfg(UnitreeGo2WSATARoughEnvCfg):
    """Fine-tuning env for step-drop skill.

    A physical kinematic block sits in each environment.  The robot spawns
    near the back of the block with zero initial velocity, settles, then
    walks forward under a sustained velocity command and drives off the
    front edge.  The curriculum raises the block height and increases the
    commanded speed together across four stages (0.3 m → 1.0 m,
    0.5–1.0 m/s → 1.5–2.5 m/s).
    """

    actions: DropActionsCfg = DropActionsCfg()
    curriculum: StepDropCurriculumCfg = StepDropCurriculumCfg()
    rewards: StepDropRewardsCfg = StepDropRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Flat terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        self.scene.num_envs = 64

        # Env spacing must accommodate the 3 m block with clearance on each side.
        self.scene.env_spacing = max(self.scene.env_spacing, 8.0)

        # No push disturbances — landing dynamics are the challenge
        if hasattr(self.events, "push_robot"):
            self.events.push_robot = None

        # ---------------------------------------------------------------
        # Static kinematic platform.
        # Block spans x = [-_BLOCK_LENGTH, 0] (front edge at x=0).
        # Initial center z places top surface at _STAGE0_TOP.
        # The curriculum moves the block up each stage by writing root state.
        # ---------------------------------------------------------------
        self.scene.platform = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Platform",
            spawn=sim_utils.CuboidCfg(
                size=(_BLOCK_LENGTH, _BLOCK_WIDTH, _BLOCK_HEIGHT),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1e6),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.35, 0.3)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(-_BLOCK_LENGTH / 2.0, 0.0, _STAGE0_TOP - _BLOCK_HALF_H),
            ),
        )

        # ---------------------------------------------------------------
        # Robot spawns near the back of the block, zero initial velocity.
        # pose_range adds offset to default root state (0, 0, 0.45).
        #   spawn_z = _STAGE0_TOP + _SPAWN_Z_CLEAR
        #   robot COM = 0.45 + spawn_z = _STAGE0_TOP + 0.45 + 0.05
        # The curriculum updates pose_range["z"] each stage.
        # ---------------------------------------------------------------
        spawn_z = _STAGE0_TOP + _SPAWN_Z_CLEAR
        self.events.randomize_reset_base.params["pose_range"]["x"] = (_SPAWN_X, _SPAWN_X)
        self.events.randomize_reset_base.params["pose_range"]["y"] = (0.0, 0.0)
        self.events.randomize_reset_base.params["pose_range"]["z"] = (spawn_z, spawn_z)
        self.events.randomize_reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
        self.events.randomize_reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }

        # Velocity command — stage 0 range; curriculum updates this each stage.
        self.commands.base_velocity.ranges.lin_vel_x = _STAGE0_VX
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (4.0, 6.0)

        # Long episodes: robot walks 2.5 m to the edge + fall + recovery + walking.
        self.episode_length_s = 15.0
