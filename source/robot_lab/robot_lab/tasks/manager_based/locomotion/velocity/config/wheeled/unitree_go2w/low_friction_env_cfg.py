from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg as CurrTerm
import isaaclab.sim as sim_utils

from robot_lab.tasks.manager_based.locomotion.velocity import mdp
from .flat_env_cfg import UnitreeGo2WFlatEnvCfg

@configclass
class UnitreeGo2WLowFrictionEnvCfg(UnitreeGo2WFlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # low friction terrain
        self.scene.terrain.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.1,
            dynamic_friction=0.1,
            restitution=1.0,
        )
        self.sim.physics_material = self.scene.terrain.physics_material

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeGo2WLowFrictionEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class UnitreeGo2WLowFrictionCurriculumEnvCfg(UnitreeGo2WFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=1.0,
        )
        self.sim.physics_material = self.scene.terrain.physics_material

        # curriculum on terrain friction driven by tracking reward
        self.curriculum.terrain_friction = CurrTerm(
            func=mdp.terrain_friction_levels,
            params={
                "reward_term_name": "track_lin_vel_xy_exp",
                "friction_range": (0.1, 1.0),  # target, start
                "step": 0.05,
            },
        )

        if self.__class__.__name__ == "UnitreeGo2WLowFrictionCurriculumEnvCfg":
            self.disable_zero_weight_rewards()
