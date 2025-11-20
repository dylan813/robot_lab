from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
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
