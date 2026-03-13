"""Flat-terrain environment config for the accel/brake evaluation experiment."""

from isaaclab.utils import configclass

from .rough_env_cfg import UnitreeGo2WSATARoughEnvCfg


@configclass
class UnitreeGo2WSATAAccelBrakeEnvCfg(UnitreeGo2WSATARoughEnvCfg):
    """Minimal override: flat ground, smaller env count for evaluation."""

    def __post_init__(self):
        super().__post_init__()

        # Replace procedural terrain with a flat plane
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # Smaller default for eval
        self.scene.num_envs = 16
