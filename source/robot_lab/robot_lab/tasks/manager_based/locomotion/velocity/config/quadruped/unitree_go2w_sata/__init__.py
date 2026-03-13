import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-SATA-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo2WSATARoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2WSATARoughPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Braking-Train-Unitree-Go2W-SATA-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.braking_train_env_cfg:UnitreeGo2WSATABrakingTrainEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.braking_train_ppo_cfg:UnitreeGo2WSATABrakingPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Drop-Train-Unitree-Go2W-SATA-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drop_train_env_cfg:UnitreeGo2WSATADropTrainEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.drop_train_ppo_cfg:UnitreeGo2WSATADropPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-StepDrop-Train-Unitree-Go2W-SATA-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.step_drop_train_env_cfg:UnitreeGo2WSATAStepDropTrainEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.step_drop_train_ppo_cfg:UnitreeGo2WSATAStepDropPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-AccelBrake-Unitree-Go2W-SATA-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.accel_brake_env_cfg:UnitreeGo2WSATAAccelBrakeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2WSATARoughPPORunnerCfg",
    },
)
