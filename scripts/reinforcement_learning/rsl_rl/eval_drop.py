"""Drop test evaluation script for Unitree Go2W SATA.

Spawns robots at a configurable height above flat ground with zero velocity
command. Measures whether each robot can survive the fall and stabilize.
Logs height, vertical velocity, pitch, roll, and torque each step.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Drop test evaluation for Go2W SATA.")
parser.add_argument("--num_envs", type=int, default=10, help="Number of robots to drop.")
parser.add_argument("--task", type=str, default="RobotLab-Isaac-AccelBrake-Unitree-Go2W-SATA-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--drop_height", type=float, default=1.5, help="Drop height above ground (m).")
parser.add_argument("--eval_steps", type=int, default=500, help="Steps to observe after drop (~5 s at 100 Hz).")
parser.add_argument("--save_csv", type=str, default=None, help="Optional path to save metrics CSV.")
parser.add_argument("--video", action="store_true", default=False, help="Record video of the evaluation.")
parser.add_argument("--video_length", type=int, default=0, help="Video length in steps (0 = full run).")
parser.add_argument("--real-time", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401  # isort: skip


_DEFAULT_SPAWN_Z = 0.45  # robot's default z in asset config (unitree.py)
_failed_envs: set = set()


def _zero_command(env):
    """Always return zero velocity — robot must survive and stabilize on its own."""
    return torch.zeros(env.num_envs, 3, device=env.device)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Flat terrain — no curriculum
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # Disable observation noise for clean eval
    env_cfg.observations.policy.enable_corruption = False

    # Disable push disturbances
    if hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None

    # Override spawn z to drop_height by adjusting the reset pose_range z offset
    z_offset = args_cli.drop_height - _DEFAULT_SPAWN_Z
    if hasattr(env_cfg.events, "randomize_reset_base"):
        env_cfg.events.randomize_reset_base.params["pose_range"]["z"] = (z_offset, z_offset)
    else:
        print(f"[drop-eval] WARNING: randomize_reset_base not found — spawn height may not be {args_cli.drop_height:.2f} m")

    # Extend episode to cover the full eval duration plus a buffer
    env_cfg.episode_length_s = args_cli.eval_steps * 0.01 + 5.0

    # Inject zero velocity command throughout — focus is on impact survival
    env_cfg.observations.policy.velocity_commands = ObsTerm(func=_zero_command)

    # Load checkpoint
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Set SATA growth to training-end scale
    try:
        _sata = env.unwrapped.action_manager.get_term("sata_torque")
        num_steps_per_env = agent_cfg.num_steps_per_env
        if getattr(_sata.cfg, "freeze_growth", False):
            _sata._physics_step_counter = _sata.cfg.growth_initial_steps
        else:
            _sata._physics_step_counter = _sata.cfg.growth_initial_steps + num_steps_per_env * agent_cfg.max_iterations
        _sata._growth_scale = math.exp(
            -math.exp(-_sata.cfg.growth_k * (_sata._physics_step_counter - _sata.cfg.growth_x0))
        )
        env.unwrapped._sata_growth_scale = _sata._growth_scale
        _sata.current_torque_scale = (
            _sata._growth_scale * (_sata.cfg.max_torque_scale - _sata.cfg.initial_torque_scale)
            + _sata.cfg.initial_torque_scale
        )
        _sata.rear_torque_scale = (
            _sata._growth_scale * (_sata.cfg.max_rear_torque_scale - _sata.cfg.initial_rear_torque_scale)
            + _sata.cfg.initial_rear_torque_scale
        )
        _sata.cfg.action_loss_rate = 0.0
        _sata._obs_dropout_installed = True
        _sata.cfg.motor_fatigue_enabled = False
        _sata.motor_fatigue.zero_()
        print(f"[drop-eval] SATA growth set: step={_sata._physics_step_counter}, "
              f"growth={_sata._growth_scale:.4f}, torque_scale={_sata.current_torque_scale:.4f}")
    except Exception as e:
        print(f"[drop-eval] WARNING: Could not set SATA growth: {e}")

    total_steps = args_cli.eval_steps
    video_length = args_cli.video_length if args_cli.video_length > 0 else total_steps

    if args_cli.video:
        video_dir = os.path.join(log_dir, "videos", "drop_eval")
        video_kwargs = {
            "video_folder": video_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": video_length,
            "disable_logger": True,
        }
        print(f"[INFO] Recording video to: {video_dir}")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    dt = env.unwrapped.step_dt

    records = []
    termination_log = []  # (step, env_id)

    obs = env.get_observations()
    print(f"\n[drop-eval] Dropping {args_cli.num_envs} robot(s) from {args_cli.drop_height:.2f} m above ground\n")
    print(f"{'step':>6}  {'height_m':>8}  {'vz_ms':>7}  {'pitch_deg':>10}  {'roll_deg':>9}  {'mean_tau':>10}  {'alive':>6}")
    print("-" * 72)

    step = 0
    while simulation_app.is_running() and step < total_steps:
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)

        base_env = env.unwrapped
        robot = base_env.scene["robot"]

        height = robot.data.root_pos_w[:, 2].mean().item()
        vz = robot.data.root_lin_vel_w[:, 2].mean().item()

        gravity_x = robot.data.projected_gravity_b[:, 0].mean().item()
        gravity_y = robot.data.projected_gravity_b[:, 1].mean().item()
        pitch_deg = torch.rad2deg(torch.asin(torch.clamp(torch.tensor(gravity_x), -1.0, 1.0))).item()
        roll_deg = torch.rad2deg(torch.asin(torch.clamp(torch.tensor(gravity_y), -1.0, 1.0))).item()

        try:
            _sata = base_env.action_manager.get_term("sata_torque")
            mean_tau = _sata.processed_actions.abs().mean().item()
        except Exception:
            mean_tau = float("nan")

        # Track first termination per env
        terminated_ids = dones.nonzero(as_tuple=False).squeeze(-1).tolist()
        for env_id in terminated_ids:
            if env_id not in _failed_envs:
                _failed_envs.add(env_id)
                termination_log.append((step, env_id))
                print(f"  [FAILED] step={step}  env={env_id}  height={robot.data.root_pos_w[env_id, 2].item():.3f} m  "
                      f"pitch={pitch_deg:.1f} deg")

        n_alive = args_cli.num_envs - len(_failed_envs)
        record = {
            "step": step,
            "height_m": height,
            "vz_ms": vz,
            "pitch_deg": pitch_deg,
            "roll_deg": roll_deg,
            "mean_tau": mean_tau,
            "n_alive": n_alive,
        }
        records.append(record)

        if step % 20 == 0:
            print(f"{step:>6}  {height:>8.3f}  {vz:>7.3f}  {pitch_deg:>10.2f}  {roll_deg:>9.2f}  "
                  f"{mean_tau:>10.3f}  {n_alive:>4}/{args_cli.num_envs}")

        step += 1

        if args_cli.video and step >= video_length:
            break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    print("\n[drop-eval] Done.")
    n = args_cli.num_envs
    n_failed = len(_failed_envs)
    print(f"\n  Drop height : {args_cli.drop_height:.2f} m")
    print(f"  Result      : {n - n_failed}/{n} survived")
    if termination_log:
        from collections import Counter
        fail_steps = [s for s, _ in termination_log]
        print(f"  Failure steps: {fail_steps}")
        print(f"  Mean step at failure: {sum(fail_steps) / len(fail_steps):.1f}")
    else:
        print("  No terminations — all robots survived the drop.")

    if args_cli.save_csv and records:
        import csv
        with open(args_cli.save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        print(f"[drop-eval] Metrics saved to {args_cli.save_csv}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
