"""Accel/brake evaluation script for Unitree Go2W SATA.

Spawns the robot on flat ground and drives a deterministic velocity command
schedule: full-speed forward for ACCEL_STEPS, then zero for BRAKE_STEPS.
Logs forward velocity, commanded velocity, body pitch, and mean torque each step.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Accel/brake evaluation for Go2W SATA.")
parser.add_argument("--num_envs", type=int, default=10, help="Number of environments.")
parser.add_argument("--task", type=str, default="RobotLab-Isaac-AccelBrake-Unitree-Go2W-SATA-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--accel_vel", type=float, default=2.5, help="Target forward velocity (m/s) during acceleration.")
parser.add_argument("--settle_steps", type=int, default=200, help="Steps at vx=0 before accel (~2 s at 200 Hz).")
parser.add_argument("--accel_steps", type=int, default=500, help="Steps at full speed (~5 s).")
parser.add_argument("--brake_steps", type=int, default=500, help="Steps at vx=0 after accel (~5 s).")
parser.add_argument("--save_csv", type=str, default=None, help="Optional path to save metrics CSV.")
parser.add_argument("--video", action="store_true", default=False, help="Record video of the evaluation.")
parser.add_argument("--video_length", type=int, default=0, help="Video length in steps (0 = full run).")
parser.add_argument("--real-time", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Cameras must be enabled before AppLauncher starts
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

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


# ---------------------------------------------------------------------------
# Command schedule
# ---------------------------------------------------------------------------

_step = [0]
_accel_vel = [2.5]
_accel_steps = [300]
_brake_steps = [350]
_settle_steps = [300]
_failed_envs: set = set()  # env indices that failed — receive vx=0 for rest of run


def _scheduled_command(env):
    """Returns a SATA-scaled velocity command: settle → linear ramp to accel_vel → brake (one shot).
    Failed envs always receive vx=0 after their first termination.
    """
    s = _step[0]
    if s < _settle_steps[0]:
        vel_x = 0.0
    elif s < _settle_steps[0] + _accel_steps[0]:
        progress = (s - _settle_steps[0]) / max(_accel_steps[0] - 1, 1)
        vel_x = _accel_vel[0] * progress
    else:
        vel_x = 0.0
    cmd = torch.zeros(env.num_envs, 3, device=env.device)
    for i in range(env.num_envs):
        if i not in _failed_envs:
            cmd[i, 0] = vel_x * 2.0  # SATA observation scaling
    return cmd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    # Push CLI parameters into the schedule globals
    _accel_vel[0] = args_cli.accel_vel
    _accel_steps[0] = args_cli.accel_steps
    _brake_steps[0] = args_cli.brake_steps
    _settle_steps[0] = args_cli.settle_steps

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Flat ground — already set in accel_brake_env_cfg, ensure no terrain curriculum
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # Disable observation noise for clean eval
    env_cfg.observations.policy.enable_corruption = False

    # Disable push disturbances
    if hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None

    # Extend episode length to cover settle + full experiment duration
    total_seconds = (args_cli.settle_steps + args_cli.accel_steps + args_cli.brake_steps) * 0.01
    env_cfg.episode_length_s = total_seconds + 10.0  # extra buffer

    # Inject deterministic command into policy observations
    env_cfg.observations.policy.velocity_commands = ObsTerm(func=_scheduled_command)

    # Load checkpoint
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Set SATA growth to match training-end scale (same approach as original SATA play.py:
    # step_count = num_steps_per_env * checkpoint = 24 * 3000 = 72,000)
    try:
        _sata = env.unwrapped.action_manager.get_term("sata_torque")
        num_steps_per_env = agent_cfg.num_steps_per_env  # 24
        if getattr(_sata.cfg, "freeze_growth", False):
            _sata._physics_step_counter = _sata.cfg.growth_initial_steps
        else:
            _sata._physics_step_counter = _sata.cfg.growth_initial_steps + num_steps_per_env * agent_cfg.max_iterations
        import math
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
        print(f"[accel-brake] SATA growth set to training-end: step={_sata._physics_step_counter}, "
              f"growth={_sata._growth_scale:.4f}, torque_scale={_sata.current_torque_scale:.4f}")
    except Exception as e:
        print(f"[accel-brake] WARNING: Could not set SATA growth: {e}")

    total_steps = _settle_steps[0] + _accel_steps[0] + _brake_steps[0]
    video_length = args_cli.video_length if args_cli.video_length > 0 else total_steps

    # Wrap for video recording
    if args_cli.video:
        video_dir = os.path.join(log_dir, "videos", "accel_brake")
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

    # Metrics log
    records = []
    termination_log = []  # (step, phase, env_ids)

    obs = env.get_observations()
    print(f"\n[accel-brake] Starting: settle {_settle_steps[0]} steps, "
          f"accel {_accel_vel[0]} m/s for {_accel_steps[0]} steps, "
          f"brake for {_brake_steps[0]} steps\n")
    print(f"{'step':>6}  {'phase':>6}  {'cmd_vx':>8}  {'vx':>8}  {'pitch_deg':>10}  {'mean_tau':>10}")
    print("-" * 60)

    while simulation_app.is_running() and _step[0] < total_steps:
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)

        # --- metrics ---
        base_env = env.unwrapped
        robot = base_env.scene["robot"]

        vx = robot.data.root_lin_vel_b[:, 0].mean().item()
        gravity_x = robot.data.projected_gravity_b[:, 0].mean().item()
        pitch_deg = torch.rad2deg(torch.asin(torch.clamp(torch.tensor(gravity_x), -1.0, 1.0))).item()

        s = _step[0]
        if s < _settle_steps[0]:
            cmd_vx = 0.0
            phase_label = "SETTLE"
        elif s < _settle_steps[0] + _accel_steps[0]:
            cmd_vx = _accel_vel[0]
            phase_label = "ACCEL"
        else:
            cmd_vx = 0.0
            phase_label = "BRAKE"

        # track first termination per env — subsequent resets of a failed env are ignored
        terminated_ids = dones.nonzero(as_tuple=False).squeeze(-1).tolist()
        for env_id in terminated_ids:
            if env_id not in _failed_envs:
                _failed_envs.add(env_id)
                termination_log.append((_step[0], phase_label, env_id))
                print(f"  [FAILED] step={_step[0]}  phase={phase_label}  env={env_id}")

        try:
            _sata = base_env.action_manager.get_term("sata_torque")
            mean_tau = _sata.processed_actions.abs().mean().item()
        except Exception:
            mean_tau = float("nan")

        record = {
            "step": _step[0],
            "phase": phase_label,
            "cmd_vx": cmd_vx,
            "vx": vx,
            "pitch_deg": pitch_deg,
            "mean_tau": mean_tau,
        }
        records.append(record)

        if _step[0] % 20 == 0:
            print(f"{_step[0]:>6}  {phase_label:>6}  {cmd_vx:>8.2f}  {vx:>8.3f}  {pitch_deg:>10.2f}  {mean_tau:>10.3f}")

        _step[0] += 1

        # Stop video wrapper after video_length steps (RecordVideo handles this internally,
        # but we also stop the loop for non-video runs)
        if args_cli.video and _step[0] >= video_length:
            break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    print("\n[accel-brake] Done.")
    n = args_cli.num_envs
    n_failed = len(_failed_envs)
    print(f"\n  Result: {n - n_failed}/{n} succeeded")
    if termination_log:
        from collections import Counter
        phase_counts = Counter(phase for _, phase, _ in termination_log)
        print(f"  Failures by phase: {dict(phase_counts)}")
    else:
        print("  No terminations — all robots completed the run.")

    # Optional CSV save
    if args_cli.save_csv and records:
        import csv
        with open(args_cli.save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        print(f"[accel-brake] Metrics saved to {args_cli.save_csv}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
