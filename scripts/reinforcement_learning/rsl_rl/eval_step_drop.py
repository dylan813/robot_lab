"""Step-drop evaluation script for Unitree Go2W SATA.

Spawns a static platform in each environment. The robot starts on top of the
block, receives a forward velocity command, walks to the edge, and falls off.
Block height and length are configurable. Measures whether each robot survives
the landing and continues walking.

Logs phase (ON_BLOCK / FALLING / LANDED), forward velocity, height, pitch, roll,
and torque each step.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Step-drop evaluation for Go2W SATA.")
parser.add_argument("--num_envs", type=int, default=10, help="Number of robots.")
parser.add_argument("--task", type=str, default="RobotLab-Isaac-AccelBrake-Unitree-Go2W-SATA-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--block_height", type=float, default=0.5, help="Block height above ground (m).")
parser.add_argument("--block_length", type=float, default=3.0, help="Block length in driving direction (m).")
parser.add_argument("--drive_speed", type=float, default=1.0, help="Commanded forward speed (m/s).")
parser.add_argument("--eval_steps", type=int, default=1200, help="Steps to observe (~12 s at 100 Hz).")
parser.add_argument("--settle_steps", type=int, default=100, help="Steps with zero command at episode start (~1 s).")
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

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401  # isort: skip


_DEFAULT_SPAWN_Z = 0.45  # robot's default z in asset config (unitree.py)
_GROUND_HEIGHT = 0.45    # approximate standing body height above ground
_failed_envs: set = set()
_drive_speed = [1.0]
_settle_steps = [100]


def _drive_command(env):
    """Forward velocity command, withheld for the first settle_steps of each episode."""
    cmd = torch.zeros(env.num_envs, 3, device=env.device)
    active = env.episode_length_buf > _settle_steps[0]
    cmd[active, 0] = _drive_speed[0] * 2.0  # SATA observation scaling
    return cmd


def _phase(height: float, block_height: float) -> str:
    if height > block_height * 0.7 + _GROUND_HEIGHT:
        return "ON_BLOCK"
    elif height > _GROUND_HEIGHT + 0.15:
        return "FALLING"
    else:
        return " LANDED"


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    _drive_speed[0] = args_cli.drive_speed
    _settle_steps[0] = args_cli.settle_steps

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Flat terrain required — platform sits on flat ground
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None

    # Wider env spacing to avoid block overlap between environments
    env_cfg.scene.env_spacing = max(env_cfg.scene.env_spacing, args_cli.block_length + 3.0)

    # Disable observation noise
    env_cfg.observations.policy.enable_corruption = False

    # Disable push disturbances
    if hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None

    # -----------------------------------------------------------------------
    # Add static platform to the scene
    # Block spans x = [-block_length, 0] relative to env origin.
    # Front face (drop edge) is at x = 0.
    # -----------------------------------------------------------------------
    bh = args_cli.block_height
    bl = args_cli.block_length
    bw = max(bl, 1.5)  # width at least 1.5 m so robot has room

    env_cfg.scene.platform = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Platform",
        spawn=sim_utils.CuboidCfg(
            size=(bl, bw, bh),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1e6),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.35)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-bl / 2.0, 0.0, bh / 2.0)
        ),
    )

    # -----------------------------------------------------------------------
    # Spawn matches training distribution (step_drop_train_env_cfg.py):
    #   x = -2.5  (2.5 m from the drop edge — same run-up as training)
    #   z_offset = bh + 0.05  (5 cm clearance so robot settles gently)
    #   yaw = 0   (robot faces the drop edge, same as training)
    #   velocity = zero  (robot accelerates naturally under the drive command)
    # -----------------------------------------------------------------------
    x_spawn = -(bl - 0.5)  # 0.5 m from the back edge, matches training (3 m block → -2.5)
    z_offset = bh + 0.05
    if hasattr(env_cfg.events, "randomize_reset_base"):
        env_cfg.events.randomize_reset_base.params["pose_range"]["x"] = (x_spawn, x_spawn)
        env_cfg.events.randomize_reset_base.params["pose_range"]["y"] = (0.0, 0.0)
        env_cfg.events.randomize_reset_base.params["pose_range"]["z"] = (z_offset, z_offset)
        env_cfg.events.randomize_reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
        env_cfg.events.randomize_reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
    else:
        print("[step-drop] WARNING: randomize_reset_base not found — spawn position may be wrong")

    # Disable dof_pos_limits — policy applies large forces on first step settling
    # onto the hard kinematic surface before motion begins.
    # Disable head_contact — robot pitches forward on launch and the head
    # briefly touches the ground during landing; this is normal dynamics, not failure.
    # Only robot_flipped (true flip) and time_out remain as terminations.
    if hasattr(env_cfg.terminations, "dof_pos_limits"):
        env_cfg.terminations.dof_pos_limits = None
    if hasattr(env_cfg.terminations, "head_contact"):
        env_cfg.terminations.head_contact = None

    # Extend episode to cover full eval
    env_cfg.episode_length_s = args_cli.eval_steps * 0.01 + 5.0

    # Sustained forward command — robot drives off the block and continues
    env_cfg.observations.policy.velocity_commands = ObsTerm(func=_drive_command)

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
        print(f"[step-drop] SATA growth set: step={_sata._physics_step_counter}, "
              f"growth={_sata._growth_scale:.4f}, torque_scale={_sata.current_torque_scale:.4f}")
    except Exception as e:
        print(f"[step-drop] WARNING: Could not set SATA growth: {e}")

    total_steps = args_cli.eval_steps
    video_length = args_cli.video_length if args_cli.video_length > 0 else total_steps

    if args_cli.video:
        video_dir = os.path.join(log_dir, "videos", "step_drop")
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
    termination_log = []  # (step, env_id, phase)

    obs = env.get_observations()
    print(f"\n[step-drop] {args_cli.num_envs} robot(s) — block: {bl:.1f} m long x {bh:.2f} m tall, "
          f"drive_speed={args_cli.drive_speed:.2f} m/s\n")
    print(f"{'step':>6}  {'phase':>8}  {'vx_ms':>7}  {'height_m':>8}  {'vz_ms':>7}  "
          f"{'pitch_deg':>10}  {'alive':>6}")
    print("-" * 68)

    step = 0
    while simulation_app.is_running() and step < total_steps:
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)

        base_env = env.unwrapped
        robot = base_env.scene["robot"]

        vx = robot.data.root_lin_vel_b[:, 0].mean().item()
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

        phase_label = _phase(height, bh)

        # Track first termination per env
        terminated_ids = dones.nonzero(as_tuple=False).squeeze(-1).tolist()
        for env_id in terminated_ids:
            if env_id not in _failed_envs:
                _failed_envs.add(env_id)
                termination_log.append((step, env_id, phase_label.strip()))
                print(f"  [FAILED] step={step}  env={env_id}  phase={phase_label.strip()}  "
                      f"height={robot.data.root_pos_w[env_id, 2].item():.3f} m  pitch={pitch_deg:.1f} deg")

        n_alive = args_cli.num_envs - len(_failed_envs)
        record = {
            "step": step,
            "phase": phase_label.strip(),
            "vx_ms": vx,
            "height_m": height,
            "vz_ms": vz,
            "pitch_deg": pitch_deg,
            "roll_deg": roll_deg,
            "mean_tau": mean_tau,
            "n_alive": n_alive,
        }
        records.append(record)

        if step % 20 == 0:
            print(f"{step:>6}  {phase_label:>8}  {vx:>7.3f}  {height:>8.3f}  {vz:>7.3f}  "
                  f"{pitch_deg:>10.2f}  {n_alive:>4}/{args_cli.num_envs}")

        step += 1

        if args_cli.video and step >= video_length:
            break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    print("\n[step-drop] Done.")
    n = args_cli.num_envs
    n_failed = len(_failed_envs)
    print(f"\n  Block        : {bl:.1f} m long x {bh:.2f} m tall")
    print(f"  Drive speed  : {args_cli.drive_speed:.2f} m/s")
    print(f"  Result       : {n - n_failed}/{n} survived")
    if termination_log:
        from collections import Counter
        phase_counts = Counter(ph for _, _, ph in termination_log)
        print(f"  Failures by phase: {dict(phase_counts)}")
        fail_steps = [s for s, _, _ in termination_log]
        print(f"  Mean step at failure: {sum(fail_steps) / len(fail_steps):.1f}")
    else:
        print("  No terminations — all robots drove off and survived.")

    if args_cli.save_csv and records:
        import csv
        with open(args_cli.save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        print(f"[step-drop] Metrics saved to {args_cli.save_csv}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
