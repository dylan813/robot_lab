"""Curriculum terms for SATA fine-tuning (braking, drop recovery, and step-drop)."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class BrakingVelocityCurriculum(ManagerTermBase):
    """Performance-based curriculum for braking fine-tuning.

    Advances through stages based on measured robot competence rather than
    fixed step thresholds. Each stage has a performance criterion that must
    be met over a rolling window before advancing. A max_steps cap prevents
    the curriculum from stalling indefinitely.

    Stages:
        0 - Slow walk: vx=(0.3, 1.0). Advance when mean tracking reward > threshold.
        1 - Fast walk: vx=(0.8, 2.0). Advance when mean tracking reward > threshold.
        2 - Pre-brake: vx=(1.0, 2.0). Advance when mean tracking reward > threshold.
        3 - Braking:   vx=(1.5, 2.5), short resampling. Final stage.
    """

    # 4500 total iter budget: ~750 iter per stage (18k steps), braking gets remainder (~1500 iter).
    # At 24 steps/iter: 18k steps = 750 iter.
    STAGES = [
        {"vx": (0.3, 1.0), "resample": (5.0, 8.0), "standing": 0.1, "max_steps": 18000, "label": "slow_walk"},
        {"vx": (0.8, 2.0), "resample": (4.0, 6.0), "standing": 0.1, "max_steps": 18000, "label": "fast_walk"},
        {"vx": (1.0, 2.0), "resample": (3.0, 5.0), "standing": 0.1, "max_steps": 18000, "label": "pre_brake"},
        {"vx": (1.5, 2.5), "resample": (2.0, 4.0), "standing": 0.1, "max_steps": 0,     "label": "braking"},
    ]

    # Rolling window size and threshold for optional performance gate.
    WINDOW_SIZE = 50
    WALK_THRESHOLD = 5.5

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._stage = 0
        self._stage_start_step = 0
        self._perf_window: deque[float] = deque(maxlen=self.WINDOW_SIZE)
        self._apply_stage(env, 0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_stage(self, env: ManagerBasedRLEnv, stage_idx: int):
        stage = self.STAGES[stage_idx]
        cmd_term = env.command_manager.get_term("base_velocity")
        cmd_term.cfg.ranges.lin_vel_x = stage["vx"]
        cmd_term.cfg.resampling_time_range = stage["resample"]
        cmd_term.cfg.rel_standing_envs = stage["standing"]
        print(
            f"[BrakingCurriculum] --> Stage {stage_idx} '{stage['label']}': "
            f"vx={stage['vx']}, resample={stage['resample']}, standing={stage['standing']}"
        )

    def _mean_tracking_reward(self, env: ManagerBasedRLEnv) -> float:
        """Mean forward tracking reward per step, averaged across all envs."""
        try:
            ep_sums = env.reward_manager.episode_sums["forward"]
            ep_len = env.episode_length_buf.float().clamp(min=1)
            return (ep_sums / ep_len).mean().item()
        except Exception:
            return 0.0

    def _should_advance(self, env: ManagerBasedRLEnv) -> bool:
        """Return True if the robot meets the performance criterion for the current stage."""
        if self._stage >= len(self.STAGES) - 1:
            return False  # already at final stage

        stage = self.STAGES[self._stage]
        steps_in_stage = env.common_step_counter - self._stage_start_step

        # Force-advance if max_steps exceeded
        if stage["max_steps"] > 0 and steps_in_stage >= stage["max_steps"]:
            print(f"[BrakingCurriculum] Stage {self._stage} max_steps reached — advancing.")
            return True

        # Early advance if performance is already consistently good
        perf = self._mean_tracking_reward(env)
        self._perf_window.append(perf)
        if len(self._perf_window) < self.WINDOW_SIZE:
            return False
        return (sum(self._perf_window) / len(self._perf_window)) > self.WALK_THRESHOLD

    # ------------------------------------------------------------------
    # Curriculum term entrypoint
    # ------------------------------------------------------------------

    def __call__(self, env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> float:
        if self._should_advance(env):
            self._stage += 1
            self._stage_start_step = env.common_step_counter
            self._perf_window.clear()
            self._apply_stage(env, self._stage)

        return float(self._stage)


class DropHeightCurriculum(ManagerTermBase):
    """Performance-based curriculum for drop recovery fine-tuning.

    Gradually increases the drop height as the robot demonstrates it can
    survive at the current height. Advancement is gated on mean episode
    length (a proxy for survival rate) over a rolling window.

    Stages:
        0 - Low  : drop from 0.5 m. Advance when mean episode length > threshold.
        1 - Mid  : drop from 0.8 m. Advance when mean episode length > threshold.
        2 - High : drop from 1.2 m. Advance when mean episode length > threshold.
        3 - Max  : drop from 1.5 m. Final stage.
    """

    _DEFAULT_SPAWN_Z = 0.45  # robot's default z in UNITREE_GO2W_SATA_CFG

    # 3000 iter budget: ~750 iter per stage (18k steps), max stage gets remainder.
    # At 24 steps/iter: 18k steps = 750 iter.
    STAGES = [
        {"height": 0.5, "max_steps": 18000, "label": "low"},
        {"height": 0.8, "max_steps": 18000, "label": "mid"},
        {"height": 1.2, "max_steps": 18000, "label": "high"},
        {"height": 1.5, "max_steps": 0,     "label": "max"},
    ]

    # Advance early if mean episode length stays above this for WINDOW_SIZE calls.
    # At episode_length_s=8s / dt=0.01 → max 800 steps. Threshold of 150 means
    # the robot is surviving ~1.5 s on average before failing.
    WINDOW_SIZE = 50
    SURVIVE_THRESHOLD = 150.0

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._stage = 0
        self._stage_start_step = 0
        self._perf_window: deque[float] = deque(maxlen=self.WINDOW_SIZE)
        self._apply_stage(env, 0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_stage(self, env: ManagerBasedRLEnv, stage_idx: int):
        stage = self.STAGES[stage_idx]
        z_offset = stage["height"] - self._DEFAULT_SPAWN_Z
        try:
            event_term = env.event_manager.get_term("randomize_reset_base")
            event_term.cfg.params["pose_range"]["z"] = (z_offset, z_offset)
        except Exception as e:
            print(f"[DropCurriculum] WARNING: Could not set drop height: {e}")
        print(
            f"[DropCurriculum] --> Stage {stage_idx} '{stage['label']}': "
            f"drop_height={stage['height']:.1f} m (z_offset={z_offset:+.2f})"
        )

    def _mean_episode_length(self, env: ManagerBasedRLEnv) -> float:
        """Mean episode length across all envs — proxy for survival rate."""
        return env.episode_length_buf.float().mean().item()

    def _should_advance(self, env: ManagerBasedRLEnv) -> bool:
        if self._stage >= len(self.STAGES) - 1:
            return False

        stage = self.STAGES[self._stage]
        steps_in_stage = env.common_step_counter - self._stage_start_step

        if stage["max_steps"] > 0 and steps_in_stage >= stage["max_steps"]:
            print(f"[DropCurriculum] Stage {self._stage} max_steps reached — advancing.")
            return True

        perf = self._mean_episode_length(env)
        self._perf_window.append(perf)
        if len(self._perf_window) < self.WINDOW_SIZE:
            return False
        return (sum(self._perf_window) / len(self._perf_window)) > self.SURVIVE_THRESHOLD

    # ------------------------------------------------------------------
    # Curriculum term entrypoint
    # ------------------------------------------------------------------

    def __call__(self, env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> float:
        if self._should_advance(env):
            self._stage += 1
            self._stage_start_step = env.common_step_counter
            self._perf_window.clear()
            self._apply_stage(env, self._stage)

        return float(self._stage)


class StepDropCurriculum(ManagerTermBase):
    """Performance-based curriculum for step-drop (drive-off-ledge) fine-tuning.

    The robot spawns on the back of a physical block, walks forward, and drives
    off the front edge. Each stage raises the block height and increases the
    forward speed range, so the robot must launch faster as the drop gets taller.

    The block is a kinematic rigid body in the scene ("platform"). Its Z position
    is updated each stage so the top surface sits at the target height.

    Block geometry (set in step_drop_train_env_cfg.py):
        length=3.0 m, width=2.0 m, height=1.0 m
        front edge at x=0 (env local), center at x=-1.5

    Robot spawns at x=-2.5 (2.5 m from edge) with zero velocity — settles
    on the block then walks toward the edge under the velocity command.

    Stages:
        0 - low    : block top=0.3 m, vx=(0.5, 1.0) m/s.
        1 - target : block top=0.5 m, vx=(0.8, 1.5) m/s. Final stage.
    """

    # Block geometry constants — must match step_drop_train_env_cfg.py
    BLOCK_LENGTH = 3.0
    BLOCK_HALF_HEIGHT = 0.5   # block is 1.0 m tall
    SPAWN_X_OFFSET = -2.5     # robot x relative to env origin (2.5 m from front edge)
    SPAWN_Z_CLEARANCE = 0.05  # small gap above block surface so robot settles gently

    # ~500 RL iterations per stage (500 * 24 steps = 12 000 common_step_counter increments).
    # Final stage gets the remainder of the 3000-iteration budget.
    STAGES = [
        {"height": 0.3, "vx": (0.5, 1.0), "max_steps": 12000, "label": "low"},
        {"height": 0.5, "vx": (0.8, 1.5), "max_steps": 0,     "label": "target"},
    ]

    # Performance gate: robot must survive an average of SURVIVE_THRESHOLD steps
    # to advance early. Max episode = 15 s × 200 Hz = 3000 steps.
    # At vx=1.0 m/s, reaching the edge takes ~250 steps. Threshold of 700 means
    # the robot must reliably land AND continue walking for ~4.5 s after the edge.
    # MIN_STAGE_STEPS prevents the gate from firing before 250 RL iterations
    # (250 * 24 = 6000) — avoids the base policy instantly blasting through stages.
    WINDOW_SIZE = 50
    SURVIVE_THRESHOLD = 700.0
    MIN_STAGE_STEPS = 6000

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._stage = 0
        self._stage_start_step = 0
        self._perf_window: deque[float] = deque(maxlen=self.WINDOW_SIZE)
        self._apply_stage(env, 0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_stage(self, env: ManagerBasedRLEnv, stage_idx: int):
        import torch
        stage = self.STAGES[stage_idx]
        height = stage["height"]
        z_offset = height + self.SPAWN_Z_CLEARANCE  # robot COM = 0.45 + z_offset

        # Move kinematic block so its top surface is at stage height.
        # Block center z = height - half_height (may be negative = partially underground).
        try:
            platform = env.scene["platform"]
            root_state = platform.data.root_state_w.clone()
            root_state[:, 2] = height - self.BLOCK_HALF_HEIGHT
            platform.write_root_state_to_sim(root_state)
        except Exception as e:
            print(f"[StepDropCurriculum] WARNING: Could not move platform: {e}")

        # Update robot spawn height to match new block top surface.
        try:
            event_term = env.event_manager.get_term("randomize_reset_base")
            event_term.cfg.params["pose_range"]["z"] = (z_offset, z_offset)
        except Exception as e:
            print(f"[StepDropCurriculum] WARNING: Could not set spawn z: {e}")

        # Update velocity command range — faster at higher stages.
        try:
            cmd_term = env.command_manager.get_term("base_velocity")
            cmd_term.cfg.ranges.lin_vel_x = stage["vx"]
        except Exception as e:
            print(f"[StepDropCurriculum] WARNING: Could not set command range: {e}")

        print(
            f"[StepDropCurriculum] --> Stage {stage_idx} '{stage['label']}': "
            f"block_top={height:.1f} m, vx={stage['vx']} m/s"
        )

    def _mean_episode_length(self, env: ManagerBasedRLEnv) -> float:
        return env.episode_length_buf.float().mean().item()

    def _should_advance(self, env: ManagerBasedRLEnv) -> bool:
        if self._stage >= len(self.STAGES) - 1:
            return False

        stage = self.STAGES[self._stage]
        steps_in_stage = env.common_step_counter - self._stage_start_step

        if stage["max_steps"] > 0 and steps_in_stage >= stage["max_steps"]:
            print(f"[StepDropCurriculum] Stage {self._stage} max_steps reached — advancing.")
            return True

        # Performance gate disabled until minimum stage duration has elapsed.
        if steps_in_stage < self.MIN_STAGE_STEPS:
            return False

        perf = self._mean_episode_length(env)
        self._perf_window.append(perf)
        if len(self._perf_window) < self.WINDOW_SIZE:
            return False
        return (sum(self._perf_window) / len(self._perf_window)) > self.SURVIVE_THRESHOLD

    # ------------------------------------------------------------------
    # Curriculum term entrypoint
    # ------------------------------------------------------------------

    def __call__(self, env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> float:
        if self._should_advance(env):
            self._stage += 1
            self._stage_start_step = env.common_step_counter
            self._perf_window.clear()
            self._apply_stage(env, self._stage)

        return float(self._stage)
