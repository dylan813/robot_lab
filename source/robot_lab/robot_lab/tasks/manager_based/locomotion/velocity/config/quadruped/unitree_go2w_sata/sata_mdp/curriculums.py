"""Curriculum terms for SATA braking fine-tuning."""

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
