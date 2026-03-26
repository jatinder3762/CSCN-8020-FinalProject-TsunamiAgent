"""Project configuration for the RL-Based Tsunami Alert Decision System."""

from __future__ import annotations

from math import prod
from pathlib import Path
from typing import Any


class ProjectConfig:
    """Stores constants, hyperparameters, and project paths."""

    def __init__(
        self,
        base_dir: Path | None = None,
        training_episodes: int = 2500,
        evaluation_episodes: int = 400,
        alpha: float = 0.15,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.997,
        min_epsilon: float = 0.05,
        random_seed: int = 42,
    ) -> None:
        self.project_title = "RL-Based Tsunami Alert Decision System"
        self.base_dir = (base_dir or Path(__file__).resolve().parent).resolve()

        self.magnitude_levels = ("Low", "Medium", "High")
        self.depth_levels = ("Deep", "Moderate", "Shallow")
        self.wave_risk_levels = ("Low", "Medium", "High")
        self.confidence_levels = ("Low", "Medium", "High")
        self.time_levels = ("Early", "Mid", "Late")
        self.risk_levels = ("Low", "Medium", "High")

        self.action_names = {
            0: "Wait",
            1: "Verify",
            2: "Regional Alert",
            3: "Full Alert",
        }

        self.training_episodes = int(training_episodes)
        self.evaluation_episodes = int(evaluation_episodes)
        self.max_steps_per_episode = 3
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.min_epsilon = float(min_epsilon)
        self.random_seed = int(random_seed)

        # Reward weights (configurable for experiments).
        self.reward_correct_full_alert = 100.0
        self.reward_correct_regional_alert = 60.0
        self.reward_smart_verify = 30.0
        self.penalty_missed_dangerous_alert = -100.0
        self.penalty_false_full_alert = -60.0
        self.penalty_false_regional_alert = -30.0
        self.reward_delay_per_step = -10.0
        self.penalty_unnecessary_verify = -15.0
        self.penalty_late_wait_in_risk = -20.0
        self.reward_partial_regional_on_high = 25.0
        self.penalty_overreaction_full_on_medium = -20.0
        self.reward_safe_no_alert_low_risk = 10.0

        self.moving_average_window = 50

        self.data_dir = self.base_dir / "data"
        self.outputs_dir = self.base_dir / "outputs"
        self.models_dir = self.outputs_dir / "models"
        self.plots_dir = self.outputs_dir / "plots"
        self.logs_dir = self.outputs_dir / "logs"

        self.ensure_directories()

    @property
    def state_shape(self) -> tuple[int, int, int, int, int]:
        """Returns discrete dimensionality for each state component."""
        return (
            len(self.magnitude_levels),
            len(self.depth_levels),
            len(self.wave_risk_levels),
            len(self.confidence_levels),
            len(self.time_levels),
        )

    @property
    def state_size(self) -> int:
        """Returns the total number of discrete states."""
        return prod(self.state_shape)

    @property
    def action_size(self) -> int:
        """Returns the total number of actions."""
        return len(self.action_names)

    def ensure_directories(self) -> None:
        """Creates required project directories if they do not exist."""
        for path in (self.data_dir, self.outputs_dir, self.models_dir, self.plots_dir, self.logs_dir):
            path.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        """Exports config values as a serializable dictionary."""
        return {
            "project_title": self.project_title,
            "training_episodes": self.training_episodes,
            "evaluation_episodes": self.evaluation_episodes,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "min_epsilon": self.min_epsilon,
            "random_seed": self.random_seed,
            "state_shape": self.state_shape,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "rewards": {
                "reward_correct_full_alert": self.reward_correct_full_alert,
                "reward_correct_regional_alert": self.reward_correct_regional_alert,
                "reward_smart_verify": self.reward_smart_verify,
                "penalty_missed_dangerous_alert": self.penalty_missed_dangerous_alert,
                "penalty_false_full_alert": self.penalty_false_full_alert,
                "penalty_false_regional_alert": self.penalty_false_regional_alert,
                "reward_delay_per_step": self.reward_delay_per_step,
                "penalty_unnecessary_verify": self.penalty_unnecessary_verify,
                "penalty_late_wait_in_risk": self.penalty_late_wait_in_risk,
                "reward_partial_regional_on_high": self.reward_partial_regional_on_high,
                "penalty_overreaction_full_on_medium": self.penalty_overreaction_full_on_medium,
                "reward_safe_no_alert_low_risk": self.reward_safe_no_alert_low_risk,
            },
            "paths": {
                "base_dir": str(self.base_dir),
                "data_dir": str(self.data_dir),
                "outputs_dir": str(self.outputs_dir),
                "models_dir": str(self.models_dir),
                "plots_dir": str(self.plots_dir),
                "logs_dir": str(self.logs_dir),
            },
        }
