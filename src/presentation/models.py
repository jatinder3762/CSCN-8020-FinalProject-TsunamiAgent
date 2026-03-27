"""Presentation-layer data models for deterministic dashboard scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TsunamiState:
    """Represents the observable tsunami decision state at a point in time."""

    magnitude: str
    depth: str
    wave_risk: str
    confidence: str
    time_label: str

    def to_card_mapping(self) -> dict[str, str]:
        """Returns a label-to-value mapping for dashboard state cards."""
        return {
            "Magnitude": self.magnitude,
            "Depth": self.depth,
            "WaveRisk": self.wave_risk,
            "Confidence": self.confidence,
            "Time": self.time_label,
        }


@dataclass(frozen=True)
class DecisionStep:
    """Represents one step in a deterministic presentation scenario."""

    step_index: int
    time_label: str
    magnitude: str
    depth: str
    wave_risk: str
    confidence: str
    action: str
    reward: float
    explanation: str
    status: str

    @property
    def state(self) -> TsunamiState:
        """Builds a state view from the step fields."""
        return TsunamiState(
            magnitude=self.magnitude,
            depth=self.depth,
            wave_risk=self.wave_risk,
            confidence=self.confidence,
            time_label=self.time_label,
        )

    def to_history_row(self, cumulative_reward: float, is_current: bool) -> dict[str, Any]:
        """Returns a flat record for the dashboard history table."""
        return {
            "Current": "Yes" if is_current else "",
            "Step": self.step_index + 1,
            "Time": self.time_label,
            "Action": self.action,
            "Reward": float(self.reward),
            "Cumulative Reward": round(float(cumulative_reward), 2),
            "Status": self.status,
            "Explanation": self.explanation,
        }


@dataclass(frozen=True)
class EpisodeTrace:
    """Represents a named scenario with ordered decision steps."""

    name: str
    summary: str
    risk_level: str
    outcome_label: str
    steps: tuple[DecisionStep, ...]

    def __post_init__(self) -> None:
        """Validates that the scenario contains at least one step."""
        if not self.steps:
            raise ValueError("EpisodeTrace requires at least one decision step.")

    @property
    def total_reward(self) -> float:
        """Returns the total reward across all steps."""
        return round(sum(float(step.reward) for step in self.steps), 2)

    @property
    def step_count(self) -> int:
        """Returns the number of steps in the scenario."""
        return len(self.steps)

    @property
    def last_step_index(self) -> int:
        """Returns the zero-based index of the final step."""
        return len(self.steps) - 1

    def get_step(self, step_index: int) -> DecisionStep:
        """Returns a safely clamped step for the requested index."""
        bounded_index = max(0, min(int(step_index), self.last_step_index))
        return self.steps[bounded_index]

    def cumulative_reward_at(self, step_index: int) -> float:
        """Returns cumulative reward through the requested step."""
        bounded_index = max(0, min(int(step_index), self.last_step_index))
        return round(
            sum(float(step.reward) for step in self.steps[: bounded_index + 1]),
            2,
        )

    def history_rows(self, visible_step_index: int) -> list[dict[str, Any]]:
        """Builds history rows through the currently visible step."""
        bounded_index = max(0, min(int(visible_step_index), self.last_step_index))
        rows: list[dict[str, Any]] = []
        running_total = 0.0
        for index, step in enumerate(self.steps[: bounded_index + 1]):
            running_total += float(step.reward)
            rows.append(
                step.to_history_row(
                    cumulative_reward=running_total,
                    is_current=index == bounded_index,
                )
            )
        return rows

    def sidebar_summary(self) -> dict[str, str]:
        """Returns a compact summary for sidebar presentation."""
        return {
            "Risk Level": self.risk_level,
            "Steps": str(self.step_count),
            "Outcome": self.outcome_label,
            "Total Reward": f"{self.total_reward:+.2f}",
        }
