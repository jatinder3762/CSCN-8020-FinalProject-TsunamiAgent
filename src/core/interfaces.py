"""Protocol-based interfaces for extensible RL components."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class AgentProtocol(Protocol):
    """Defines the required behavior for any policy-learning agent."""

    epsilon: float

    def choose_action(
        self,
        state_idx: int,
        training: bool = True,
        valid_actions: list[int] | tuple[int, ...] | None = None,
    ) -> int:
        """Returns the selected action id."""

    def update(self, state_idx: int, action: int, reward: float, next_state_idx: int, done: bool) -> None:
        """Applies one learning update step."""

    def decay_epsilon(self) -> None:
        """Updates exploration rate after an episode."""

    def save_q_table(self, path: Path) -> None:
        """Persists learned table to disk."""

    def load_q_table(self, path: Path) -> None:
        """Loads learned table from disk."""


class EnvironmentProtocol(Protocol):
    """Defines the required behavior for any discrete tsunami environment."""

    def reset(self) -> int:
        """Resets episode and returns encoded initial state."""

    def step(self, action: int) -> tuple[int, float, bool, dict[str, Any]]:
        """Runs one transition and returns (next_state, reward, done, info)."""

    def get_valid_actions(self) -> list[int]:
        """Returns valid action ids for the current operational state."""

