"""Q-Learning agent implementation using a NumPy Q-table."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class QLearningAgent:
    """Tabular Q-learning agent with epsilon-greedy exploration."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        alpha: float,
        gamma: float,
        epsilon: float,
        epsilon_decay: float,
        min_epsilon: float,
        seed: int = 42,
    ) -> None:
        """Initializes the Q-table and learning hyperparameters."""
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.rng = np.random.default_rng(seed)
        self.q_table = np.zeros((state_size, action_size), dtype=np.float64)

    def choose_action(self, state_idx: int, training: bool = True) -> int:
        """Chooses an action by epsilon-greedy policy."""
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.action_size))

        q_values = self.q_table[state_idx]
        max_value = np.max(q_values)
        best_actions = np.flatnonzero(np.isclose(q_values, max_value))
        return int(self.rng.choice(best_actions))

    def update(self, state_idx: int, action: int, reward: float, next_state_idx: int, done: bool) -> None:
        """Performs one Q-learning update step."""
        current_q = self.q_table[state_idx, action]
        next_max_q = 0.0 if done else float(np.max(self.q_table[next_state_idx]))
        td_target = reward + (self.gamma * next_max_q)
        td_error = td_target - current_q
        self.q_table[state_idx, action] = current_q + (self.alpha * td_error)

    def decay_epsilon(self) -> None:
        """Decays exploration rate after each episode."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_q_table(self, path: Path) -> None:
        """Saves Q-table to a .npy file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.q_table)

    def load_q_table(self, path: Path) -> None:
        """Loads Q-table from a .npy file."""
        self.q_table = np.load(path)
