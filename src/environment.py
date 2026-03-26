"""Custom discrete tsunami alert environment built from scratch."""

from __future__ import annotations

from typing import Any

import numpy as np

from config import ProjectConfig
from src.utils import LabelFormatter, StateCodec


class TsunamiAlertEnvironment:
    """Simulates a tsunami warning decision process with discrete states and actions."""

    def __init__(self, config: ProjectConfig, seed: int | None = None) -> None:
        """Initializes environment state, RNG, and hidden episode variables."""
        self.config = config
        self.rng = np.random.default_rng(config.random_seed if seed is None else seed)

        self.state: tuple[int, int, int, int, int] = (0, 0, 0, 0, 0)
        self.hidden_wave_risk: int = 0
        self.hidden_true_risk: int = 0
        self.step_count: int = 0
        self.done: bool = False

    def reset(self) -> int:
        """Starts a new episode and returns encoded initial state index."""
        self.state = self._generate_initial_state()
        self.step_count = 0
        self.done = False
        return self.state_to_index(self.state)

    def step(self, action: int) -> tuple[int, float, bool, dict[str, Any]]:
        """Performs one environment step and returns standard RL tuple outputs."""
        if self.done:
            raise RuntimeError("Cannot call step() after episode is done. Call reset() first.")
        if action not in self.config.action_names:
            raise ValueError(f"Invalid action {action}. Valid actions: {list(self.config.action_names)}")

        previous_state = self.state
        next_state = self._transition_state(action)
        terminal = self._is_terminal(action, next_state)

        reward, alert_correct, false_alert, missed_alert = self._calculate_reward(
            action=action,
            previous_state=previous_state,
            next_state=next_state,
            terminal=terminal,
        )

        self.state = next_state
        self.done = terminal
        self.step_count += 1

        info: dict[str, Any] = {
            "actual_risk_level": self.config.risk_levels[self.hidden_true_risk],
            "action_meaning": LabelFormatter.action_name(action, self.config),
            "alert_correct": alert_correct,
            "false_alert": false_alert,
            "missed_alert": missed_alert,
            "step_count": self.step_count,
            "state_tuple": self.state,
            "state_text": LabelFormatter.state_name(self.state, self.config),
        }
        return self.state_to_index(self.state), reward, self.done, info

    def state_to_index(self, state: tuple[int, int, int, int, int]) -> int:
        """Converts a tuple state into an integer index."""
        return StateCodec.encode_state(state, self.config.state_shape)

    def index_to_state(self, index: int) -> tuple[int, int, int, int, int]:
        """Converts an integer index back into tuple state."""
        return StateCodec.decode_state(index, self.config.state_shape)

    def set_state_for_testing(
        self,
        state: tuple[int, int, int, int, int],
        hidden_true_risk: int | None = None,
        hidden_wave_risk: int | None = None,
    ) -> None:
        """Injects deterministic state values for tests."""
        self.state = state
        self.hidden_true_risk = self._clamp(hidden_true_risk if hidden_true_risk is not None else self._get_ground_truth_risk(state), 0, 2)
        self.hidden_wave_risk = self._clamp(hidden_wave_risk if hidden_wave_risk is not None else state[2], 0, 2)
        self.done = False

    def _generate_initial_state(self) -> tuple[int, int, int, int, int]:
        """Builds an initial observable state while creating hidden event truth."""
        magnitude = int(self.rng.choice([0, 1, 2], p=[0.40, 0.38, 0.22]))
        depth = int(self.rng.choice([0, 1, 2], p=[0.36, 0.36, 0.28]))

        self.hidden_wave_risk = self._sample_hidden_wave_risk(magnitude, depth)
        observed_wave_noise = int(self.rng.choice([-1, 0, 1], p=[0.20, 0.60, 0.20]))
        observed_wave_risk = self._clamp(self.hidden_wave_risk + observed_wave_noise, 0, 2)

        confidence = int(self.rng.choice([0, 1, 2], p=[0.42, 0.40, 0.18]))
        time_index = 0

        state = (magnitude, depth, observed_wave_risk, confidence, time_index)
        self.hidden_true_risk = self._get_ground_truth_risk((magnitude, depth, self.hidden_wave_risk, confidence, time_index))
        return state

    def _sample_hidden_wave_risk(self, magnitude: int, depth: int) -> int:
        """Samples hidden wave risk using magnitude and depth as priors."""
        center = (magnitude + depth) / 2.0
        noisy_value = int(round(center + self.rng.normal(0.0, 0.65)))
        return self._clamp(noisy_value, 0, 2)

    def _get_ground_truth_risk(self, state: tuple[int, int, int, int, int]) -> int:
        """Computes hidden risk class from magnitude, depth, and wave risk."""
        magnitude, depth, wave_risk, _, _ = state
        score = (1.4 * magnitude) + (1.2 * depth) + (1.6 * wave_risk)
        if score >= 5.0:
            return 2
        if score >= 2.8:
            return 1
        return 0

    def _transition_state(self, action: int) -> tuple[int, int, int, int, int]:
        """Applies simple stochastic transitions to the observable state."""
        magnitude, depth, wave_risk, confidence, time_index = self.state

        if action in (2, 3):
            return self.state

        time_index = self._clamp(time_index + 1, 0, 2)

        if action == 1:
            if self.rng.random() < 0.90:
                confidence = self._clamp(confidence + 1, 0, 2)

            if self.rng.random() < 0.80:
                if wave_risk < self.hidden_wave_risk:
                    wave_risk += 1
                elif wave_risk > self.hidden_wave_risk:
                    wave_risk -= 1
            elif self.rng.random() < 0.40:
                wave_risk += int(self.rng.choice([-1, 1]))
        else:
            if self.rng.random() < 0.35:
                confidence = self._clamp(confidence - 1, 0, 2)
            if self.rng.random() < 0.25:
                wave_risk += int(self.rng.choice([-1, 1]))

        if self.hidden_true_risk == 2 and action == 0 and self.rng.random() < 0.30:
            wave_risk += 1
        if self.hidden_true_risk == 0 and action == 0 and self.rng.random() < 0.25:
            wave_risk -= 1

        wave_risk = self._clamp(wave_risk, 0, 2)
        return magnitude, depth, wave_risk, confidence, time_index

    def _calculate_reward(
        self,
        action: int,
        previous_state: tuple[int, int, int, int, int],
        next_state: tuple[int, int, int, int, int],
        terminal: bool,
    ) -> tuple[float, bool, bool, bool]:
        """Computes reward and outcome flags for the selected action."""
        reward = 0.0
        alert_correct = False
        false_alert = False
        missed_alert = False

        if action in (0, 1):
            reward += self.config.reward_delay_per_step

        if action == 1:
            if self._is_uncertain(previous_state):
                reward += self.config.reward_smart_verify
            else:
                reward += self.config.penalty_unnecessary_verify

        if action == 2:
            if self.hidden_true_risk == 1:
                reward += self.config.reward_correct_regional_alert
                alert_correct = True
            elif self.hidden_true_risk == 2:
                reward += self.config.reward_partial_regional_on_high
            else:
                reward += self.config.penalty_false_regional_alert
                false_alert = True

        if action == 3:
            if self.hidden_true_risk == 2:
                reward += self.config.reward_correct_full_alert
                alert_correct = True
            elif self.hidden_true_risk == 1:
                reward += self.config.penalty_overreaction_full_on_medium
                false_alert = True
            else:
                reward += self.config.penalty_false_full_alert
                false_alert = True

        if terminal and action in (0, 1):
            if self.hidden_true_risk >= 1:
                reward += self.config.penalty_missed_dangerous_alert
                missed_alert = True
                if action == 0:
                    reward += self.config.penalty_late_wait_in_risk
            else:
                reward += self.config.reward_safe_no_alert_low_risk

        if terminal and action == 2 and self.hidden_true_risk == 2:
            reward += self.config.penalty_late_wait_in_risk

        if terminal and action == 3 and self.hidden_true_risk == 1 and next_state[4] == 2:
            reward += self.config.penalty_late_wait_in_risk

        return reward, alert_correct, false_alert, missed_alert

    def _is_uncertain(self, state: tuple[int, int, int, int, int]) -> bool:
        """Checks whether the observed state has meaningful uncertainty."""
        observed_wave_risk = state[2]
        confidence = state[3]
        wave_gap = abs(observed_wave_risk - self.hidden_wave_risk)
        return confidence == 0 or (confidence <= 1 and wave_gap >= 1)

    def _is_terminal(self, action: int, next_state: tuple[int, int, int, int, int]) -> bool:
        """Determines whether an episode should terminate."""
        if action in (2, 3):
            return True
        return next_state[4] >= 2

    @staticmethod
    def _clamp(value: int, low: int, high: int) -> int:
        """Clamps integer value within bounds."""
        return max(low, min(high, value))
